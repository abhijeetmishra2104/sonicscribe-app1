from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import json
import base64
import mimetypes
import pickle
import traceback
import pandas as pd
from dotenv import load_dotenv
from urllib.request import urlopen
from datetime import datetime
import gdown

# === Load environment variables ===
load_dotenv()

# === LangChain Setup (Google Gemini) ===
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.messages import HumanMessage

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env or host config.")

# Google retires model ids periodically — a retired id returns 404 NOT_FOUND at
# request time, so this is overridable without a redeploy. Check the live list:
#   curl https://generativelanguage.googleapis.com/v1beta/models -H "x-goog-api-key: $GEMINI_API_KEY"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# Transcription client — same model, kept separate from the analysis chains so
# audio and text calls can be tuned independently. No temperature: Gemini 3.x
# flash models use fixed sampling defaults and warn if one is passed.
transcriber = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GEMINI_API_KEY)

# LangChain tracing env vars are optional — only set them if provided,
# otherwise os.environ[...] = None crashes at import time.
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "sonicscribe")

# === Flask App Setup ===
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# === Model4 (risk prediction) setup ===
model4_path = os.path.join(os.path.dirname(__file__), 'model4.pkl')
if not os.path.exists(model4_path):
    print("Downloading model4.pkl from Google Drive...")
    gdown.download("https://drive.google.com/uc?id=1lXRkB3qWgoqwXpo4E12mQtTZtdKObZ4e", model4_path, quiet=False)

with open(model4_path, 'rb') as f:
    model4 = pickle.load(f)

# === Utility Functions ===
def save_uploaded_audio(audio, filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    audio.save(path)
    return path

def download_audio_from_url(url, filename="downloaded_audio.mp3"):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(path, 'wb') as f:
        f.write(urlopen(url).read())
    return path

def _message_text(content):
    """Flatten a LangChain message body to plain text.

    Gemini returns `content` as a list of typed blocks (text, thought signatures,
    ...) rather than a bare string, so pull out and join the text blocks.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts).strip()
    return str(content)


def transcribe_audio(file_path):
    """Transcribe audio with Gemini's native multimodal audio support."""
    mime_type = mimetypes.guess_type(file_path)[0] or "audio/mpeg"
    with open(file_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    message = HumanMessage(content=[
        {"type": "text", "text": "Transcribe this audio verbatim. Output only the transcript text."},
        {"type": "media", "mime_type": mime_type, "data": audio_b64},
    ])
    return _message_text(transcriber.invoke([message]).content)

# === LangChain Prompts and Chains (Gemini-backed) ===
parser = JsonOutputParser()

prompt_1 = ChatPromptTemplate.from_messages([
    ("system", """You are a medical assistant. Extract this structured JSON from the user's clinical note.
Respond with ONLY valid JSON, no markdown formatting, no code fences, no extra text.
{{
  "name": "",
  "age_gender": "",
  "medical_history": [],
  "symptoms": [],
  "notes": "",
  "risk_prediction": "",
  "possible_disease": [],
  "recommendation": {{
    "next_steps": "",
    "should_be_admitted": true
  }}
}}"""),
    ("user", "{input}")
])


prompt_2 = ChatPromptTemplate.from_messages([
    ("system", """You are a professional healthcare assistant. The user will enter their symptoms.
Based on the symptoms, provide the following structured JSON.
Respond with ONLY valid JSON, no markdown formatting, no code fences, no extra text.

{{
  "probable_conditions": ["Condition 1", "Condition 2", "Condition 3"],
  "triage_level": "Emergency / Urgent / Non-Urgent",
  "specialist_to_consult": "Specialist Name",
  "advice": "Always recommend consulting a real doctor."
}}"""),
    ("user", "{input}")
])

llm_1 = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GEMINI_API_KEY)
llm_2 = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GEMINI_API_KEY)

chain_1 = prompt_1 | llm_1 | parser
chain_2 = prompt_2 | llm_2 | parser

# === Routes ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze-note', methods=['POST'])
def analyze_note():
    try:
        audio = request.files.get('audio_file')
        audio_url = request.json.get("url") if request.is_json else None

        if not audio and not audio_url:
            return jsonify({"success": False, "error": "Audio file or URL missing"}), 400

        if audio:
            file_path = save_uploaded_audio(audio, audio.filename)
            original_name = audio.filename
        else:
            file_path = download_audio_from_url(audio_url)
            original_name = "downloaded_audio.mp3"

        transcript = transcribe_audio(file_path)
        structured_data = chain_1.invoke({"input": transcript})
        triage_data = chain_2.invoke({"input": transcript})

        return jsonify({
            "success": True,
            "file": {
                "originalName": original_name,
                "uploadedAt": datetime.utcnow().isoformat(),
                "url": audio_url
            },
            "analysis": {
                "transcript": transcript,
                "structured": structured_data,
                "triage": triage_data
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/analyze-symptoms', methods=['POST'])
def analyze_symptoms():
    try:
        text_input = request.form.get('text_input')
        audio_file = request.files.get('audio_file')
        transcript = None

        if text_input:
            result = chain_2.invoke({"input": text_input})
        elif audio_file and audio_file.filename != '':
            file_path = save_uploaded_audio(audio_file, audio_file.filename)
            transcript = transcribe_audio(file_path)
            result = chain_2.invoke({"input": transcript})
        else:
            return jsonify({"success": False, "error": "No input provided"}), 400

        return jsonify({"success": True, "response": result, "transcript": transcript})

    except Exception as e:
        traceback.print_exc()  # full stack trace lands in your Flask/host logs
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        features = pd.DataFrame([[
            int(data['age']),
            int(data['gender']),
            int(data['primaryDiagnosis']),
            int(data['numProcedures']),
            int(data['daysInHospital']),
            int(data['comorbidityScore']),
            int(data['dischargeTo'])
        ]], columns=[
            "age", "gender", "primary_diagnosis", "num_procedures",
            "days_in_hospital", "comorbidity_score", "discharge_to"
        ])

        risk = model4.predict_proba(features)[0][1] * 100
        decision = "Hospitalize Patient" if risk > 50 else "No Hospitalization Needed"

        return jsonify({"success": True, "risk": risk, "decision": decision})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# === Start Server ===
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)