from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import pickle
import pandas as pd
from dotenv import load_dotenv
import gdown
from urllib.request import urlopen

# Load environment variables
load_dotenv()

# OpenAI setup
from openai import OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# LangChain environment variables
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# LangChain setup
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangChain chains
llm_1 = ChatOpenAI(model="gpt-4o")
prompt_1 = ChatPromptTemplate.from_messages([
    ("system", """Extract the following structured details from the given clinical note: Name , Age/Gender, Medical History, Symptoms, Notes (Summarize any additional context or observations), Risk Prediction (based on symptoms and medical history), Possible Disease (You have to predict possible disease), Recommendation (next steps for care or treatment , tell whether the person should be admitted to hospital or not)"""),
    ("user", "{input}")
])
chain_1 = prompt_1 | llm_1 | StrOutputParser()

llm_2 = ChatOpenAI(model="gpt-4o")
prompt_2 = ChatPromptTemplate.from_messages([
    ("system", '''You are a professional healthcare assistant. The user will enter their symptoms. 
                  Based on the symptoms, provide:
                  1. Probable conditions (up to 3).
                  2. Triage level: Emergency / Urgent / Non-Urgent.
                  3. Specialist to consult.
                  Always advise consulting a real doctor.'''),
    ("user", "{input}")
])
chain_2 = prompt_2 | llm_2 | StrOutputParser()

# Load ML model for risk prediction
model4_path = os.path.join(os.path.dirname(__file__), 'model4.pkl')
if not os.path.exists(model4_path):
    print("Downloading model4.pkl from Google Drive...")
    gdown.download("https://drive.google.com/uc?id=1lXRkB3qWgoqwXpo4E12mQtTZtdKObZ4e", model4_path, quiet=False)

with open(model4_path, 'rb') as f:
    model4 = pickle.load(f)

# Flask app setup
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

# === Endpoint 1: Transcription + Diagnosis (App1 logic) ===
@app.route('/api/analyze-note', methods=['POST'])
def analyze_note():
    audio = request.files.get('audio_file')
    audio_url = request.json.get("url") if request.is_json else None

    if not audio and not audio_url:
        return jsonify({"error": "Audio file or URL missing"}), 400

    if audio:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio.filename)
        audio.save(file_path)
    elif audio_url:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'downloaded_audio.mp3')
        with open(file_path, 'wb') as f:
            f.write(urlopen(audio_url).read())

    # Transcribe
    with open(file_path, "rb") as f:
        transcript_data = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )

    response = chain_1.invoke({"input": transcript_data.text})
    return jsonify({"transcript": transcript_data.text, "response": response})

# === Endpoint 2: Text or Audio Symptom Analysis (App2 logic) ===
@app.route('/api/analyze-symptoms', methods=['POST'])
def analyze_symptoms():
    text_input = request.form.get('text_input')
    audio_file = request.files.get('audio_file')
    transcript = None

    if text_input:
        response = chain_2.invoke({"input": text_input})
    elif audio_file and audio_file.filename != '':
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
        audio_file.save(audio_path)
        with open(audio_path, "rb") as f:
            transcript_data = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
            transcript = transcript_data.text
            response = chain_2.invoke({"input": transcript})
    else:
        return jsonify({"error": "No input provided"}), 400

    return jsonify({"response": response, "transcript": transcript})

# === Endpoint 3: Hospital Risk Prediction (App4 logic) ===
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
        ]], columns=["age", "gender", "primary_diagnosis", "num_procedures", "days_in_hospital", "comorbidity_score", "discharge_to"])

        risk = model4.predict_proba(features)[0][1] * 100
        decision = "Hospitalize Patient" if risk > 50 else "No Hospitalization Needed"

        return jsonify({"risk": risk, "decision": decision})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
