from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
import os
import sys
import json
import logging
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
# flash-lite over flash: each /analyze-note upload spends three calls (transcribe
# + both chains) and the free tier caps gemini-3.6-flash at 20, which one person
# testing exhausts in minutes. Quota is tracked per model, so this bucket is separate.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

# Every Gemini call must fail *inside* gunicorn's --timeout (see Dockerfile) or
# the worker is SIGKILLed mid-request and the caller gets gunicorn's HTML error
# page instead of our JSON. The SDK retries 429/5xx with exponential backoff, so
# an exhausted quota otherwise stacks retries until the worker dies -- cap both.
GEMINI_TIMEOUT = float(os.getenv("GEMINI_TIMEOUT", "45"))
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "1"))

_client_opts = dict(
    google_api_key=GEMINI_API_KEY,
    timeout=GEMINI_TIMEOUT,
    max_retries=GEMINI_MAX_RETRIES,
)

# Transcription client — same model, kept separate from the analysis chains so
# audio and text calls can be tuned independently. No temperature: Gemini 3.x
# flash models use fixed sampling defaults and warn if one is passed.
transcriber = ChatGoogleGenerativeAI(model=GEMINI_MODEL, **_client_opts)

# Tracing is opt-in via LANGCHAIN_TRACING_V2, not merely "a key is present".
# A stale or revoked key otherwise switches on an exporter that 403s on every
# run ("Failed to multipart ingest runs: ... 403 Forbidden"), which floods the
# logs and burns background CPU without ever surfacing in the request path.
# load_dotenv already put the key in os.environ, so nothing to copy across.
if os.getenv("LANGCHAIN_TRACING_V2", "").strip().lower() in ("1", "true", "yes") \
        and os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "sonicscribe")
else:
    # Clear both spellings so an inherited value can't re-enable the exporter.
    os.environ.pop("LANGCHAIN_TRACING_V2", None)
    os.environ.pop("LANGSMITH_TRACING", None)

# === Flask App Setup ===
# Without an explicit config, app.logger records are dropped under gunicorn and
# the traceback never reaches the host's log stream -- which is what made this
# route's failures invisible in the Render logs.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stderr,
)

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
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

# === Errors ===
class StageError(Exception):
    """A failure we can name. `stage` tells the caller which step broke, so the
    JSON error says "transcribe: ..." instead of surfacing a downstream SDK
    message from a stage that was only the messenger."""

    def __init__(self, stage, message, status=502):
        super().__init__(message)
        self.stage = stage
        self.status = status


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

    def ask(instruction):
        return _message_text(transcriber.invoke([HumanMessage(content=[
            {"type": "text", "text": instruction},
            {"type": "media", "mime_type": mime_type, "data": audio_b64},
        ])]).content)

    text = ask("Transcribe this audio verbatim. Output only the transcript text.")

    # A thinking model can spend its entire output budget on reasoning and return
    # zero text parts (finish_reason=STOP, output_token_details.reasoning == all
    # of output_tokens) -- observed on gemini-3.6-flash with short//ambiguous
    # clips. That yields "" here. Nudge once for a text-only answer.
    if not text.strip():
        text = ask(
            "Transcribe the speech in this audio. Reply with the transcript text "
            "only -- no reasoning, no preamble. If there is no speech, reply NO_SPEECH."
        )

    # Never hand "" to the chains: LangChain drops a HumanMessage with empty
    # content, leaving the request with no contents at all, and the Gemini SDK
    # then raises the unrelated "contents are required." three lines later.
    if not text.strip() or text.strip() == "NO_SPEECH":
        raise StageError(
            "transcribe",
            "Transcription returned no text for this audio (the model produced no "
            "transcript). The file may contain no intelligible speech.",
        )
    return text

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

llm_1 = ChatGoogleGenerativeAI(model=GEMINI_MODEL, **_client_opts)
llm_2 = ChatGoogleGenerativeAI(model=GEMINI_MODEL, **_client_opts)

chain_1 = prompt_1 | llm_1 | parser
chain_2 = prompt_2 | llm_2 | parser

# === Routes ===
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/healthz')
def healthz():
    """Which code is actually serving, and with what config.

    Added because confirming a deploy otherwise meant inferring the running
    commit from the *shape* of an error response. Render injects the git
    metadata; `model` reports the id in effect after any dashboard override,
    which is the setting most likely to differ from what the source says.
    """
    return jsonify({
        "ok": True,
        "commit": (os.getenv("RENDER_GIT_COMMIT") or "unknown")[:12],
        "branch": os.getenv("RENDER_GIT_BRANCH") or "unknown",
        "model": GEMINI_MODEL,
        "gemini_timeout": GEMINI_TIMEOUT,
        "gemini_max_retries": GEMINI_MAX_RETRIES,
        "langsmith_tracing": bool(os.environ.get("LANGCHAIN_TRACING_V2")),
    })

def _run_stage(stage, fn, *args, **kwargs):
    """Run one pipeline step, tagging any failure with the step that raised it."""
    try:
        return fn(*args, **kwargs)
    except StageError:
        raise
    except Exception as e:
        app.logger.exception("analyze-note failed during stage=%s", stage)
        raise StageError(stage, f"{type(e).__name__}: {e}") from e


@app.route('/api/analyze-note', methods=['POST'])
def analyze_note():
    audio = request.files.get('audio_file')
    audio_url = request.json.get("url") if request.is_json else None

    if not audio and not audio_url:
        return jsonify({"success": False, "error": "Audio file or URL missing"}), 400

    if audio:
        file_path = _run_stage("save_upload", save_uploaded_audio, audio, audio.filename)
        original_name = audio.filename
    else:
        file_path = _run_stage("download", download_audio_from_url, audio_url)
        original_name = "downloaded_audio.mp3"

    transcript = _run_stage("transcribe", transcribe_audio, file_path)
    structured_data = _run_stage("structured_analysis", chain_1.invoke, {"input": transcript})
    triage_data = _run_stage("triage_analysis", chain_2.invoke, {"input": transcript})

    # Response contract consumed by the Next.js /api/upload route -- do not reshape.
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

# === Error handlers ===
# Every error out of this app is JSON. Flask would otherwise answer an unhandled
# exception -- or an aborted 400/413 raised inside request parsing, outside any
# view's try block -- with an HTML page, which the Next.js client cannot read.
@app.errorhandler(StageError)
def handle_stage_error(e):
    app.logger.error("stage=%s failed: %s", e.stage, e)
    return jsonify({"success": False, "stage": e.stage,
                    "error": f"{e.stage}: {e}"}), e.status


@app.errorhandler(HTTPException)
def handle_http_exception(e):
    return jsonify({"success": False, "error": e.description}), e.code


@app.errorhandler(Exception)
def handle_unexpected_error(e):
    app.logger.exception("Unhandled exception")
    traceback.print_exc()
    return jsonify({"success": False, "error": f"{type(e).__name__}: {e}"}), 500


# === Start Server ===
if __name__ == '__main__':
    # 8080 matches NEXT_PUBLIC_API_BASE_URL in the web-app; hosts that inject
    # their own $PORT (Render, Heroku) override it. Avoid 5001 -- macOS AirPlay
    # and IPFS Desktop both squat on it.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))