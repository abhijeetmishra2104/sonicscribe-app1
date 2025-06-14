import pandas as pd
from flask import Flask , request , jsonify , render_template
import whisper
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from flask_cors import CORS
load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
CORS(app)

# Load Whisper model
model = whisper.load_model("base") 

llm=ChatOpenAI(model="gpt-4o")
### Chatprompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","Extract the following structured details from the given clinical note: Name , Age/Gender,Medical History,Symptoms,Notes (Summarize any additional context or observations),Risk Prediction (based on symptoms and medical history),Possible Disease(You have to predict possible disease) , Recommendation (next steps for care or treatment , tell wheather the person should admitted to hospital or not)"),
        ("user","{input}")
    ]

)

output_parser=StrOutputParser()
chain=prompt|llm|output_parser



@app.route('/', methods=['GET', 'POST'])
def index():
    transcript = None
    if request.method == 'POST':
        audio = request.files['audio_file']
        if audio:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio.filename)
            audio.save(file_path)

            result = model.transcribe(file_path, task="translate")  # auto translates to English
            response=chain.invoke({"input":result["text"]})
            print(response)
            transcript = response
            
    return render_template('index.html', transcript=transcript)

@app.route('/process-audio-url', methods=['POST'])
def process_audio_url():
    data = request.get_json()
    audio_url = data.get('url')

    if not audio_url:
        return jsonify({"error": "No audio URL provided"}), 400

    # Download audio from URL
    import requests
    with requests.get(audio_url, stream=True) as r:
        with open("temp_audio.mp3", 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    result = model.transcribe("temp_audio.mp3", task="translate")
    response = chain.invoke({"input": result["text"]})
    return jsonify({"result": response})


