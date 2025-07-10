from flask import Flask, request, jsonify, send_from_directory
import os
import pickle
from dotenv import load_dotenv
from twilio.rest import Client
from transformers import pipeline

app = Flask(__name__, static_folder='static')

load_dotenv()
model = pickle.load(open("cctv_audio_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
whisper_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/video', methods=['GET'])
def video():
    return "CCTV Video YOLO Model"
    

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        audio_file = request.files['audio']
        audio_path = "uploaded_audio.wav"
        audio_file.save(audio_path)

        transcription = whisper_pipe(audio_path)['text']
        print(f"Transcription: {transcription}")

        # Threat detection using your ML model
        input_text_vectorized = vectorizer.transform([transcription])
        is_threat = model.predict(input_text_vectorized)[0] == 1

        if is_threat:
            send_twilio_alert(transcription)

        return jsonify({
            'transcription': transcription,
            'is_threat': int(is_threat),  # Convert to integer (1 or 0)
            'threat_status': "🚨 Cyber Threat Detected!" if is_threat else "✅ No Threat"
        })

    except Exception as e:
        print(f"Error during processing: {e}")
        return jsonify({'error': str(e)}), 500

def send_twilio_alert(message):
    client = Client(
        os.getenv("TWILIO_API_KEY"),
        os.getenv("TWILIO_API_SECRET"),
        os.getenv("TWILIO_SID")
    )
    client.messages.create(
        body=f"🚨THREAT ALERT: {message}",
        from_=os.getenv("TWILIO_PHONE"),
        to=os.getenv("ADMIN_PHONE")
    )

if __name__ == '__main__':
    app.debug = True
    app.run(port=5000)
