import streamlit as st
import base64
import pickle
import numpy as np
import speech_recognition as sr
from twilio.rest import Client
from dotenv import load_dotenv
import os


# Twilio Credentials (Make sure these are secure in real deployment)
load_dotenv()

TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_API_KEY = os.getenv("TWILIO_API_KEY")
TWILIO_API_SECRET = os.getenv("TWILIO_API_SECRET")
TWILIO_PHONE = os.getenv("TWILIO_PHONE")
ADMIN_PHONE = os.getenv("ADMIN_PHONE")

# Load AI Model
@st.cache_resource
def load_model():
    model = pickle.load(open("cctv_audio_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# Transcribe audio using Google Speech Recognition
def transcribe_audio(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand audio."
    except sr.RequestError as e:
        return f"Request error: {e}"

# Detect cyber threat from transcribed text
def detect_threat(text):
    input_text_vectorized = vectorizer.transform([text])
    prediction = model.predict(input_text_vectorized)
    return "🚨Cyber Threat Detected!" if prediction[0] == 1 else "No Threat Detected."

# Send SMS alert via Twilio
def send_alert(message):
    try:
        client = Client(TWILIO_API_KEY, TWILIO_API_SECRET, TWILIO_SID)
        alert_message = (
            "🚨 THREAT ALERT 🚨\n"
            "-----------------------------------\n"
            f"{message}\n"
            "-----------------------------------\n"
            "⚠️ Please take immediate action."
        )
        client.messages.create(body=alert_message, from_=TWILIO_PHONE, to=ADMIN_PHONE)
        st.warning("🚨 Alert Sent via SMS!")
    except Exception as e:
        st.error(f"Failed to send alert: {e}")




# Capture live audio from microphone
def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = recognizer.listen(source)
    return audio

# Encode background image to Base64
def get_base64_of_image(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Set background image from local file
def set_background_local(image_path):
    base64_str = get_base64_of_image(image_path)
    page_bg = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{base64_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

# Set background
set_background_local("cyber.jpeg")  

# Streamlit UI
st.title("🔊 AI-Powered Cyber Threat Detection")
st.markdown("Speak into the microphone to analyze speech for threats.")

if st.button("🎤 Start Recording"):
    audio = record_audio()
    with open("audio.wav", "wb") as f:
        f.write(audio.get_wav_data())

    st.success("Audio Captured Successfully!")

    transcribed_text = transcribe_audio("audio.wav")
    st.subheader("Transcribed Text:")
    st.write(transcribed_text)

    threat_status = detect_threat(transcribed_text)
    st.subheader("Threat Status:")

    if "Threat" in threat_status:
        st.warning(threat_status)
    else:
        st.success(threat_status)

    if "Cyber Threat Detected!" in threat_status:
        send_alert(f"ALERT: {threat_status}")
