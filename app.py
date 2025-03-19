import streamlit as st
import base64
import whisper
import pickle
import numpy as np
import speech_recognition as sr
from twilio.rest import Client

# Twilio Credentials
TWILIO_SID = "AC6a78fa46ae875a666c71d6c49cc6c58e"
TWILIO_AUTH_TOKEN = "348b9b06f80fd94a9697e18e2afff3e9"
TWILIO_PHONE = "+19039123996"
ADMIN_PHONE = "+917498238505"

# Load AI Model
@st.cache_resource
def load_model():
    model = pickle.load(open("cctv_audio_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# Function to transcribe audio
def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result["text"]

# Function to detect threats
def detect_threat(text):
    input_text_vectorized = vectorizer.transform([text])
    prediction = model.predict(input_text_vectorized)
    return "Cyber Threat Detected!" if prediction[0] == 1 else "No Threat Detected."

# Function to send SMS alert
def send_alert(message):
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    client.messages.create(body=message, from_=TWILIO_PHONE, to=ADMIN_PHONE)
    st.warning("Alert Sent!")

# Function to capture audio
def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = recognizer.listen(source)
    return audio

# Function to encode image to Base64
def get_base64_of_image(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Function to set background image from local file
def set_background_local(image_path):
    base64_str = get_base64_of_image(image_path)
    page_bg = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{base64_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        # background: rgba(0, 0, 0, 10.5);
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

# ✅ Set background from local image file
set_background_local("cyber.jpeg")  

# Streamlit App UI
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

    # ✅ Fixed issue by using proper if-else block
    if "Threat" in threat_status:
        st.warning(threat_status)
    else:
        st.success(threat_status)

    if "Cyber Threat Detected!" in threat_status:
        send_alert(f"ALERT: {threat_status}")
