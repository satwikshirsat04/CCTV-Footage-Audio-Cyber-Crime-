import pyaudio
import wave
import whisper
import pickle

# Record audio from microphone (CCTV or PC or LAPTOP)
# def record_audio(filename="audio.wav", duration=5):
#     p = pyaudio.PyAudio()
#     stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
    
#     frames = []
#     print("Recording...")
#     for _ in range(0, int(44100 / 1024 * duration)):
#         data = stream.read(1024)
#         frames.append(data)

#     stream.stop_stream()
#     stream.close()
#     p.terminate()

#     with wave.open(filename, 'wb') as wf:
#         wf.setnchannels(1)
#         wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
#         wf.setframerate(44100)
#         wf.writeframes(b''.join(frames))
    
#     print("Recording saved as", filename)
# record_audio()

# Transcribe the audio from speech
def transcribe_audio(filename="audio.wav"):
    model = whisper.load_model("base")
    result = model.transcribe(filename)
    return result["text"]

text = transcribe_audio()
print("\n---------x---------------------------x-----------")
print("\n> Transcribed Text:", text,"\n")


# Check for cyber threats
def detect_threat(text):
    model = pickle.load(open("cctv_audio_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

    input_text_vectorized = vectorizer.transform([text])
    prediction = model.predict(input_text_vectorized)
    
    return "Cyber Threat Detected!" if prediction[0] == 1 else "No Threat Detected."

# Check on new speech input
threat_status = detect_threat(text)
print("-----------------------------------------------")
print("\n> Threat Status:", threat_status)
print("\n---------x---------------------------x-----------\n")


# SMS Alert if Cyber Threat Detected!
# from twilio.rest import Client

# TWILIO_SID = "your_twilio_sid"
# TWILIO_AUTH_TOKEN = "your_twilio_auth_token"
# TWILIO_PHONE = "+1234567890"  # Your Twilio phone number
# ADMIN_PHONE = "+917498238505"  # Your phone number

# def send_alert(message):
#     client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
#     client.messages.create(body=message, from_=TWILIO_PHONE, to=ADMIN_PHONE)
#     print("Alert Sent!")

# if "Cyber Threat Detected!" in threat_status:
#     send_alert(f"ALERT: {threat_status}")


