from flask import Flask, request, jsonify, render_template
import os
import pickle
import tempfile
import shutil
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import ffmpeg
import tensorflow as tf
import soundfile as sf
import traceback
import cv2
import torch
import time

from transformers import (
    pipeline,
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification
)
from dotenv import load_dotenv

# --------------------------------------------------
# APP SETUP
# --------------------------------------------------

app = Flask(__name__, template_folder='templates', static_folder='static')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

load_dotenv()

# --------------------------------------------------
# MODEL PATHS
# --------------------------------------------------

MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

SPEECH_MODEL_PATH = os.path.join(MODELS_DIR, 'cctv_audio_model.pkl')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'vectorizer.pkl')
SOUND_MODEL_PATH = os.path.join(MODELS_DIR, 'sound_effects_cnn_model.keras')

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

print("\n🔄 Loading models...")

# Speech model
try:
    speech_model = pickle.load(open(SPEECH_MODEL_PATH, 'rb'))
    vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
    print("✓ Speech + Vectorizer loaded")
except Exception as e:
    print("✗ Speech model load failed:", e)
    speech_model, vectorizer = None, None

# Sound CNN
sound_model = None
try:
    sound_model = tf.keras.models.load_model(SOUND_MODEL_PATH, compile=False)
    print("✓ Sound CNN loaded")
except Exception as e:
    print("✗ Sound CNN load failed:", e)

# Whisper
try:
    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en"
    )
    print("✓ Whisper loaded")
except Exception as e:
    whisper_pipe = None
    print("✗ Whisper load failed:", e)


VISION_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "videomae-crime-detector-ultra-v1"
)

vision_processor = VideoMAEImageProcessor.from_pretrained(
    VISION_MODEL_PATH,
    local_files_only=True
)

vision_model = VideoMAEForVideoClassification.from_pretrained(
    VISION_MODEL_PATH,
    local_files_only=True
)

vision_model.eval()

print("Vision model loaded from:", VISION_MODEL_PATH)

# --------------------------------------------------
# VISION CACHE
# --------------------------------------------------

vision_cache = {}  
# key: (video_path, window_index) → probability

WINDOW_SECONDS = 2
FRAMES_PER_WINDOW = 16

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def extract_frames_window(video_path, start_sec, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames


def predict_vision_sliding(video_path):
    if vision_model is None:
        return 0.0, False

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps
    cap.release()

    max_prob = 0.0
    window_idx = 0

    for start in np.arange(0, duration, WINDOW_SECONDS):
        cache_key = (video_path, window_idx)

        if cache_key in vision_cache:
            prob = vision_cache[cache_key]
        else:
            frames = extract_frames_window(video_path, start, FRAMES_PER_WINDOW)
            if len(frames) < 8:
                continue

            inputs = vision_processor(frames, return_tensors="pt")
            with torch.no_grad():
                outputs = vision_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            prob = probs[0][1].item()  # violent class
            vision_cache[cache_key] = prob

        max_prob = max(max_prob, prob)
        window_idx += 1

    return max_prob, max_prob >= 0.75


def video_to_audio(video_path, out_audio):
    (
        ffmpeg
        .input(video_path)
        .output(out_audio, ac=1, ar=16000, acodec='pcm_s16le')
        .overwrite_output()
        .run(quiet=True)
    )


def predict_speech(audio_path):
    if whisper_pipe is None or speech_model is None:
        return "", False

    text = whisper_pipe(audio_path)["text"]
    vec = vectorizer.transform([text])
    pred = speech_model.predict(vec)[0]
    return text, bool(pred)


def predict_sound(audio_path):
    if sound_model is None:
        return 0.0, False

    y, sr = librosa.load(audio_path, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    img = Image.fromarray(S_db).resize((232, 231)).convert("RGB")
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    probs = sound_model.predict(arr, verbose=0)[0]
    prob = float(np.max(probs))
    return prob, prob >= 0.6

# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    data = request.json
    video_rel = data.get("video_path")

    video_path = os.path.join(BASE_DIR, "static", video_rel)
    if not os.path.exists(video_path):
        return jsonify({"error": "Video not found"}), 404

    tmpdir = tempfile.mkdtemp()
    audio_path = os.path.join(tmpdir, "audio.wav")

    try:
        video_to_audio(video_path, audio_path)

        # --- Predictions ---
        vision_prob, vision_threat = predict_vision_sliding(video_path)
        sound_prob, sound_threat = predict_sound(audio_path)
        speech_text, speech_threat = predict_speech(audio_path)

        # --- Fusion ---
        final_score = (
            0.55 * vision_prob +
            0.25 * sound_prob +
            0.20 * (1.0 if speech_threat else 0.0)
        )

        final_threat = final_score >= 0.6

        return jsonify({
            "vision_probability": vision_prob,
            "vision_threat": vision_threat,
            "sound_probability": sound_prob,
            "sound_threat": sound_threat,
            "speech_threat": speech_threat,
            "transcription": speech_text,
            "final_score": final_score,
            "final_threat": final_threat,
            "message": "🚨 Crime Detected" if final_threat else "✅ Monitoring"
        })

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    print("\n🚀 Server running")
    app.run(host="0.0.0.0", port=5000, debug=True)
