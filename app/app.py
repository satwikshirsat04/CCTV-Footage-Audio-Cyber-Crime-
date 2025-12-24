from flask import Flask, request, jsonify, render_template
import os, pickle, tempfile, shutil
import numpy as np
from PIL import Image
import cv2
import torch
import librosa
import ffmpeg
import tensorflow as tf

from transformers import (
    pipeline,
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification
)

# ==================================================
# APP SETUP
# ==================================================

app = Flask(__name__, template_folder="templates", static_folder="static")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================================================
# PATHS
# ==================================================

MODELS_DIR = os.path.join(BASE_DIR, "models")

SPEECH_MODEL_PATH = os.path.join(MODELS_DIR, "cctv_audio_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")
SOUND_MODEL_PATH = os.path.join(MODELS_DIR, "sound_effects_cnn_model.keras")

VISION_MODEL_NAME = "Nikeytas/videomae-crime-detector-ultra-v1"

# ==================================================
# LOAD MODELS (SAFE)
# ==================================================

print("\n🔄 Loading models...")

# Speech
try:
    speech_model = pickle.load(open(SPEECH_MODEL_PATH, "rb"))
    vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))
    print("✓ Speech + Vectorizer loaded")
except:
    speech_model, vectorizer = None, None
    print("✗ Speech model disabled")

# Sound
try:
    sound_model = tf.keras.models.load_model(
        SOUND_MODEL_PATH,
        compile=False,
        safe_mode=False
    )
    print("✓ Sound CNN loaded")
except:
    sound_model = None
    print("✗ Sound CNN disabled")

# Whisper
try:
    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        device=-1
    )
    print("✓ Whisper loaded")
except:
    whisper_pipe = None
    print("✗ Whisper disabled")

# Vision (CPU SAFE)
try:
    vision_processor = VideoMAEImageProcessor.from_pretrained(
        VISION_MODEL_NAME,
        do_rescale=False   # 🔥 prevents float64 blowup
    )
    vision_model = VideoMAEForVideoClassification.from_pretrained(
        VISION_MODEL_NAME
    )
    vision_model.eval()
    print("✓ Vision model loaded (VideoMAE)")
except Exception as e:
    vision_model = None
    print("✗ Vision model disabled:", e)

# ==================================================
# VISION SETTINGS (CRITICAL)
# ==================================================

WINDOW_SECONDS = 2
FRAMES_PER_WINDOW = 16        # ↓ from 16 (CPU safe)
MAX_WINDOWS = 2              # cap to avoid OOM
VISION_CACHE = {}

# ==================================================
# HELPERS
# ==================================================

def video_to_audio(video_path, out_audio):
    try:
        (
            ffmpeg
            .input(video_path)
            .output(out_audio, ac=1, ar=16000, acodec="pcm_s16le")
            .overwrite_output()
            .run(quiet=True)
        )
        return True
    except:
        print("⚠️ No audio stream found")
        return False


def extract_frames_window(cap, start_frame, fps):
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    stride = max(int(fps // 8), 1)  # skip frames to reduce load

    while len(frames) < 16:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(cap.get(cv2.CAP_PROP_POS_FRAMES)) + stride
        )

    return frames



def predict_vision_sliding(video_path):
    if vision_model is None:
        return 0.0, False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0, False

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    max_prob = 0.0

    try:
        for w in range(MAX_WINDOWS):
            cache_key = (video_path, w)
            if cache_key in VISION_CACHE:
                prob = VISION_CACHE[cache_key]
            else:
                start_frame = int(w * fps * WINDOW_SECONDS)
                frames = extract_frames_window(cap, start_frame, fps)

                if len(frames) < 4:
                    continue

                inputs = vision_processor(frames, return_tensors="pt")
                with torch.no_grad():
                    outputs = vision_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)

                prob = float(probs[0][1])
                VISION_CACHE[cache_key] = prob

            max_prob = max(max_prob, prob)

    except Exception as e:
        print("⚠️ Vision inference safe-fail:", e)

    finally:
        cap.release()

    return max_prob, max_prob >= 0.75


def predict_sound(audio_path):
    if sound_model is None:
        return 0.0, False

    y, sr = librosa.load(audio_path, sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel = librosa.power_to_db(mel)

    img = Image.fromarray(mel).resize((232, 231)).convert("RGB")
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)

    prob = float(np.max(sound_model.predict(arr, verbose=0)))
    return prob, prob >= 0.6


def predict_speech(audio_path):
    if whisper_pipe is None or speech_model is None:
        return "", False

    # 🔥 Load max 25 seconds only
    y, sr = librosa.load(audio_path, sr=16000, duration=25)

    tmp_wav = audio_path.replace(".wav", "_short.wav")
    librosa.output.write_wav(tmp_wav, y, sr)

    text = whisper_pipe(tmp_wav)["text"]

    vec = vectorizer.transform([text])
    pred = speech_model.predict(vec)[0]

    return text, bool(pred)


# ==================================================
# ROUTES
# ==================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze_video", methods=["POST"])
def analyze_video():
    video_rel = request.json.get("video_path")
    video_path = os.path.join(BASE_DIR, "static", video_rel)

    if not os.path.exists(video_path):
        return jsonify({"error": "Video not found"}), 404

    tmp = tempfile.mkdtemp()
    audio_path = os.path.join(tmp, "audio.wav")

    try:
        has_audio = video_to_audio(video_path, audio_path)

        vision_prob, vision_threat = predict_vision_sliding(video_path)

        if has_audio:
            sound_prob, sound_threat = predict_sound(audio_path)
            speech_text, speech_threat = predict_speech(audio_path)
        else:
            sound_prob, sound_threat = 0.0, False
            speech_text, speech_threat = "", False

        final_score = (
            0.55 * vision_prob +
            0.25 * sound_prob +
            0.20 * (1.0 if speech_threat else 0.0)
        )

        return jsonify({
            "vision_probability": vision_prob,
            "sound_probability": sound_prob,
            "speech_threat": speech_threat,
            "final_score": final_score,
            "final_threat": final_score >= 0.6,
            "transcription": speech_text,
            "message": "🚨 Crime Detected" if final_score >= 0.6 else "✅ Monitoring"
        })

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    print(f"\n🚀 Server running on port {port}")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,          # 🔥 correct
        use_reloader=False   # 🔥 VERY important
    )
