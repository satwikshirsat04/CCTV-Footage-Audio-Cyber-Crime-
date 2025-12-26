from flask import Flask, request, jsonify, render_template
import os, pickle, tempfile, shutil, threading
import numpy as np
from PIL import Image
import cv2
import torch
import librosa
import ffmpeg
import tensorflow as tf
import soundfile as sf

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

# Updated Vision Model - UCF Crime Detection
VISION_MODEL_NAME = "OPear/videomae-large-finetuned-UCF-Crime"
# If you have local model on D drive, set this path:
LOCAL_VISION_MODEL_PATH = "D:/models/videomae-large-finetuned-UCF-Crime"  # Adjust path as needed

# ==================================================
# CRIME CLASSES (UCF-Crime Dataset)
# ==================================================

CRIME_CLASSES = {
    0: "Abuse",
    1: "Arrest", 
    2: "Arson",
    3: "Assault",
    4: "Burglary",
    5: "Explosion",
    6: "Fighting",
    7: "Normal Videos",
    8: "Road Accidents",
    9: "Robbery",
    10: "Shooting",
    11: "Shoplifting",
    12: "Stealing",
    13: "Vandalism"
}

# Non-crime class
NON_CRIME_CLASS = 7  # "Normal Videos"

# ==================================================
# GLOBALS (SAFE)
# ==================================================

speech_model = vectorizer = None
sound_model = None
whisper_pipe = None
vision_model = vision_processor = None

MODELS_LOADED = False
MODEL_LOCK = threading.Lock()
INFERENCE_LOCK = threading.Lock()

# Limits (CRITICAL)
MAX_VIDEO_SIZE_MB = 50
MAX_VIDEO_FRAMES = 32
WINDOW_SECONDS = 2
MAX_WINDOWS = 2

# ==================================================
# LAZY MODEL LOADER (THREAD SAFE)
# ==================================================

def load_models():
    global MODELS_LOADED
    global speech_model, vectorizer
    global sound_model, whisper_pipe
    global vision_model, vision_processor

    if MODELS_LOADED:
        return

    with MODEL_LOCK:
        if MODELS_LOADED:
            return

        print("\n🔄 Lazy loading models (once)...")

        # Speech
        try:
            speech_model = pickle.load(open(SPEECH_MODEL_PATH, "rb"))
            vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))
            print("✓ Speech model loaded")
        except:
            speech_model = vectorizer = None
            print("✗ Speech disabled")

        # Sound
        try:
            sound_model = tf.keras.models.load_model(
                SOUND_MODEL_PATH, compile=False, safe_mode=False
            )
            print("✓ Sound CNN loaded")
        except:
            sound_model = None
            print("✗ Sound disabled")

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

        # Vision - Try local first, then HuggingFace
        try:
            if os.path.exists(LOCAL_VISION_MODEL_PATH):
                print(f"Loading vision model from local path: {LOCAL_VISION_MODEL_PATH}")
                vision_processor = VideoMAEImageProcessor.from_pretrained(
                    LOCAL_VISION_MODEL_PATH, do_rescale=False
                )
                vision_model = VideoMAEForVideoClassification.from_pretrained(
                    LOCAL_VISION_MODEL_PATH
                )
            else:
                print(f"Loading vision model from HuggingFace: {VISION_MODEL_NAME}")
                vision_processor = VideoMAEImageProcessor.from_pretrained(
                    VISION_MODEL_NAME, do_rescale=False
                )
                vision_model = VideoMAEForVideoClassification.from_pretrained(
                    VISION_MODEL_NAME
                )
            
            vision_model.eval()
            print("✓ UCF-Crime Vision model loaded")
        except Exception as e:
            vision_model = None
            print("✗ Vision disabled:", e)

        MODELS_LOADED = True


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
        return False


def extract_frames_uniform(video_path, num_frames=16):
    """Extract frames uniformly from video for better detection"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        num_frames = max(total_frames, 1)
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    return frames


def predict_vision(video_path):
    """Predict crime type using UCF-Crime model"""
    if vision_model is None:
        return {
            "crime_detected": False,
            "crime_type": "Normal Videos",
            "confidence": 0.0,
            "all_probabilities": {}
        }

    try:
        # Extract 16 frames uniformly
        frames = extract_frames_uniform(video_path, num_frames=16)
        
        if len(frames) < 4:
            return {
                "crime_detected": False,
                "crime_type": "Normal Videos",
                "confidence": 0.0,
                "all_probabilities": {}
            }

        # Process frames
        inputs = vision_processor(frames, return_tensors="pt")
        
        with torch.no_grad():
            outputs = vision_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
        
        # Get predicted class
        predicted_class_idx = torch.argmax(probs).item()
        confidence = float(probs[predicted_class_idx])
        crime_type = CRIME_CLASSES[predicted_class_idx]
        
        # Crime is detected if it's NOT "Normal Videos"
        crime_detected = predicted_class_idx != NON_CRIME_CLASS
        
        # Get all probabilities for debugging
        all_probs = {CRIME_CLASSES[i]: float(probs[i]) for i in range(len(CRIME_CLASSES))}
        
        return {
            "crime_detected": crime_detected,
            "crime_type": crime_type,
            "confidence": confidence,
            "all_probabilities": all_probs
        }
        
    except Exception as e:
        print(f"Vision prediction error: {e}")
        return {
            "crime_detected": False,
            "crime_type": "Error",
            "confidence": 0.0,
            "all_probabilities": {}
        }


def predict_sound(audio_path):
    if sound_model is None:
        return 0.0

    try:
        y, sr = librosa.load(audio_path, sr=22050)
        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr))
        img = Image.fromarray(mel).resize((232, 231)).convert("RGB")
        arr = np.expand_dims(np.array(img) / 255.0, axis=0)

        return float(np.max(sound_model.predict(arr, verbose=0)))
    except:
        return 0.0


def predict_speech(audio_path):
    if whisper_pipe is None or speech_model is None:
        return "", False

    try:
        y, sr = librosa.load(audio_path, sr=16000, duration=25)
        tmp = audio_path.replace(".wav", "_short.wav")
        sf.write(tmp, y, sr)

        text = whisper_pipe(tmp).get("text", "").strip()
        if not text:
            return "", False

        vec = vectorizer.transform([text])
        return text, bool(speech_model.predict(vec)[0])
    except:
        return "", False


# ==================================================
# ROUTES
# ==================================================

@app.route("/health")
def health():
    return "OK", 200


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze_video", methods=["POST"])
def analyze_video():
    load_models()

    video_rel = request.json.get("video_path")
    video_path = os.path.join(BASE_DIR, "static", video_rel)

    if not os.path.exists(video_path):
        return jsonify({"error": "Video not found"}), 404

    if os.path.getsize(video_path) / (1024 * 1024) > MAX_VIDEO_SIZE_MB:
        return jsonify({"error": "Video too large"}), 400

    with INFERENCE_LOCK:
        tmp = tempfile.mkdtemp()
        audio_path = os.path.join(tmp, "audio.wav")

        try:
            # Extract audio
            has_audio = video_to_audio(video_path, audio_path)
            
            # Vision Analysis (NEW UCF-Crime Model)
            vision_result = predict_vision(video_path)

            # Audio Analysis
            if has_audio:
                sound_prob = predict_sound(audio_path)
                speech_text, speech_threat = predict_speech(audio_path)
            else:
                sound_prob, speech_text, speech_threat = 0.0, "", False

            # Multi-modal decision
            # If vision detects specific crime with confidence > 0.3, trust it
            # Otherwise use combined score
            if vision_result["crime_detected"] and vision_result["confidence"] > 0.3:
                final_threat = True
                threat_reason = f"{vision_result['crime_type']} detected"
            else:
                # Fallback to multi-modal scoring
                final_score = (
                    0.55 * (1.0 if vision_result["crime_detected"] else 0.0) +
                    0.25 * sound_prob +
                    0.20 * (1.0 if speech_threat else 0.0)
                )
                final_threat = final_score >= 0.3
                threat_reason = "Multi-modal threat detected" if final_threat else "Normal activity"

            return jsonify({
                # Vision details
                "crime_type": vision_result["crime_type"],
                "crime_confidence": vision_result["confidence"],
                "vision_crime_detected": vision_result["crime_detected"],
                
                # Audio details
                "sound_probability": sound_prob,
                "speech_threat": speech_threat,
                "transcription": speech_text,
                
                # Final decision
                "final_threat": final_threat,
                "threat_reason": threat_reason,
                
                # Debug info
                "all_crime_probabilities": vision_result["all_probabilities"]
            })
        except Exception as e:
            print(f"Analysis error: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 Server running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)