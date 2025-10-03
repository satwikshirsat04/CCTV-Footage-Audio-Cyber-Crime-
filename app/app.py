from flask import Flask, request, jsonify, render_template, send_from_directory
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
import librosa.display
import ffmpeg
import tensorflow as tf
import soundfile as sf
from transformers import pipeline
from dotenv import load_dotenv


app = Flask(__name__, template_folder='templates', static_folder='static')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT is repository root (parent of app directory)
ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

load_dotenv()

# --- Model paths (assumptions based on repository layout) ---
SPEECH_MODEL_PATH = os.path.normpath(os.path.join(ROOT, 'CCTV Speech Threat', 'cctv_audio_model.pkl'))
VECTORIZER_PATH = os.path.normpath(os.path.join(ROOT, 'CCTV Speech Threat', 'vectorizer.pkl'))
SOUND_MODEL_PATH = os.path.normpath(os.path.join(ROOT, 'CCTV Sound Threat', 'notebooks', 'sound_effects_cnn_model.keras'))

# Load models (best-effort; fail gracefully with clear errors)
speech_model = None
vectorizer = None
sound_model = None
whisper_pipe = None

try:
    if os.path.exists(SPEECH_MODEL_PATH):
        speech_model = pickle.load(open(SPEECH_MODEL_PATH, 'rb'))
    if os.path.exists(VECTORIZER_PATH):
        vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
except Exception as e:
    print('Warning: could not load speech model/vectorizer:', e)

try:
    if os.path.exists(SOUND_MODEL_PATH):
        sound_model = tf.keras.models.load_model(SOUND_MODEL_PATH)
except Exception as e:
    print('Warning: could not load sound CNN model:', e)

try:
    # Using HuggingFace transformers ASR pipeline (matches existing project file)
    whisper_pipe = pipeline('automatic-speech-recognition', model='openai/whisper-tiny.en')
except Exception as e:
    print('Warning: could not initialize Whisper pipeline:', e)


def video_to_audio(video_path, out_audio_path):
    """Extract audio from video using ffmpeg-python and save as WAV (16k mono PCM)."""
    try:
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, out_audio_path, acodec='pcm_s16le', ac=1, ar='16000')
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        return True, None
    except Exception as e:
        return False, str(e)


def audio_to_spectrogram_image(audio_path, out_image_path, img_size=(224, 224)):
    """Create a mel-spectrogram image from an audio file and save it.

    Notes/assumptions:
    - Uses a mel-spectrogram converted to dB.
    - Produces an RGB PNG sized to img_size. CNN model is expected to accept images of this shape.
    - If your trained model expects a different input shape or preprocessing, update this function.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
        if y.size == 0:
            return False, 'empty-audio'

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        if S.size == 0:
            return False, 'empty-spectrogram'
        S_db = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(3, 3), dpi=80)
        plt.axis('off')
        # Use imshow for robustness
        plt.imshow(S_db, aspect='auto', origin='lower', cmap='magma')
        plt.tight_layout(pad=0)
        plt.savefig(out_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Resize to requested img_size and ensure 3 channels
        img = Image.open(out_image_path).convert('RGB')
        img = img.resize(img_size)
        img.save(out_image_path)
        return True, None
    except Exception as e:
        return False, str(e)


def predict_sound_from_spectrogram(image_path):
    """Load image, preprocess and predict using the loaded Keras CNN model.

    Assumptions:
    - The model expects images normalized to [0,1] and shaped (1, H, W, 3).
    - The model outputs either a probability (sigmoid) or class scores (softmax). We map to binary threat/non-threat.
    """
    if sound_model is None:
        raise RuntimeError('Sound model not loaded')

    img = Image.open(image_path).convert('RGB')
    arr = np.array(img).astype('float32') / 255.0
    inp = np.expand_dims(arr, axis=0)

    preds = sound_model.predict(inp)
    # Handle common output shapes
    if preds.shape[-1] == 1:
        prob = float(preds[0][0])
        is_threat = prob >= 0.5
    else:
        class_idx = int(np.argmax(preds, axis=-1)[0])
        # Assume class 1 is threat if model was trained that way
        is_threat = class_idx == 1
        prob = float(np.max(preds))

    return {'is_threat': bool(is_threat), 'probability': prob}


def predict_speech_from_audio(audio_path):
    """Transcribe audio with Whisper pipeline and run text model prediction.

    Returns transcription and boolean threat prediction. Requires whisper_pipe, vectorizer and speech_model to be present.
    """
    if whisper_pipe is None:
        raise RuntimeError('Whisper ASR pipeline not available')

    transcription = ''
    try:
        # Try normal transcription with timestamps enabled for long-form audio
        out = whisper_pipe(audio_path, return_timestamps=True)
        transcription = out.get('text', '') if isinstance(out, dict) else ''
    except Exception as e:
        msg = str(e)
        print('Speech pipeline error:', msg)
        # If it's a long-audio issue, trim to first 30 seconds and retry
        if 'more than 3000 mel input features' in msg or 'long-form' in msg:
            try:
                y, sr = librosa.load(audio_path, sr=None)
                max_seconds = 30
                if y.size == 0:
                    transcription = ''
                else:
                    if len(y) > sr * max_seconds:
                        y_trim = y[:sr * max_seconds]
                    else:
                        y_trim = y
                    # write trimmed audio to temp file
                    tmpf = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    tmp_path = tmpf.name
                    tmpf.close()
                    sf.write(tmp_path, y_trim, sr)
                    try:
                        out2 = whisper_pipe(tmp_path, return_timestamps=True)
                        transcription = out2.get('text', '') if isinstance(out2, dict) else ''
                    finally:
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
            except Exception as e2:
                print('Fallback transcription failed:', e2)
        else:
            # Generic fallback attempt without timestamps
            try:
                out = whisper_pipe(audio_path)
                transcription = out.get('text', '') if isinstance(out, dict) else ''
            except Exception as e3:
                print('Whisper fallback failed:', e3)

    is_threat = False
    if speech_model is not None and vectorizer is not None:
        try:
            vec = vectorizer.transform([transcription])
            pred = speech_model.predict(vec)
            is_threat = int(pred[0]) == 1
        except Exception as e:
            print('Speech model prediction error:', e)

    return transcription, bool(is_threat)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    """Main endpoint to analyze a video file.

    Accepts either:
    - multipart file field named 'video' (uploaded video), or
    - JSON/form field 'video_path' pointing to a path under the app static folder (e.g. 'static/videos/feed1.mp4').

    Returns JSON with transcription, per-modality predictions and a fused final_decision.
    """
    tmpdir = tempfile.mkdtemp()
    try:
        # Accept uploaded file
        if 'video' in request.files:
            vid = request.files['video']
            video_path = os.path.join(tmpdir, vid.filename)
            vid.save(video_path)
        else:
            # Accept path
            video_path_param = request.form.get('video_path') or request.json.get('video_path') if request.is_json else None
            if not video_path_param:
                return jsonify({'error': 'No video provided (no file and no video_path)'}), 400
            # Limit to static folder by default to avoid arbitrary paths
            static_root = os.path.join(BASE_DIR, 'static')
            candidate = os.path.normpath(os.path.join(static_root, video_path_param.replace('static/', '').lstrip('/\\')))
            if not candidate.startswith(static_root):
                return jsonify({'error': 'video_path must be under static folder'}), 400
            if not os.path.exists(candidate):
                return jsonify({'error': f'Video not found: {candidate}'}), 404
            video_path = candidate

        audio_path = os.path.join(tmpdir, 'extracted_audio.wav')
        ok, msg = video_to_audio(video_path, audio_path)
        if not ok:
            return jsonify({'error': 'audio extraction failed', 'detail': msg}), 500

        # Speech prediction
        transcription = ''
        speech_threat = False
        try:
            transcription, speech_threat = predict_speech_from_audio(audio_path)
        except Exception as e:
            # Don't fail the whole request if speech fails; record the error
            print('Speech pipeline error:', e)

        # Sound-effect (spectrogram) prediction
        spect_img = os.path.join(tmpdir, 'spec.png')
        ok, msg = audio_to_spectrogram_image(audio_path, spect_img)
        sound_result = {'is_threat': False, 'probability': 0.0}
        if ok:
            try:
                sound_result = predict_sound_from_spectrogram(spect_img)
            except Exception as e:
                print('Sound model prediction error:', e)
        else:
            print('Spectrogram generation failed:', msg)

        # Late fusion: priority to speech (voice) if it detected threat, otherwise use sound model
        final_threat = speech_threat or sound_result.get('is_threat', False)

        response = {
            'transcription': transcription,
            'speech_threat': bool(speech_threat),
            'sound_threat': bool(sound_result.get('is_threat', False)),
            'sound_probability': float(sound_result.get('probability', 0.0)),
            'final_threat': bool(final_threat),
            'message': '🚨 Threat' if final_threat else '✅ No Threat'
        }

        return jsonify(response)

    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
