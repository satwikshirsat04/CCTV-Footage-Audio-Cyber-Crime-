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
import traceback


app = Flask(__name__, template_folder='templates', static_folder='static')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT is repository root (parent of app directory)
ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

load_dotenv()

# --- Model paths and initialization ---
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Define model source and destination paths
required_models = {
    'speech': ('CCTV Speech Threat/cctv_audio_model.pkl', 'cctv_audio_model.pkl'),
    'vectorizer': ('CCTV Speech Threat/vectorizer.pkl', 'vectorizer.pkl'),
    'sound': ('CCTV Sound Threat/notebooks/sound_effects_cnn_model.keras', 'sound_effects_cnn_model.keras')
}

# Copy models to app/models directory if they don't exist
def ensure_models_exist():
    """Copy models from source locations to app/models if needed"""
    missing_models = []
    for model_type, (src_rel, dst_name) in required_models.items():
        src_path = os.path.join(ROOT, src_rel)
        dst_path = os.path.join(MODELS_DIR, dst_name)
        if not os.path.exists(dst_path):
            if os.path.exists(src_path):
                print(f"Copying {model_type} model to {dst_path}")
                shutil.copy2(src_path, dst_path)
            else:
                missing_models.append(f"{model_type} ({src_path})")
    if missing_models:
        print("WARNING: Missing model files:", ', '.join(missing_models))

# Copy models if they don't exist in models directory
for model_type, (src_rel, dst_name) in required_models.items():
    src_path = os.path.join(ROOT, src_rel)
    dst_path = os.path.join(MODELS_DIR, dst_name)
    if os.path.exists(src_path) and not os.path.exists(dst_path):
        print(f"Copying {model_type} model to {dst_path}")
        shutil.copy2(src_path, dst_path)

SPEECH_MODEL_PATH = os.path.normpath(os.path.join(MODELS_DIR, 'cctv_audio_model.pkl'))
VECTORIZER_PATH = os.path.normpath(os.path.join(MODELS_DIR, 'vectorizer.pkl'))
SOUND_MODEL_PATH = os.path.normpath(os.path.join(MODELS_DIR, 'sound_effects_cnn_model.keras'))

# Load all models at startup
print("\nLoading models...")

try:
    speech_model = pickle.load(open(SPEECH_MODEL_PATH, 'rb'))
    print("✓ Speech model loaded")
except Exception as e:
    print(f"✗ Failed to load speech model: {e}")
    speech_model = None

try:
    vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
    print("✓ Vectorizer loaded")
except Exception as e:
    print(f"✗ Failed to load vectorizer: {e}")
    vectorizer = None

sound_model = None
def load_sound_model():
    """Load the sound model with multiple fallback strategies."""
    global sound_model
    try:
        print(f"Loading sound model from: {SOUND_MODEL_PATH}")
        if not os.path.exists(SOUND_MODEL_PATH) or os.path.getsize(SOUND_MODEL_PATH) == 0:
            print("ERROR: Sound model file is missing or empty")
            return None

        # Try different loading approaches
        load_attempts = [
            # Attempt 1: Standard load without compilation
            lambda: tf.keras.models.load_model(SOUND_MODEL_PATH, compile=False),
            # Attempt 2: Load with custom object scope
            lambda: tf.keras.models.load_model(SOUND_MODEL_PATH,
                compile=False,
                custom_objects={'BatchNormalization': tf.keras.layers.BatchNormalization}),
            # Attempt 3: Load with full compilation
            lambda: tf.keras.models.load_model(SOUND_MODEL_PATH)
        ]

        last_error = None
        for i, load_fn in enumerate(load_attempts, 1):
            try:
                print(f"Attempt {i} to load model...")
                loaded_model = load_fn()
                print(f'Sound model loaded successfully on attempt {i}')

                # Verify model can make predictions
                test_input = np.zeros((1, 232, 231, 3))
                try:
                    _ = loaded_model.predict(test_input, verbose=0)
                    print("Model verified - test prediction successful")
                    sound_model = loaded_model
                    return loaded_model
                except Exception as e:
                    print(f"Model loaded but test prediction failed: {e}")
                    raise

            except Exception as e:
                last_error = e
                print(f'Load attempt {i} failed:')
                traceback.print_exc()

        print('All load attempts failed. Last error:', last_error)
        return None
    except Exception as e:
        print('Warning: could not load sound CNN model:', e)
        traceback.print_exc()
        return None

# Load sound model at startup
load_sound_model()


try:
    whisper_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")
    print("✓ Whisper ASR pipeline initialized")
except Exception as e:
    print(f"✗ Failed to initialize Whisper ASR: {e}")
    whisper_pipe = None


def video_to_audio(video_path, out_audio_path):
    """Extract audio from video using ffmpeg-python and save as WAV (16k mono PCM).
    
    Args:
        video_path (str): Path to input video file
        out_audio_path (str): Path to save extracted audio
        
    Returns:
        tuple: (success, error_message)
    """
    try:
        if not os.path.exists(video_path):
            return False, f"Video file not found: {video_path}"
            
        print(f"Extracting audio from {video_path}")
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, out_audio_path, 
                             acodec='pcm_s16le',  # 16-bit PCM
                             ac=1,                # mono
                             ar='16000')          # 16kHz sampling
        ffmpeg.run(stream, overwrite_output=True, capture_stderr=True)
        
        if os.path.exists(out_audio_path) and os.path.getsize(out_audio_path) > 0:
            print(f"Audio extracted successfully to {out_audio_path}")
            return True, None
    except Exception as e:
        return False, str(e)


def audio_to_spectrogram_image(audio_path, out_image_path, img_size=(232, 231)):
    """Create a mel-spectrogram image from an audio file and save it.
    
    Exactly matches training preprocessing:
    - Sample rate: 22050 Hz
    - Mel bands: 128
    - Output size: 232x231 pixels (matches training)
    - Color mode: RGB
    - Normalization: Same as training notebook
    
    Args:
        audio_path (str): Path to input audio file
        out_image_path (str): Path to save spectrogram image
        img_size (tuple): Target image size, default (232,231) matches training
        
    Returns:
        tuple: (success, error_message)
    """
    try:
        y, sr = librosa.load(audio_path, sr=22050)  # Match training sample rate
        if y.size == 0:
            return False, 'empty-audio'

        # Match training parameters exactly
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        if S is None or S.size == 0:
            return False, 'empty-spectrogram'
        S_db = librosa.power_to_db(S, ref=np.max)

        # sanitize NaN/inf
        if not np.isfinite(S_db).all():
            S_db = np.nan_to_num(S_db, nan=-80.0, neginf=-80.0, posinf=np.nanmax(S_db[np.isfinite(S_db)]) if np.any(np.isfinite(S_db)) else 0.0)

        fig, ax = plt.subplots(figsize=(3, 3), dpi=80)
        try:
            print(f"Generating spectrogram with shape {S_db.shape}")
            ax.set_axis_off()
            ax.imshow(S_db, aspect='auto', origin='lower', cmap='magma')
            fig.tight_layout(pad=0)
            # Ensure output directory exists
            outdir = os.path.dirname(out_image_path)
            if outdir and not os.path.exists(outdir):
                os.makedirs(outdir, exist_ok=True)
            fig.savefig(out_image_path, bbox_inches='tight', pad_inches=0)
        finally:
            plt.close(fig)

        # Resize to requested img_size and ensure 3 channels
        img = Image.open(out_image_path).convert('RGB')
        img = img.resize(img_size)
        img.save(out_image_path)
        return True, None
    except Exception as e:
        tb = traceback.format_exc()
        return False, f"{str(e)}\n{tb}"


def predict_sound_from_spectrogram(image_path):
    """Predict sound threat from spectrogram using CNN model.
    
    Preprocessing matches training exactly:
    - Image resized to 232x231
    - RGB input normalized to [0,1]
    - Batch shape (1, 232, 231, 3)
    
    Model output:
    - 4 classes: bomb, crash, gun, shout
    - Returns probability and threat status (bomb/crash/gun are threats)
    
    Args:
        image_path (str): Path to input spectrogram image
        
    Returns:
        dict: Contains is_threat boolean and probability float
    """
    global sound_model
    if sound_model is None:
        print("Warning: Sound model not loaded")
        return {'is_threat': False, 'probability': 0.0, 'class': None}

    img = Image.open(image_path).convert('RGB')
    arr = np.array(img).astype('float32') / 255.0
    inp = np.expand_dims(arr, axis=0)

    print("Running prediction with input shape:", inp.shape)
    preds = sound_model.predict(inp, verbose=0)
    print("Raw prediction shape:", preds.shape)
    
    # Model was trained with categorical labels: bomb, crash, gun, shout
    # Consider gun, bomb and crash as threats
    class_idx = int(np.argmax(preds, axis=-1)[0])
    class_probs = preds[0]
    
    # Map class indices to threat levels
    threat_classes = ['bomb', 'crash', 'gun']  # These are considered threats
    class_names = ['bomb', 'crash', 'gun', 'shout']
    predicted_class = class_names[class_idx]
    
    # Consider it a threat if the predicted class is in threat_classes
    is_threat = predicted_class in threat_classes
    prob = float(class_probs[class_idx])

    return {'is_threat': bool(is_threat), 'probability': prob}


def predict_speech_from_audio(audio_path):
    """Transcribe audio with Whisper pipeline and run text model prediction.

    Returns transcription and boolean threat prediction. Requires whisper_pipe, vectorizer and speech_model to be present.
    
    Args:
        audio_path (str): Path to input audio file
        
    Returns:
        tuple: (transcription, is_threat, error)
    """
    global whisper_pipe, speech_model, vectorizer
    if whisper_pipe is None:
        return '', False, 'Whisper ASR pipeline not available'
        
    if speech_model is None or vectorizer is None:
        return '', False, 'Speech model or vectorizer not available'

    transcription = ''
    try:
        # First try normal transcription
        out = whisper_pipe(audio_path)
        transcription = out.get('text', '') if isinstance(out, dict) else str(out)
        
        if not transcription.strip():
            return '', False, 'No speech detected'
            
        # Vectorize the text
        X = vectorizer.transform([transcription])
        
        # Predict threat
        prediction = speech_model.predict(X)[0]
        
        return transcription, bool(prediction), None
        
    except Exception as e:
        print(f'Speech prediction error: {e}')
        return '', False, str(e)
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


@app.route('/load_models', methods=['GET'])
def load_models():
    """Report model loading status."""
    global speech_model, vectorizer, sound_model, whisper_pipe
    statuses = {
        'speech_model_exists': os.path.exists(SPEECH_MODEL_PATH),
        'vectorizer_exists': os.path.exists(VECTORIZER_PATH),
        'sound_model_exists': os.path.exists(SOUND_MODEL_PATH),
        'speech_model_loaded': bool(speech_model is not None),
        'vectorizer_loaded': bool(vectorizer is not None),
        'sound_model_loaded': bool(sound_model is not None)
    }

    # Try loading sound model if not loaded
    if not statuses['sound_model_loaded'] and statuses['sound_model_exists']:
        loaded = load_sound_model()
        statuses['sound_model_loaded'] = bool(loaded is not None)

    return jsonify(statuses)


@app.route('/health', methods=['GET'])
def health():
    """Simple health check: returns 200 if server up and models exist (not necessarily loaded)."""
    ok = os.path.exists(SPEECH_MODEL_PATH) and os.path.exists(VECTORIZER_PATH)
    return jsonify({'ok': ok, 'speech_model_exists': os.path.exists(SPEECH_MODEL_PATH), 'vectorizer_exists': os.path.exists(VECTORIZER_PATH)})


@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    """Analyze video for speech and sound-based threats.
    
    Steps:
    1. Accept video input (file upload or path)
    2. Extract audio using ffmpeg
    3. Generate spectrogram for sound model
    4. Run sound threat detection (CNN)
    5. Transcribe audio and check for speech threats
    6. Combine predictions
    
    Input:
        - Multipart file upload named 'video', or
        - Form/JSON field 'video_path' relative to static folder
    
    Returns:
        JSON with transcription, threat predictions and confidence
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

        # Track progress and errors
        progress = {'status': {}, 'errors': []}
        
        # 1. Speech prediction
        print("Starting speech analysis...")
        transcription = ''
        speech_threat = False
        try:
            transcription, speech_threat = predict_speech_from_audio(audio_path)
            progress['status']['speech'] = 'success'
            print(f"Speech analysis complete: {'THREAT' if speech_threat else 'NO THREAT'}")
        except Exception as e:
            error_msg = f'Speech pipeline error: {str(e)}'
            print(error_msg)
            progress['status']['speech'] = 'failed'
            progress['errors'].append(error_msg)

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


def verify_models():
    """Verify all models are present and can be loaded."""
    ensure_models_exist()
    
    print("\nVerifying models...")
    
    # Check file existence
    models_exist = all([
        os.path.exists(SPEECH_MODEL_PATH),
        os.path.exists(VECTORIZER_PATH),
        os.path.exists(SOUND_MODEL_PATH)
    ])
    
    if not models_exist:
        print("❌ Some model files are missing!")
        return False
        
    # Try loading sound model if not loaded
    if sound_model is None:
        try:
            load_sound_model()
        except Exception as e:
            print(f"❌ Failed to load sound model: {e}")
    
    # Report status
    print("\nModel Status:")
    print(f"{'✓' if speech_model else '✗'} Speech Model")
    print(f"{'✓' if vectorizer else '✗'} Vectorizer")
    print(f"{'✓' if sound_model else '✗'} Sound Model")
    print(f"{'✓' if whisper_pipe else '✗'} Whisper ASR")
    
    all_loaded = all([
        speech_model is not None,
        vectorizer is not None,
        sound_model is not None,
        whisper_pipe is not None
    ])
    
    if all_loaded:
        print("\n✨ All models loaded successfully!")
    else:
        print("\n⚠️ Some models failed to load")
        
    return all_loaded

if __name__ == '__main__':
    if not verify_models():
        print("\nWARNING: Some models are missing. The application may not work correctly.")
    print("\nStarting server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
