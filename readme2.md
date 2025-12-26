

# 🚨 Multi-Modal CCTV Crime Detection System

A **real-time AI-powered CCTV crime detection web application** that detects **violent and non-violent crimes** using **video, audio, and speech analysis** with an advanced **multi-modal fusion strategy**.

---

## 📌 Project Overview

Traditional CCTV systems only record footage and require **manual monitoring**, which is inefficient and error-prone.
This project introduces an **intelligent CCTV surveillance system** that automatically analyzes video feeds and detects potential criminal activities in real time.

The system combines:

* 🎥 **Visual crime detection**
* 🔊 **Sound anomaly detection**
* 🗣 **Speech threat analysis**

using a **rule-based weighted late fusion approach** to improve accuracy and reduce false alarms.

---

## 🧠 Key Features

* ✅ Detects multiple crime types (UCF-Crime classes)
* ✅ Multi-modal AI fusion (Vision + Audio + Speech)
* ✅ Priority-based decision making
* ✅ Real-time Flask web interface
* ✅ Works on CPU (no GPU required)
* ✅ Robust fallback if any model fails
* ✅ Designed for CCTV-style footage

---

## 🏗 System Architecture

```
CCTV Video Input
       │
       ├── 🎥 Vision Model (VideoMAE – UCF Crime)
       │
       ├── 🔊 Audio Model (Sound CNN)
       │
       ├── 🗣 Speech Model (Whisper + ML Classifier)
       │
       ▼
   Late Fusion Engine
       │
       ▼
  Crime / Normal Decision
```

---

## 🤖 Models Used

### 1️⃣ Vision Model (Primary)

* **Model**: `OPear/videomae-large-finetuned-UCF-Crime`
* **Architecture**: VideoMAE-Large (Transformer-based)
* **Dataset**: UCF-Crime
* **Purpose**: Detect visual crimes from CCTV footage

**Supported Crime Classes**:

* Abuse
* Arrest
* Arson
* Assault
* Burglary
* Explosion
* Fighting
* Robbery
* Shooting
* Shoplifting
* Stealing
* Vandalism
* Road Accidents
* Normal Videos

---

### 2️⃣ Audio Model

* **Model Type**: CNN (Mel-Spectrogram based)
* **Purpose**:

  * Detect screams
  * Detect explosions
  * Detect abnormal sound patterns

---

### 3️⃣ Speech Model

* **ASR Model**: OpenAI Whisper (Tiny-EN)
* **Text Classifier**: Traditional ML (TF-IDF + classifier)
* **Purpose**:

  * Convert speech to text
  * Detect verbal threats or abusive language

---

## 🔀 Fusion Technique Used

### ✅ **Rule-Based Weighted Late Fusion**

This system uses **decision-level (late) fusion**, where each model makes an independent prediction, and the final decision is made by a fusion engine.

---

### 🔹 Step 1: Vision Priority Rule (Hard Decision)

If the vision model detects a crime with **confidence > 0.3**, it **overrides all other models**.

```text
Vision confident → Crime detected immediately
```

This is critical for CCTV systems where **visual evidence is dominant**.

---

### 🔹 Step 2: Weighted Fusion (Soft Decision)

If vision confidence is low, a weighted score is computed:

| Modality  | Weight  |
| --------- | ------- |
| 🎥 Vision | **55%** |
| 🔊 Audio  | **25%** |
| 🗣 Speech | **20%** |

```text
Final Score ≥ 0.3 → Crime detected
```

---

### 🧠 Fusion Type Summary

* **Fusion Level**: Late fusion (decision-level)
* **Method**: Rule-based + weighted scoring
* **Design**: Confidence-aware hybrid fusion

---

## ⚙️ Technology Stack

* **Backend**: Flask (Python)
* **Frontend**: HTML, CSS, JavaScript
* **AI Frameworks**:

  * PyTorch
  * TensorFlow / Keras
  * Hugging Face Transformers
* **Video Processing**: OpenCV, FFmpeg
* **Audio Processing**: Librosa, SoundFile

---

## 📂 Project Structure

```
CCTV-Footage-Audio-Cyber-Crime/
│
├── app/
│   ├── app2.py
│   ├── templates/
│   └── static/
│       ├── videos/
│       ├── css/
│       └── js/
│
├── models/
│   ├── cctv_audio_model.pkl
│   ├── vectorizer.pkl
│   └── sound_effects_cnn_model.keras
│
└── README.md
```

---

## 🚀 How It Works (Execution Flow)

1. User selects a CCTV video feed
2. Video frames are extracted uniformly
3. Audio is extracted from video
4. Vision, audio, and speech models run independently
5. Fusion engine decides final threat
6. Result is displayed on the web interface

---

## 🛡 Fault Tolerance

* If **vision model fails** → system falls back to audio & speech
* If **audio is missing** → vision still works
* If **speech is absent** → system continues safely
* No crashes, no false detections

---

## 📊 Advantages of This Approach

* Higher accuracy than single-modal systems
* Reduced false positives
* Works on real CCTV footage
* Scalable for future upgrades
* Research-grade architecture

---

## 🔮 Future Enhancements

* GPU acceleration (CUDA)
* Live camera stream integration
* Temporal crime localization
* Face anonymization
* Alert system (SMS / Email)
* Crime heatmap analytics

---

## 📚 Key Terms Explained

* **CCTV**: Closed-Circuit Television
* **VideoMAE**: Masked Autoencoder for Video
* **UCF-Crime Dataset**: Benchmark dataset for crime detection
* **Late Fusion**: Combining model outputs at decision level
* **ASR**: Automatic Speech Recognition
* **Mel-Spectrogram**: Audio representation for CNNs
* **Transformer**: Attention-based neural network architecture

---

## 🏁 Conclusion

This project demonstrates a **robust, intelligent, and practical CCTV crime detection system** using **state-of-the-art AI models** and a **carefully designed fusion strategy** suitable for real-world surveillance applications.

---

