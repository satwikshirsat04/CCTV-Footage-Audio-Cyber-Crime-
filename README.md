
# 🚨 Multimodal Crime Detection System

**A YOLOv8 and Audio Analysis and Speech (NLP) Pipeline for Automated CCTV Surveillance**  
*Final Year BTech Project (AI & Data Science)*  
<h2>Flowchart :</h2>


![Project Pipeline](/MCDFLOWCHART.drawio.svg)  

*Figure: Complete System Architecture of Multimodal Crime Detection*

---

## 📌 Overview
This project implements a **multimodal AI system** that detects crimes in CCTV footage by combining:
- **YOLOv8** for visual detection (fighting, weapons, burglary)  
- **CNN-based Audio Analysis** for anomalous sounds (gunshots, screams)
- **NLP- SPEECH Model** for Detecting Criminal Vocals from Public voice
- **Late/Early Fusion** strategies to improve accuracy
- <h2>Real-Time Processing</h2>
![Project Pipeline](/RTP.svg)  
*Figure: Real-Time Processing*
 

---

## SOUND EFFECT DETECTION MODEL TRAINING:
![SPECTROGRAM](/CCTV%20Sound%20Threat/notebooks/spectrogram_crash1.png)  
*Figure: Car Accident Crash CNN Processing with SPECTROGRAM TECHNIQUE*

---
![SPECTROGRAM](/CCTV%20Sound%20Threat/notebooks/spectrogram_gun1.png)
*Figure: Gun Shot CNN Processing with SPECTROGRAM TECHNIQUE*

---
![SPECTROGRAM](/CCTV%20Sound%20Threat/spectrogram%20dataset/bomb/bomb1.png)  
*Figure: Bomb Attack CNN Processing with SPECTROGRAM TECHNIQUE*

---

## SOUND EFFECT DETECTION MODEL TRAINING:
![SPEECH MODEL](/CCTV%20Speech%20Threat/assets/cyber-threat.png)  
*Figure: Cyber Threat Detected Live Example*

![ALERT SYSTEM TWILIO](/CCTV%20Speech%20Threat/assets/Twilio%20Alert.jpg)  
*Figure: Cyber Threat Detected Twilio Alert System*

## 🛠️ Installation
```bash
# Clone repository
git clone https://github.com/satwikshirsat04/CCTV-Footage-Audio-Cyber-Crime-.git


# Install dependencies
pip install -r requirements.txt
```

**Hardware Requirements**:
- NVIDIA GPU MIN 4GB VRAM (for training) - USE GOOGLE COLAB
- Minimum 8 GB RAM (for dataset processing)  


---



## 📝 Research Paper
This work (in process........)  


---

## 🤝 Contributors
- [Satwik Shirsat](https://github.com/satwikshirsat04)  
- [Vrushabh Salunke](https://github.com/Vrushabh6454)  

---


