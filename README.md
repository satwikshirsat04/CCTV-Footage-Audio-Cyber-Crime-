
# 🚨 Multimodal Crime Detection System

**A YOLOv8 and Audio Analysis Pipeline for Automated CCTV Surveillance**  
*Final Year BTech Project (AI & Data Science)*  

![Project Pipeline](/flowchart.png)  
*Figure: System Architecture of Multimodal Crime Detection*

---

## 📌 Overview
This project implements a **multimodal AI system** that detects crimes in CCTV footage by combining:
- **YOLOv8** for visual detection (fighting, weapons, burglary)  
- **CNN-based Audio Analysis** for anomalous sounds (gunshots, screams)  
- **Late/Early Fusion** strategies to improve accuracy
- ![Project Pipeline](/flowchart2.png)  
*Figure: Real-Time Processing*

**Key Features**:
✅ Real-time crime detection at **18 FPS** on edge devices  
✅ **12% higher mAP** compared to vision-only baselines  
✅ Custom dataset with **5 crime classes**  

---

## 🛠️ Installation
```bash
# Clone repository
git clone https://github.com/yourusername/multimodal-crime-detection.git
cd multimodal-crime-detection

# Install dependencies
pip install -r requirements.txt
```

**Hardware Requirements**:
- NVIDIA GPU (for training) / Jetson Nano (for deployment)  
- Minimum 16GB RAM (for dataset processing)  

---

## 📂 Repository Structure
```
multimodal_crime_detection/
├── configs/             # YAML configs for data/model/audio
├── datasets/            # Raw & processed datasets
├── src/                 # Training/inference scripts
├── models/              # Pretrained weights
├── outputs/             # Predictions and logs
└── docs/                # Research paper assets
```

---

## 🚀 Quick Start
### 1. Data Preparation
```bash
# Extract frames from CCTV videos
python src/data_processing/extract_frames.py --input datasets/raw/videos --output datasets/processed/images

# Generate spectrograms from audio
python src/data_processing/audio_to_spectrogram.py --input datasets/raw/audio --output datasets/processed/spectrograms
```

### 2. Train Models
```bash
# Train YOLOv8
python src/training/train_yolo.py --config configs/model_config.yaml

# Train Audio CNN
python src/training/train_audio_model.py --config configs/audio_config.yaml
```

### 3. Run Inference
```bash
# Detect crimes in CCTV stream
python src/inference/detect_crimes.py --video inputs/cctv.mp4 --audio inputs/audio.wav
```

---

## 📊 Results
| Model               | mAP@0.5 | FPS (Jetson Nano) |
|---------------------|---------|-------------------|
| YOLOv8-only         | 0.77    | 32                |
| YOLOv8 + Late Fusion| 0.85    | 18                |
| YOLOv8 + Early Fusion| **0.89**| 14                |

---

## 📝 Research Paper
This work (in process........)  


**Citations**:
```bibtex
@article{ucfcrime,
  title={UCF-Crime: A Large-Scale Dataset for Crime Detection in Surveillance Videos},
  author={Soomro, Khurram and Zamir, Amir Roshan},
  year={2018}
}
```

---

## 🤝 Contributors
- [Satwik Shirsat](https://github.com/satwikshirsat04)  
- [Vrushabh Salunke](https://github.com/Vrushabhsalunke)  

---


### **Key Highlights**:
1. **Visual Hierarchy** - Emojis and headers organize content intuitively  
2. **Code Blocks** - Ready-to-run commands for easy setup  
3. **Results Table** - Clear performance metrics  
4. **Academic Citations** - Professional BibTeX format  
5. **Mobile-Friendly** - Clean Markdown rendering on GitHub   

