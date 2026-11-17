# SIGNOVA 🤟
### Indian Sign Language (ISL) Recognition & Multilingual Voice Translation System

SIGNOVA is an AI-powered assistive communication platform that recognizes Indian Sign Language (ISL) gestures in real-time through a webcam, translates them into English, Hindi, and Kannada text, and speaks the translation aloud in authentic native audio.

---

## 🌟 Key Features

- **Real-Time Gesture Recognition**: Uses MediaPipe Holistic landmarker tracking (pose and dual-hand keypoints) combined with an SVM classifier to recognize continuous sign sequences.
- **Multilingual Support**: Real-time simultaneous translation into:
  - **English**
  - **Hindi (हिंदी)**
  - **Kannada (ಕನ್ನಡ)**
- **Native Voice Output (Offline Ready)**: Crystal-clear human-like voice synthesis in English, Hindi, and Kannada.
- **Modern Web Dashboard**: Glassmorphism UI built with Flask, HTML5, CSS3, and JavaScript featuring:
  - Live webcam streaming
  - Start/Stop recognition triggers
  - Dynamic confidence meter gauge
  - Voice ON/Mute toggle switch
  - One-click language selector
  - Conversation chat history with timestamps
- **Emergency Alert Mode**: Prominently highlights distress phrases (*"I am in danger"*, *"I need help"*) with animated emergency banners.
- **Prediction Stability**: Employs confidence thresholds and margin validation to prevent false positives and hand jitter.

---

## 🛠️ Tech Stack

- **Machine Learning**: Scikit-Learn (SVM with RBF kernel), StandardScaler, Joblib, NumPy
- **Computer Vision**: OpenCV, Google MediaPipe Holistic Vision
- **Audio & Speech**: Pygame Mixer, Windows SAPI (pywin32), gTTS
- **Text & Indic Font Rendering**: Pillow (PIL) with Windows Nirmala UI
- **Web Application**: Python Flask, HTML5, Modern CSS Grid, JavaScript

---

## 📂 Project Structure

```text
SIGNOVA/
├── app.py                            # Flask Web Application & API Server
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git exclusion rules
├── templates/
│   └── index.html                    # Modern web dashboard interface
└── ml_model/
    ├── collect_data.py               # Webcam gesture recording tool
    ├── train.py                      # Model training & evaluation pipeline
    ├── live_sentence.py              # Standalone OpenCV desktop HUD system
    ├── generate_multilingual_audio.py# Multilingual audio generator
    ├── audio/                        # Pre-generated MP3 voice clips
    ├── dataset/                      # ISL landmark sequence dataset
    └── models/
        ├── holistic_landmarker.task  # MediaPipe holistic model
        └── sentence_model.pkl        # Trained SVM classification model
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/SIGNOVA.git
cd SIGNOVA
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
.\venv\Scripts\activate   # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run SIGNOVA

#### Option A: Modern Web Dashboard (Recommended)
```bash
python app.py
```
Open your browser and navigate to **`http://127.0.0.1:5000`**.

#### Option B: Standalone Desktop Window
```bash
python ml_model/live_sentence.py
```
- Press **SPACE** to start/stop signing.
- Press **L** to cycle voice language (English / Hindi / Kannada).
- Press **R** to replay the last sentence.
- Press **C** to clear conversation history.
- Press **Q** to quit.

---

## 📊 Recognized Sentences

| Sign Class | English | Hindi (हिंदी) | Kannada (ಕನ್ನಡ) | Emergency |
|---|---|---|---|:---:|
| `thank_you` | Thank you | धन्यवाद | ಧನ್ಯವಾದಗಳು | No |
| `i_need_help` | I need help | मुझे मदद चाहिए | ನನಗೆ ಸಹಾಯ ಬೇಕು | **Yes** |
| `i_need_water` | I need water | मुझे पानी चाहिए | ನನಗೆ ನೀರು ಬೇಕು | No |
| `i_am_in_danger` | I am in danger | मैं खतरे में हूँ | ನಾನು ಅಪಾಯದಲ್ಲಿದ್ದೇನೆ | **Yes** |
| `i_need_food` | I need food | मुझे खाना चाहिए | ನನಗೆ ಊಟ ಬೇಕು | No |

---

## 👥 Authors
Developed for Hackathon by the SIGNOVA Team.
