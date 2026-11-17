# SIGNOVA 🤟
### Indian Sign Language (ISL) Recognition & Multilingual Voice Translation System

SIGNOVA is an AI-powered assistive communication platform designed to help bridge the communication gap between sign language users and non-signers.

It uses a webcam to recognize selected Indian Sign Language (ISL) gestures, converts them into meaningful text, and provides multilingual voice output in English, Hindi and Kannada.

## 🌟 Key Features

- **Real-Time Sign Recognition:** Uses webcam input, MediaPipe Holistic landmarks and a machine learning classifier.
- **Machine Learning:** Uses an SVM classifier with an RBF kernel to recognize trained sign sequences.
- **Multilingual Translation:** Displays recognized phrases in:
  - English
  - Hindi (हिंदी)
  - Kannada (ಕನ್ನಡ)
- **Voice Output:** Speaks recognized phrases in supported languages.
- **Web Dashboard:** Flask-based interface with:
  - Live webcam interaction
  - Start/Stop recognition
  - Confidence indicator
  - Voice ON/OFF control
  - Language selection
  - Conversation history
  - Replay and clear options
- **Emergency Alerts:** Highlights important phrases such as "I need help" and "I am in danger".
- **Prediction Stability:** Uses confidence thresholds and prediction validation to reduce unstable predictions.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn, SVM, StandardScaler |
| Computer Vision | OpenCV, MediaPipe |
| Data Processing | NumPy |
| Model Storage | Joblib |
| Backend | Flask |
| Frontend | HTML5, CSS3, JavaScript |
| Audio | Python speech and audio libraries |

## 📂 Project Structure

```text
SIGNOVA/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── templates/
│   └── index.html
│
└── ml_model/
    ├── collect_data.py
    ├── train.py
    ├── live_sentence.py
    ├── test_saved.py
    ├── test_model.py
    ├── generate_multilingual_audio.py
    │
    ├── dataset/
    │
    └── models/
        ├── holistic_landmarker.task
        ├── sentence_model.pkl
        └── signova_model.pkl
```

## 📊 Currently Supported Sign Classes

| Sign Class | English Phrase | Hindi | Kannada |
|---|---|---|---|
| `thank_you` | Thank you | धन्यवाद | ಧನ್ಯವಾದಗಳು |
| `i_need_help` | I need help | मुझे मदद चाहिए | ನನಗೆ ಸಹಾಯ ಬೇಕು |
| `i_need_water` | I need water | मुझे पानी चाहिए | ನನಗೆ ನೀರು ಬೇಕು |
| `i_am_in_danger` | I am in danger | मैं खतरे में हूँ | ನಾನು ಅಪಾಯದಲ್ಲಿದ್ದೇನೆ |

**Note:** SIGNOVA currently recognizes a limited set of trained sign sequences. It is a prototype and does not yet support the complete Indian Sign Language vocabulary.

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/amulyaagowda123-stack/SIGNOVA.git
cd SIGNOVA
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

Start the Flask application:

```bash
python app.py
```

Open your browser and visit:

http://127.0.0.1:5000

Allow webcam access when prompted.

### 5. Run the Standalone Recognition System (Optional)

```bash
python ml_model/live_sentence.py
```

Keyboard controls:

- `SPACE` – Start/Stop signing
- `L` – Change voice language
- `R` – Replay the last sentence
- `C` – Clear conversation history
- `Q` – Quit the application

## 🧠 How SIGNOVA Works

1. **Webcam Input:** Captures the user's sign language gestures.
2. **Landmark Detection:** MediaPipe extracts body pose and hand landmarks.
3. **Feature Processing:** Landmark data is processed into a sequence suitable for the trained model.
4. **Sign Classification:** The SVM model predicts the corresponding sign class.
5. **Text Translation:** The recognized phrase is displayed in the selected language.
6. **Voice Output:** The translated phrase is converted into speech.

## 🔮 Future Improvements

- Expand the supported ISL vocabulary.
- Improve recognition accuracy across different users and environments.
- Develop continuous sign language sentence recognition.
- Add more Indian languages.
- Explore deep learning models for improved gesture recognition.
- Develop a mobile application.

## 👩‍💻 Author

Developed by **Amulya M** as an AI-powered assistive communication project for a hackathon.

GitHub: https://github.com/amulyaagowda123-stack

---

⭐ If you find this project interesting, consider giving the repository a star!

## 📸 Project Screenshots

### SIGNOVA Homepage
![SIGNOVA Homepage](screenshots/homepage.png)

### Live Sign Recognition
![SIGNOVA Demo](screenshots/signova_demo.png)