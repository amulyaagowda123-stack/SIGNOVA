import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import time
import threading
import pygame
import win32com.client
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request

app = Flask(__name__)

# -------------------------------
# FILE PATHS & SETUP
# -------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(BASE_DIR, "ml_model")

MODEL_PATH = os.path.join(ML_DIR, "models", "sentence_model.pkl")
LANDMARKER_PATH = os.path.join(ML_DIR, "models", "holistic_landmarker.task")
AUDIO_DIR = os.path.join(ML_DIR, "audio")

FRAME_COUNT = 30
CONFIDENCE_THRESHOLD = 40.0
CONFIDENCE_MARGIN = 5.0

# -------------------------------
# MULTILINGUAL DICTIONARY
# -------------------------------

MULTILINGUAL_MAP = {
    "thank_you": {
        "en": "Thank you",
        "hi": "धन्यवाद",
        "kn": "ಧನ್ಯವಾದಗಳು",
        "is_emergency": False
    },
    "i_need_help": {
        "en": "I need help",
        "hi": "मुझे मदद चाहिए",
        "kn": "ನನಗೆ ಸಹಾಯ ಬೇಕು",
        "is_emergency": True
    },
    "i_need_water": {
        "en": "I need water",
        "hi": "मुझे पानी चाहिए",
        "kn": "ನನಗೆ ನೀರು ಬೇಕು",
        "is_emergency": False
    },
    "i_am_in_danger": {
        "en": "I am in danger",
        "hi": "मैं खतरे में हूँ",
        "kn": "ನಾನು ಅಪಾಯದಲ್ಲಿದ್ದೇನೆ",
        "is_emergency": True
    },
    "i_need_food": {
        "en": "I need food",
        "hi": "मुझे खाना चाहिए",
        "kn": "ನನಗೆ ಊಟ ಬೇಕು",
        "is_emergency": False
    }
}

# -------------------------------
# STATE & LOCKS
# -------------------------------

state_lock = threading.Lock()

is_recording = False
recorded_sequence = []
voice_enabled = True
active_language = "en"  # "en", "hi", "kn"

last_result = None
last_spoken_key = None
last_spoken_time = 0
conversation_history = []

# -------------------------------
# VOICE SYSTEM
# -------------------------------

pygame.mixer.init()

speaker = None
try:
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    speaker.Rate = 0
    speaker.Volume = 100
except Exception as e:
    print("SAPI init warning:", e)


def play_voice(prediction_key, lang):
    global last_spoken_key, last_spoken_time

    if not voice_enabled:
        return

    audio_file = os.path.join(AUDIO_DIR, f"{prediction_key}_{lang}.mp3")

    if os.path.exists(audio_file):
        try:
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            last_spoken_key = prediction_key
            last_spoken_time = time.time()
            return
        except Exception as e:
            print("Audio error:", e)

    # Fallback to English SAPI
    if speaker and lang == "en":
        text = MULTILINGUAL_MAP.get(prediction_key, {}).get("en", prediction_key.replace("_", " "))
        try:
            speaker.Speak(text, 1)  # 1 = Asynchronous
        except Exception:
            pass


# -------------------------------
# LOAD ML MODEL & MEDIAPIPE
# -------------------------------

print("Loading SIGNOVA ML model...")
model = joblib.load(MODEL_PATH)
print("SIGNOVA model loaded successfully!")

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HolisticLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=LANDMARKER_PATH),
    running_mode=RunningMode.VIDEO
)

landmarker = HolisticLandmarker.create_from_options(options)

# -------------------------------
# FEATURE EXTRACTION & INFERENCE
# -------------------------------

def extract_keypoints(results):
    def get_points(landmarks, count):
        if landmarks:
            pts = []
            for p in landmarks:
                pts.extend([p.x, p.y, p.z])
            return pts
        return [0.0] * (count * 3)

    face = get_points(results.face_landmarks, 478)
    pose = get_points(results.pose_landmarks, 33)
    left_hand = get_points(results.left_hand_landmarks, 21)
    right_hand = get_points(results.right_hand_landmarks, 21)

    return np.array(face + pose + left_hand + right_hand, dtype=np.float32)


def preprocess_sequence(data):
    data = data[:, 1434:]  # Skip face

    old_frames = np.linspace(0, 1, len(data))
    new_frames = np.linspace(0, 1, FRAME_COUNT)

    resized = np.zeros((FRAME_COUNT, data.shape[1]), dtype=np.float32)

    for i in range(data.shape[1]):
        resized[:, i] = np.interp(new_frames, old_frames, data[:, i])

    return resized.flatten().reshape(1, -1)


def predict_sentence(seq):
    data = np.array(seq, dtype=np.float32)
    processed = preprocess_sequence(data)

    probabilities = model.predict_proba(processed)[0]
    best_idx = np.argmax(probabilities)

    prediction = model.classes_[best_idx]
    confidence = float(probabilities[best_idx] * 100)

    # Margin check
    sorted_probs = sorted(probabilities, reverse=True)
    top_p = sorted_probs[0] * 100
    sec_p = sorted_probs[1] * 100 if len(sorted_probs) > 1 else 0
    margin = float(top_p - sec_p)

    return prediction, confidence, margin


# -------------------------------
# WEBCAM THREAD & STREAMING
# -------------------------------

camera = cv2.VideoCapture(0)
last_timestamp = 0


def generate_frames():
    global is_recording, recorded_sequence, last_timestamp

    while True:
        success, frame = camera.read()
        if not success:
            time.sleep(0.03)
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp = max(time.monotonic_ns() // 1_000_000, last_timestamp + 1)
        last_timestamp = timestamp

        results = landmarker.detect_for_video(mp_image, timestamp)

        with state_lock:
            if is_recording:
                kp = extract_keypoints(results)
                recorded_sequence.append(kp)

        # Draw a small indicator dot if recording
        if is_recording:
            cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)
            cv2.putText(frame, f"REC ({len(recorded_sequence)})", (50, 37),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# -------------------------------
# FLASK ROUTES
# -------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/toggle_recording', methods=['POST'])
def toggle_recording():
    global is_recording, recorded_sequence, last_result, last_spoken_key, last_spoken_time

    with state_lock:
        if not is_recording:
            # Start
            is_recording = True
            recorded_sequence = []
            return jsonify({"status": "started", "recording": True})
        else:
            # Stop & Predict
            is_recording = False
            seq = list(recorded_sequence)

    if len(seq) < 10:
        return jsonify({
            "status": "error",
            "recording": False,
            "message": "Too few frames recorded. Please sign again."
        })

    prediction, confidence, margin = predict_sentence(seq)

    is_stable = (confidence >= CONFIDENCE_THRESHOLD) and (margin >= CONFIDENCE_MARGIN)

    if not is_stable:
        return jsonify({
            "status": "unstable",
            "recording": False,
            "confidence": round(confidence, 1),
            "message": "Sign unclear. Please repeat."
        })

    info = MULTILINGUAL_MAP.get(prediction, {
        "en": prediction.replace("_", " "),
        "hi": prediction.replace("_", " "),
        "kn": prediction.replace("_", " "),
        "is_emergency": False
    })

    # Duplicate Prevention Logic:
    now = time.time()
    is_duplicate = (last_spoken_key == prediction) and ((now - last_spoken_time) < 4.0)

    result_data = {
        "status": "success",
        "recording": False,
        "prediction_key": prediction,
        "en": info["en"],
        "hi": info["hi"],
        "kn": info["kn"],
        "confidence": round(confidence, 1),
        "is_emergency": info.get("is_emergency", False),
        "time": datetime.now().strftime("%I:%M %p")
    }

    last_result = result_data

    # Add to history
    conversation_history.insert(0, {
        "time": result_data["time"],
        "en": result_data["en"],
        "hi": result_data["hi"],
        "kn": result_data["kn"],
        "is_emergency": result_data["is_emergency"]
    })

    # Speak if voice enabled and not an accidental immediate duplicate
    if not is_duplicate:
        threading.Thread(target=play_voice, args=(prediction, active_language)).start()

    return jsonify(result_data)


@app.route('/api/set_language', methods=['POST'])
def set_language():
    global active_language
    data = request.get_json() or {}
    lang = data.get("lang", "en")
    if lang in ["en", "hi", "kn"]:
        active_language = lang
    return jsonify({"status": "ok", "active_language": active_language})


@app.route('/api/toggle_voice', methods=['POST'])
def toggle_voice():
    global voice_enabled
    voice_enabled = not voice_enabled
    return jsonify({"status": "ok", "voice_enabled": voice_enabled})


@app.route('/api/replay', methods=['POST'])
def replay():
    if last_result and "prediction_key" in last_result:
        threading.Thread(target=play_voice, args=(last_result["prediction_key"], active_language)).start()
        return jsonify({"status": "replaying", "text": last_result["en"]})
    return jsonify({"status": "none"})


@app.route('/api/clear', methods=['POST'])
def clear_all():
    global last_result, conversation_history, last_spoken_key
    last_result = None
    conversation_history = []
    last_spoken_key = None
    return jsonify({"status": "cleared"})


@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        "recording": is_recording,
        "voice_enabled": voice_enabled,
        "active_language": active_language,
        "history": conversation_history[:8],
        "last_result": last_result
    })


if __name__ == '__main__':
    print("\n=======================================================")
    print("   SIGNOVA Web App Running at: http://127.0.0.1:5000   ")
    print("   Open this URL in Chrome, Edge, or your browser!     ")
    print("=======================================================\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
