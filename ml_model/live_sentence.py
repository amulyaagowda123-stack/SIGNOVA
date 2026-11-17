import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import time
import pygame
import win32com.client
from PIL import Image, ImageDraw, ImageFont
from collections import deque
from datetime import datetime

# -------------------------------
# FILE PATHS
# -------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "sentence_model.pkl")
LANDMARKER_PATH = os.path.join(BASE_DIR, "models", "holistic_landmarker.task")
AUDIO_DIR = os.path.join(BASE_DIR, "audio")

FRAME_COUNT = 30
CONFIDENCE_THRESHOLD = 40.0  # Increased for stability
CONFIDENCE_MARGIN = 5.0      # Margin over second-best guess

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

# Supported voice languages
LANGUAGES = ["en", "hi", "kn"]
LANG_NAMES = {
    "en": "English",
    "hi": "Hindi (हिंदी)",
    "kn": "Kannada (ಕನ್ನಡ)"
}
current_lang_idx = 0  # 0: English, 1: Hindi, 2: Kannada

# Conversation history (stores last 3 recognized sentences)
conversation_history = deque(maxlen=3)

# -------------------------------
# VOICE SYSTEM
# -------------------------------

pygame.mixer.init()

# Fallback SAPI engine for English
speaker = win32com.client.Dispatch("SAPI.SpVoice")
speaker.Rate = 0
speaker.Volume = 100


def speak_sentence(prediction, lang="en"):

    audio_file = os.path.join(AUDIO_DIR, f"{prediction}_{lang}.mp3")

    if os.path.exists(audio_file):
        try:
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            return
        except Exception as e:
            print("Audio playback warning:", e)

    # Fallback to Windows SAPI (English)
    fallback_text = MULTILINGUAL_MAP.get(prediction, {}).get("en", prediction.replace("_", " "))
    speaker.Speak(fallback_text)


# -------------------------------
# FONTS FOR INDIC TEXT (Pillow)
# -------------------------------

FONT_PATHS = [
    "C:/Windows/Fonts/Nirmala.ttc",
    "C:/Windows/Fonts/NirmalaB.ttf",
    "C:/Windows/Fonts/nirmala.ttf",
    "C:/Windows/Fonts/arial.ttf"
]

font_title = None
font_sub = None
font_text = None
font_small = None

for path in FONT_PATHS:
    if os.path.exists(path):
        try:
            font_title = ImageFont.truetype(path, 21)
            font_sub = ImageFont.truetype(path, 16)
            font_text = ImageFont.truetype(path, 19)
            font_small = ImageFont.truetype(path, 14)
            break
        except Exception:
            continue

if font_title is None:
    font_title = font_sub = font_text = font_small = ImageFont.load_default()


# -------------------------------
# DRAW ENHANCED HUD UI
# -------------------------------

def draw_enhanced_ui(frame, recording, frame_count, result_data, history, current_lang_name):

    h, w, _ = frame.shape

    # Semi-transparent overlay
    overlay = frame.copy()

    # 1. Top status bar
    cv2.rectangle(overlay, (0, 0), (w, 50), (18, 18, 22), -1)

    # Split bottom into Translation Card (left) and History / Confidence (right)
    split_x = int(w * 0.60)
    card_top = h - 145

    # Translation card (left)
    cv2.rectangle(overlay, (12, card_top), (split_x - 8, h - 12), (22, 22, 28), -1)

    # History & Confidence panel (right)
    cv2.rectangle(overlay, (split_x + 4, card_top), (w - 12, h - 12), (22, 22, 28), -1)

    # Blend overlay
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    # Check if current result is an emergency
    is_emergency = result_data.get("is_emergency", False) if result_data else False

    # Border colors
    if recording:
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 220), 2)
    else:
        cv2.rectangle(frame, (0, 0), (w, 50), (45, 160, 45), 1)

    # Translation card border (Red glowing border if Emergency!)
    card_border_color = (0, 40, 255) if is_emergency else (0, 200, 255)
    card_border_thickness = 3 if is_emergency else 2
    cv2.rectangle(frame, (12, card_top), (split_x - 8, h - 12), card_border_color, card_border_thickness)

    # History card border
    cv2.rectangle(frame, (split_x + 4, card_top), (w - 12, h - 12), (100, 100, 110), 1)

    # -----------------------------------
    # Confidence Bar Graphic (Right Panel)
    # -----------------------------------
    if result_data and result_data.get("conf", 0) > 0:
        conf_val = result_data["conf"]
        bar_x1 = split_x + 18
        bar_y1 = card_top + 28
        bar_w_max = (w - 12) - bar_x1 - 18
        bar_h = 10

        # Background track
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + bar_w_max, bar_y1 + bar_h), (50, 50, 55), -1)

        # Progress fill width
        fill_w = int(bar_w_max * min(conf_val / 100.0, 1.0))

        # Color based on confidence
        if conf_val >= 70:
            bar_color = (0, 220, 100)    # Green
        elif conf_val >= 45:
            bar_color = (0, 200, 255)    # Amber/Yellow
        else:
            bar_color = (0, 50, 255)     # Red

        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y1 + bar_h), bar_color, -1)

    # -----------------------------------
    # Text Rendering with Pillow
    # -----------------------------------
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Top Status Bar
    if recording:
        draw.text(
            (20, 12),
            f"● RECORDING GESTURE... [Frames: {frame_count}]",
            font=font_title,
            fill=(255, 70, 70)
        )
    else:
        draw.text(
            (20, 14),
            f"READY | [SPACE] Sign  [L] Voice:{current_lang_name}  [R] Replay  [C] Clear  [Q] Quit",
            font=font_sub,
            fill=(90, 255, 120)
        )

    # Left Panel: Translation Card
    if result_data:
        en_t = result_data.get("en", "")
        hi_t = result_data.get("hi", "")
        kn_t = result_data.get("kn", "")

        header_label = "🚨 EMERGENCY DETECTED" if is_emergency else "LIVE TRANSLATION"
        header_color = (255, 80, 80) if is_emergency else (0, 220, 255)

        draw.text((22, card_top + 8), header_label, font=font_small, fill=header_color)
        draw.text((22, card_top + 34), f"EN:  {en_t}", font=font_text, fill=(255, 255, 255))
        draw.text((22, card_top + 68), f"HI:   {hi_t}", font=font_text, fill=(255, 220, 100))
        draw.text((22, card_top + 102), f"KN:  {kn_t}", font=font_text, fill=(100, 240, 255))
    else:
        draw.text(
            (22, card_top + 55),
            "Perform a gesture to see live translation in English, Hindi & Kannada.",
            font=font_sub,
            fill=(170, 170, 170)
        )

    # Right Panel: Confidence & Conversation History
    draw.text((split_x + 18, card_top + 8), "CONFIDENCE", font=font_small, fill=(200, 200, 200))

    if result_data and result_data.get("conf", 0) > 0:
        conf_val = result_data["conf"]
        draw.text((split_x + 120, card_top + 8), f"{conf_val:.1f}%", font=font_small, fill=(0, 240, 180))

    draw.text((split_x + 18, card_top + 45), "HISTORY (Last 3)", font=font_small, fill=(180, 180, 180))

    if history:
        for idx, item in enumerate(history):
            t_str = item.get("time", "")
            msg = item.get("text", "")
            draw.text(
                (split_x + 18, card_top + 68 + (idx * 22)),
                f"• [{t_str}] {msg}",
                font=font_small,
                fill=(220, 220, 220)
            )
    else:
        draw.text(
            (split_x + 18, card_top + 75),
            "No recent sentences yet.",
            font=font_small,
            fill=(130, 130, 140)
        )

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# -------------------------------
# LOAD MODEL
# -------------------------------

print("Loading SIGNOVA model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")
print("Multilingual voice system ready!")


# -------------------------------
# MEDIAPIPE SETUP
# -------------------------------

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HolisticLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=LANDMARKER_PATH
    ),
    running_mode=RunningMode.VIDEO
)


# -------------------------------
# EXTRACT KEYPOINTS
# -------------------------------

def extract_keypoints(results):

    def get_points(landmarks, count):

        if landmarks:
            points = []
            for point in landmarks:
                points.extend([point.x, point.y, point.z])
            return points

        return [0.0] * (count * 3)

    face = get_points(results.face_landmarks, 478)
    pose = get_points(results.pose_landmarks, 33)
    left_hand = get_points(results.left_hand_landmarks, 21)
    right_hand = get_points(results.right_hand_landmarks, 21)

    return np.array(
        face + pose + left_hand + right_hand,
        dtype=np.float32
    )


# -------------------------------
# PREPROCESS SEQUENCE
# -------------------------------

def preprocess_sequence(data):

    data = data[:, 1434:]

    old_frames = np.linspace(0, 1, len(data))
    new_frames = np.linspace(0, 1, FRAME_COUNT)

    resized = np.zeros(
        (FRAME_COUNT, data.shape[1]),
        dtype=np.float32
    )

    for i in range(data.shape[1]):
        resized[:, i] = np.interp(
            new_frames,
            old_frames,
            data[:, i]
        )

    return resized.flatten().reshape(1, -1)


# -------------------------------
# PREDICTION
# -------------------------------

def predict_sentence(sequence):

    data = np.array(sequence, dtype=np.float32)
    processed = preprocess_sequence(data)

    probabilities = model.predict_proba(processed)[0]
    best_index = np.argmax(probabilities)

    prediction = model.classes_[best_index]
    confidence = probabilities[best_index] * 100

    return prediction, confidence, probabilities


# -------------------------------
# CAMERA
# -------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not detected!")
    exit()

recording = False
sequence = []
last_timestamp = 0
last_result_data = None
last_prediction_key = None

print("\n================================================")
print("     SIGNOVA MULTILINGUAL ISL TRANSLATOR        ")
print("================================================")
print("Hotkeys:")
print("  SPACE : START / STOP signing")
print("  L     : TOGGLE VOICE LANGUAGE (English / Hindi / Kannada)")
print("  R     : REPLAY LAST SPOKEN SENTENCE")
print("  C     : CLEAR SCREEN & CONVERSATION HISTORY")
print("  Q     : QUIT")
print("================================================\n")


# -------------------------------
# MAIN LOOP
# -------------------------------

with HolisticLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = camera.read()

        if not success:
            print("Camera frame error!")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = max(
            time.monotonic_ns() // 1_000_000,
            last_timestamp + 1
        )
        last_timestamp = timestamp

        results = landmarker.detect_for_video(mp_image, timestamp)

        # ---------------------------
        # RECORD LANDMARKS
        # ---------------------------

        if recording:
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

        # ---------------------------
        # DRAW ENHANCED HUD UI
        # ---------------------------

        current_lang = LANGUAGES[current_lang_idx]
        current_lang_name = LANG_NAMES[current_lang]

        frame = draw_enhanced_ui(
            frame,
            recording,
            len(sequence),
            last_result_data,
            conversation_history,
            current_lang_name
        )

        cv2.imshow("SIGNOVA - Multilingual ISL Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        # ---------------------------
        # QUIT (Q)
        # ---------------------------

        if key == ord('q') or key == ord('Q'):
            break

        # ---------------------------
        # SWITCH LANGUAGE (L)
        # ---------------------------

        if key == ord('l') or key == ord('L'):
            current_lang_idx = (current_lang_idx + 1) % len(LANGUAGES)
            active_lang = LANGUAGES[current_lang_idx]
            print(f"\n[LANGUAGE TOGGLED] Current Voice: {LANG_NAMES[active_lang]}")

        # ---------------------------
        # REPLAY LAST SENTENCE (R)
        # ---------------------------

        if (key == ord('r') or key == ord('R')) and last_prediction_key:
            active_lang = LANGUAGES[current_lang_idx]
            print(f"\n[REPLAY] Speaking again in {LANG_NAMES[active_lang]}...")
            speak_sentence(last_prediction_key, lang=active_lang)

        # ---------------------------
        # CLEAR CONVERSATION (C)
        # ---------------------------

        if key == ord('c') or key == ord('C'):
            conversation_history.clear()
            last_result_data = None
            last_prediction_key = None
            print("\n[CLEARED] Conversation history and screen reset.")

        # ---------------------------
        # START RECORDING (SPACE)
        # ---------------------------

        if key == 32 and not recording:
            recording = True
            sequence = []
            print("\nRecording started...")

        # ---------------------------
        # STOP RECORDING (SPACE)
        # ---------------------------

        elif key == 32 and recording:
            recording = False
            print("Recording stopped.")

            if len(sequence) < 10:
                print("Too few frames. Please sign again.")
                continue

            prediction, confidence, probabilities = predict_sentence(sequence)

            print("\nSIGNOVA RESULT")
            print("Prediction:", prediction)
            print(f"Confidence: {round(confidence, 2)}%")

            print("Class breakdown:")
            for cls, prob in zip(model.classes_, probabilities):
                print(f"  - {cls}: {round(prob * 100, 1)}%")

            # Prediction Stability Check
            sorted_probs = sorted(probabilities, reverse=True)
            top_prob = sorted_probs[0] * 100
            second_prob = sorted_probs[1] * 100 if len(sorted_probs) > 1 else 0
            margin = top_prob - second_prob

            is_stable = (confidence >= CONFIDENCE_THRESHOLD) and (margin >= CONFIDENCE_MARGIN)

            if is_stable:
                translations = MULTILINGUAL_MAP.get(
                    prediction,
                    {
                        "en": prediction.replace("_", " "),
                        "hi": prediction.replace("_", " "),
                        "kn": prediction.replace("_", " "),
                        "is_emergency": False
                    }
                )

                last_result_data = {
                    "en": translations["en"],
                    "hi": translations["hi"],
                    "kn": translations["kn"],
                    "conf": confidence,
                    "is_emergency": translations.get("is_emergency", False)
                }
                last_prediction_key = prediction

                active_lang = LANGUAGES[current_lang_idx]
                spoken_text = translations.get(active_lang, translations["en"])

                # Add to Conversation History
                now_str = datetime.now().strftime("%H:%M")
                conversation_history.append({
                    "time": now_str,
                    "text": translations["en"]
                })

                print(f"\nRecognized: {translations['en']} | {translations['hi']} | {translations['kn']}")
                print(f"Speaking ({LANG_NAMES[active_lang]}): {spoken_text}")

                speak_sentence(prediction, lang=active_lang)

            else:
                last_result_data = {
                    "en": "Uncertain - Please repeat",
                    "hi": "अस्पष्ट - कृपया दोहराएं",
                    "kn": "ಅಸ್ಪಷ್ಟ - ಪುನರಾವರ್ತಿಸಿ",
                    "conf": confidence,
                    "is_emergency": False
                }
                print("PREDICTION UNSTABLE / LOW CONFIDENCE (Ignored)")

            print("------------------------------------------------")


# -------------------------------
# CLEANUP
# -------------------------------

camera.release()
cv2.destroyAllWindows()
pygame.mixer.quit()

print("SIGNOVA stopped.")
