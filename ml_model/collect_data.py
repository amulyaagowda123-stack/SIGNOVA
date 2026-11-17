import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "holistic_landmarker.task")

# Dataset path (save in sentences folder alongside existing data)
sentences_dir = os.path.join(BASE_DIR, "dataset", "sentences")
DATASET_PATH = sentences_dir if os.path.isdir(sentences_dir) else os.path.join(BASE_DIR, "dataset")

# Initialize MediaPipe
BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HolisticLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO
)

# Extract numerical landmarks
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

    return np.array(face + pose + left_hand + right_hand,
                    dtype=np.float32)


# Create webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not detected!")
    exit()

# Ask sentence label
label = input("Enter sign label (example: hello): ").strip().lower()

if not label.isalnum() and "_" not in label:
    print("Invalid label!")
    camera.release()
    exit()

save_path = os.path.join(DATASET_PATH, label)
os.makedirs(save_path, exist_ok=True)

existing = [
    f for f in os.listdir(save_path)
    if f.endswith(".npy")
]

sample_number = len(existing) + 1

print("\nSIGNOVA DATA COLLECTION")
print("Press SPACE to start recording.")
print("Press Q to quit.")

with HolisticLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = time.monotonic_ns() // 1_000_000

        results = landmarker.detect_for_video(
            mp_image, timestamp
        )

        cv2.putText(
            frame,
            f"Label: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "SPACE = Record | Q = Quit",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("SIGNOVA Dataset Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == 32:

            print(f"Recording sample {sample_number}...")

            sequence = []

            for i in range(60):

                success, frame = camera.read()

                if not success:
                    break

                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp = time.monotonic_ns() // 1_000_000

                results = landmarker.detect_for_video(
                    mp_image, timestamp
                )

                keypoints = extract_keypoints(results)
                sequence.append(keypoints)

                cv2.putText(
                    frame,
                    f"Recording: {i + 1}/60",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.imshow(
                    "SIGNOVA Dataset Collection", frame
                )

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if len(sequence) == 60:

                filename = f"sample_{sample_number}.npy"

                np.save(
                    os.path.join(save_path, filename),
                    np.array(sequence, dtype=np.float32)
                )

                print("Saved:", filename)

                sample_number += 1

            else:
                print("Recording incomplete. Not saved.")

camera.release()
cv2.destroyAllWindows()

print("Dataset collection stopped.")