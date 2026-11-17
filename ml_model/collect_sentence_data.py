import cv2
import mediapipe as mp
import numpy as np
import os
import time

MODEL_PATH = "ml_model/models/holistic_landmarker.task"
DATASET_PATH = "ml_model/dataset/sentences"

BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HolisticLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO
)


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


label = input("Enter sentence label (example: i_need_help): ").strip().lower()

if not label or not all(c.isalnum() or c == "_" for c in label):
    print("Invalid sentence label!")
    exit()

save_path = os.path.join(DATASET_PATH, label)
os.makedirs(save_path, exist_ok=True)

existing = [
    f for f in os.listdir(save_path)
    if f.endswith(".npy")
]

sample_number = len(existing) + 1

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not detected!")
    exit()

print("\nSIGNOVA CONTINUOUS SENTENCE COLLECTION")
print("SPACE = Start recording")
print("SPACE again = Stop recording")
print("Q = Quit")

with HolisticLandmarker.create_from_options(options) as landmarker:

    recording = False
    sequence = []

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

        results = landmarker.detect_for_video(mp_image, timestamp)

        if recording:
            sequence.append(extract_keypoints(results))

        status = "RECORDING..." if recording else "READY"

        cv2.putText(
            frame,
            f"Sentence: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            status,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255) if recording else (0, 255, 0),
            2
        )

        cv2.imshow("SIGNOVA Sentence Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == 32:

            if not recording:
                sequence = []
                recording = True
                print("Recording started. Perform the complete sentence.")

            else:
                recording = False

                if len(sequence) > 0:
                    filename = f"sample_{sample_number}.npy"

                    np.save(
                        os.path.join(save_path, filename),
                        np.array(sequence, dtype=np.float32)
                    )

                    print("Saved:", filename)
                    print("Frames:", len(sequence))

                    sample_number += 1

                else:
                    print("No frames recorded.")

camera.release()
cv2.destroyAllWindows()

print("SIGNOVA sentence collection stopped.")