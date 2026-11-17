
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

# Load trained model
model = joblib.load("ml_model/models/signova_model.pkl")

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HolisticLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="ml_model/models/holistic_landmarker.task"
    ),
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


camera = cv2.VideoCapture(0)

sequence = []
recording = False
prediction_text = "Press SPACE to start"
confidence_text = ""

print("SIGNOVA - Press SPACE to start and stop signing.")
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
        results = landmarker.detect_for_video(mp_image, timestamp)

        if recording:
            sequence.append(extract_keypoints(results))

        status = "RECORDING..." if recording else "READY"

        cv2.putText(
            frame, status, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 0, 255) if recording else (0, 255, 0), 2
        )

        cv2.putText(
            frame, prediction_text, (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (255, 255, 0), 2
        )

        cv2.putText(
            frame, confidence_text, (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (255, 255, 0), 2
        )

        cv2.imshow("SIGNOVA - Sign Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == 32:

            if not recording:
                sequence = []
                recording = True
                prediction_text = "Signing..."
                confidence_text = ""
                print("Recording started.")

            else:
                recording = False

                if len(sequence) >= 5:

                    indices = np.linspace(
                        0, len(sequence) - 1, 20
                    ).astype(int)

                    features = np.array(sequence)[indices].flatten()
                    features = features.reshape(1, -1)

                    prediction = model.predict(features)[0]
                    probabilities = model.predict_proba(features)[0]
                    confidence = np.max(probabilities) * 100

                    prediction_text = prediction.replace("_", " ")
                    confidence_text = f"Confidence: {confidence:.1f}%"

                    print("Prediction:", prediction_text)
                    print(confidence_text)

                else:
                    prediction_text = "Recording too short!"
                    confidence_text = ""

camera.release()
cv2.destroyAllWindows()