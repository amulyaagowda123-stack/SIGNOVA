import cv2
import mediapipe as mp
import time

# Initialize MediaPipe
BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Model location
MODEL_PATH = "ml_model/models/holistic_landmarker.task"

# Configure Holistic model
options = HolisticLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    min_face_detection_confidence=0.5,
    min_pose_detection_confidence=0.5,
    min_hand_landmarks_confidence=0.5
)

# Draw detected landmarks
def draw_points(frame, landmarks):

    if landmarks:
        height, width, _ = frame.shape

        for point in landmarks:
            x = int(point.x * width)
            y = int(point.y * height)

            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)


# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not detected!")
    exit()

print("SIGNOVA Holistic Detection Started!")

with HolisticLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = camera.read()

        if not success:
            print("Cannot read webcam!")
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp = time.monotonic_ns() // 1_000_000

        results = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        # Face detection
        draw_points(frame, results.face_landmarks)

        # Body detection
        draw_points(frame, results.pose_landmarks)

        # Left hand
        draw_points(frame, results.left_hand_landmarks)

        # Right hand
        draw_points(frame, results.right_hand_landmarks)

        cv2.putText(
            frame,
            "SIGNOVA - AI HOLISTIC DETECTION",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("SIGNOVA", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()

print("SIGNOVA stopped!")