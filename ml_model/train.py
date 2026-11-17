
import numpy as np
import os
import glob
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "sentence_model.pkl"
)

LABELS = [
    "i_need_help",
    "thank_you",
    "i_need_water",
    "i_am_in_danger",
    "i_need_food"
]

FRAME_COUNT = 30


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

    return resized.flatten()


X = []
y = []

print("Loading SIGNOVA dataset...")

for label in LABELS:

    nested_folder = os.path.join(
        BASE_DIR, "dataset", "sentences", label
    )

    root_folder = os.path.join(
        BASE_DIR, "dataset", label
    )

    # Prefer nested folder when it exists
    folder = (
        nested_folder
        if os.path.isdir(nested_folder)
        else root_folder
    )

    files = glob.glob(os.path.join(folder, "*.npy"))

    print(label, ":", len(files), "recordings")

    for file in files:

        data = np.load(file)

        if data.ndim != 2 or data.shape[1] != 1659:
            print("Skipping invalid file:", file)
            continue

        if len(data) < 2:
            continue

        X.append(preprocess_sequence(data))
        y.append(label)


X = np.array(X)
y = np.array(y)

print("\nTotal recordings:", len(X))
print("Input shape:", X.shape)

if len(set(y)) != len(LABELS):
    raise ValueError(
        "Some sentence categories have no valid recordings."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", probability=True)
)

print("\nTraining SIGNOVA...")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)

joblib.dump(model, MODEL_PATH)

print("\nSIGNOVA model trained successfully!")
print("Model saved at:", MODEL_PATH)