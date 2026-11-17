import os
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

DATASET_PATH = "ml_model/dataset/sentences"
MODEL_PATH = "ml_model/models/signova_model.pkl"

X = []
y = []

# Load recordings
for label in os.listdir(DATASET_PATH):

    folder = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(folder):
        continue

    for filename in os.listdir(folder):

        if filename.endswith(".npy"):

            data = np.load(os.path.join(folder, filename))

            # Convert each recording into 20 time steps
            indices = np.linspace(
                0, len(data) - 1, 20
            ).astype(int)

            features = data[indices].flatten()

            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Total recordings:", len(X))
print("Sentence classes:", np.unique(y))

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Build recognition model
model = make_pipeline(
    StandardScaler(),
    PCA(n_components=10),
    SVC(kernel="rbf", probability=True, class_weight="balanced")
)

# Train
print("\nTraining SIGNOVA model...")
model.fit(X_train, y_train)

# Test
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n===== SIGNOVA RESULTS =====")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save model
joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully!")