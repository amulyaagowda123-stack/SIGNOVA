import numpy as np
import joblib
import os
import glob

from train import preprocess_sequence

MODEL_PATH = "ml_model/models/sentence_model.pkl"

DATASET_PATH = "ml_model/dataset/sentences/i_am_in_danger"

model = joblib.load(MODEL_PATH)

files = glob.glob(os.path.join(DATASET_PATH, "*.npy"))

print("Testing saved emergency recordings...")

for file in files[:10]:

    data = np.load(file)

    processed = preprocess_sequence(data).reshape(1, -1)

    prediction = model.predict(processed)[0]

    confidence = max(model.predict_proba(processed)[0]) * 100

    print(
        os.path.basename(file),
        "->",
        prediction,
        round(confidence, 2),
        "%"
    )