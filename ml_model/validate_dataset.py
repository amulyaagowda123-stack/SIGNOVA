import os
import numpy as np

DATASET_PATH = "ml_model/dataset/sentences"

print("\n===== SIGNOVA DATASET VALIDATION =====\n")

total_files = 0
valid_files = 0
invalid_files = 0

for label in os.listdir(DATASET_PATH):

    folder_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(folder_path):
        continue

    print(f"\nSentence: {label}")

    files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]

    print("Total recordings:", len(files))

    for file in files:

        file_path = os.path.join(folder_path, file)
        total_files += 1

        try:
            data = np.load(file_path)

            if data.ndim != 2:
                raise ValueError("Incorrect dimensions")

            if data.shape[0] == 0:
                raise ValueError("Empty recording")

            if not np.isfinite(data).all():
                raise ValueError("Invalid numerical values")

            valid_files += 1
            print(f"  VALID: {file} | Shape: {data.shape}")

        except Exception as error:
            invalid_files += 1
            print(f"  INVALID: {file} | {error}")

print("\n===== FINAL REPORT =====")
print("Total recordings:", total_files)
print("Valid recordings:", valid_files)
print("Invalid recordings:", invalid_files)

if invalid_files == 0 and total_files > 0:
    print("\nAll recordings passed basic validation!")
else:
    print("\nPlease check the invalid recordings.")