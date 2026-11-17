import numpy as np
import os

dataset_path = os.path.join(os.path.dirname(__file__), "dataset")

print("SIGNOVA DATASET CHECK")
print("=====================")

for root, dirs, files in os.walk(dataset_path):

    for file in files:

        if file.endswith(".npy"):

            path = os.path.join(root, file)

            data = np.load(path)

            print("Category:", os.path.basename(root))
            print("File:", file)
            print("Shape:", data.shape)
            print("-------------------")

            break