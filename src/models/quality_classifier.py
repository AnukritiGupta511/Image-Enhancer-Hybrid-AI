from sklearn import svm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PIL import Image
from preprocessing.feature_extractor import extract_features
import joblib
import numpy as np


def train_classifier(data_path):
    X, Y = [], []

    classes = {
        "normal\\images from high": 0,
        "blurry\\images": 1
    }

    for folder, label in classes.items():
        folder_path = os.path.join(data_path, folder)

        if not os.path.exists(folder_path):
            print("Folder not found:", folder_path)
            continue

        for file in os.listdir(folder_path):
            try:
                img_path = os.path.join(folder_path, file)

                if os.path.isdir(img_path):
                    continue

                if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                print("Processing:", img_path)
                img = Image.open(img_path)
                print("Extracting features...")
                features = extract_features(img)
                print("Feature extraction completed")

                if features is None or len(features) == 0:
                    print("Skipped:", file)
                    continue

                X.append(features)
                Y.append(label)

            except Exception as e:
                print("Error:", file, e)

    X = np.array(X)
    Y = np.array(Y)

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)


    if len(X) == 0:
        print("No images loaded!")
        return

    model = svm.SVC()
    model.fit(X, Y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "src/models/svm_quality.pkl")

    print("SVM Model Saved")


if __name__ == "__main__":
    train_classifier("data/raw/")  
    