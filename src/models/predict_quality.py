import sys
import os
import joblib
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.feature_extractor import extract_features

# Load trained model
model = joblib.load("src/models/svm_quality.pkl")

# Test image path
img_path = "src/models/abd.jpg"   # change this

img = Image.open(img_path)
features = extract_features(img)

prediction = model.predict([features])[0]

if prediction == 0:
    print("Normal Image")
elif prediction == 1:
    print("Blurry Image")
else:
    print("Unknown")