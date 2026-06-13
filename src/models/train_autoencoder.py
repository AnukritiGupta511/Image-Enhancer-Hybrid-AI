# Train autoencoder

import numpy as np
import os
from PIL import Image
from tensorflow.keras.optimizers import Adam
from autoencoder_model import build_autoencoder

IMG_SIZE = 128

def load_images(path):
    images = []
    for file in os.listdir(path):
        img = Image.open(os.path.join(path, file)).resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img) / 255.0
        images.append(img)
    return np.array(images)

def train():
    images = load_images("data/raw/normal")

    # create low quality images
    low_quality = images.copy()

    model = build_autoencoder()
    model.compile(optimizer=Adam(), loss='mse')

    model.fit(low_quality, images, epochs=10, batch_size=8)

    model.save("models/autoencoder.h5")
    print("Autoencoder Saved!")

if __name__ == "__main__":
    train()