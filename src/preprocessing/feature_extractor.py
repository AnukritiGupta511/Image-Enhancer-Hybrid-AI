from skimage.feature import hog
import numpy as np

def extract_features(image):
    image = image.resize((64, 64)).convert("L")
    image = np.array(image)

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    return features