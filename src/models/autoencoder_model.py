# Build Autoencoder model

from tensorflow.keras import layers, models

def build_autoencoder():
    input_img = layers.Input(shape=(128,128,3))

    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)

    decoded = layers.Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)

    model = models.Model(input_img, decoded)
    return model