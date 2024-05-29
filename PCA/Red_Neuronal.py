import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape


def load_images_from_folder(folder):
    images = []
    images_count=0
    for filename in os.listdir(folder):
        if filename.endswith(".bmp"):
            img = Image.open(os.path.join(folder, filename)).convert('L')  # Convert to grayscale
            if img is not None:
                images_count = images_count + 1
                print("IMAGENES CARGADAS:",images_count)
                images.append(np.asarray(img))
    return images

def preprocess_images(images):
    img_shape = images[0].shape
    flattened_images = [img.flatten() for img in images]
    X = np.vstack(flattened_images)
    X = X / 255.0  # Normalización a [0, 1]
    print(f"Min pixel value: {X.min()}, Max pixel value: {X.max()}")  # Verificar normalización
    return X, img_shape

folder_path = r"C:\Users\mawis\OneDrive\Escritorio\patronesfinal\normal_dataset"
images = load_images_from_folder(folder_path)
X, img_shape = preprocess_images(images)

# Definimos las dimensiones del autoencoder
input_dim = X.shape[1]
encoding_dim = 128  # Número de nodos en la capa oculta

# Crear el autoencoder
input_img = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Modelo autoencoder
autoencoder = Model(input_img, decoded)

# Modelo encoder
encoder = Model(input_img, encoded)

# Modelo decoder
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

# Compilar el autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Entrenar el autoencoder
autoencoder.fit(X, X, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)

# Reconstrucción de imágenes
encoded_imgs = encoder.predict(X)
decoded_imgs = decoder.predict(encoded_imgs)

# Mostrar imágenes originales y reconstruidas
def plot_images(original, reconstructed, img_shape, n=5):
    fig, axes = plt.subplots(2, n, figsize=(15, 5))
    for i in range(n):
        axes[0, i].imshow(original[i].reshape(img_shape), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructed[i].reshape(img_shape), cmap='gray')
        axes[1, i].axis('off')
    plt.show()

plot_images(X, decoded_imgs, img_shape)

