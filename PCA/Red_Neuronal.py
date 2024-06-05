import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import History
from sklearn.metrics import mean_squared_error

def load_images_from_folder(folder):
    images = []
    images_count = 0
    for filename in os.listdir(folder):
        if filename.endswith(".bmp"):
            img = Image.open(os.path.join(folder, filename)).convert('L')  # Convert to grayscale
            if img is not None:
                images_count += 1
                print(f"IMÁGENES CARGADAS: {images_count}")
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
history = History()
autoencoder.fit(X, X, epochs=50, batch_size=256, shuffle=True, validation_split=0.2, callbacks=[history])

# Mostrar la evolución de la pérdida durante el entrenamiento
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Evolución de la Pérdida durante el Entrenamiento')
plt.grid(True)
plt.show()

# Reconstrucción de imágenes
encoded_imgs = encoder.predict(X)
decoded_imgs = decoder.predict(encoded_imgs)

# Calcular el error de reconstrucción
reconstruction_errors = [mean_squared_error(X[i], decoded_imgs[i]) for i in range(len(X))]

# Visualizar imágenes originales y reconstruidas con sus respectivos errores
def plot_images_with_errors(original, reconstructed, errors, img_shape, n=5):
    fig, axes = plt.subplots(3, n, figsize=(15, 8))
    for i in range(n):
        axes[0, i].imshow(original[i].reshape(img_shape), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')
        axes[1, i].imshow(reconstructed[i].reshape(img_shape), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Reconstruida\nError: {errors[i]:.4f}')
        axes[2, i].hist(original[i] - reconstructed[i], bins=30, color='gray')
        axes[2, i].set_title('Distribución de Errores')
    plt.suptitle('Imágenes Originales y Reconstruidas con Errores')
    plt.show()

plot_images_with_errors(X, decoded_imgs, reconstruction_errors, img_shape)

# Visualizar la distribución de los errores de reconstrucción
plt.figure(figsize=(10, 5))
plt.hist(reconstruction_errors, bins=30, color='blue', alpha=0.7)
plt.xlabel('Error de Reconstrucción')
plt.ylabel('Frecuencia')
plt.title('Distribución de los Errores de Reconstrucción')
plt.grid(True)
plt.show()
