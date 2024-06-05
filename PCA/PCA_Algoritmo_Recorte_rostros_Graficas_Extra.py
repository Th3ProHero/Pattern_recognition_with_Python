import os
import numpy as np
from PIL import Image
import cv2  # Asegúrate de tener OpenCV instalado
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Ruta al dataset de entrenamiento
dataset_path = r'PATH'

# Rutas a las imágenes específicas a reconstruir
image_paths = [
    r'file.jpg',
    r'.jpg'
]

# Cargar Haar Cascade para la detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(image):
    print("Recortando rostros")
    gray = np.array(image)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    (x, y, w, h) = faces[0]
    face = gray[y:y+h, x:x+w]
    return face

def load_images_from_folder(folder):
    print("\nCargando imagenes desde el folder dataset\n")
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".bmp"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('L')  # Convertir a escala de grises
            face = detect_face(img)
            if face is not None:
                face = Image.fromarray(face).resize((100, 100))  # Redimensionar la imagen del rostro
                img_array = np.array(face).flatten()  # Aplanar la imagen
                images.append(img_array)
    return np.array(images)

def load_specific_images(image_paths):
    print("Cargando imagenes que serán reconstruidas\n")
    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('L')  # Convertir a escala de grises
        face = detect_face(img)
        if face is not None:
            face = Image.fromarray(face).resize((100, 100))  # Redimensionar la imagen del rostro
            img_array = np.array(face).flatten()  # Aplanar la imagen
            images.append(img_array)
    return np.array(images)

# Cargar las imágenes de entrenamiento
faces_train = load_images_from_folder(dataset_path)
print("Imágenes cargadas\n")
n_samples, n_features = faces_train.shape

# Cargar las imágenes específicas a reconstruir
faces_test = load_specific_images(image_paths)
print("Imagenes para reconstruir cargadas\n")

# Estandarizar los datos de entrenamiento
scaler = StandardScaler()
faces_train_std = scaler.fit_transform(faces_train)

# Calcular la matriz de covarianza
print("Generando matriz de covarianza\n")
cov_matrix = np.cov(faces_train_std.T)
print(len(cov_matrix))

# Realizar la descomposición en valores propios
print("Descomposicion en valores propios\n")
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Ordenar los valores y vectores propios
print("Ordenar los valores y vectores propios\n")
sorted_index = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_index]
eigenvectors = eigenvectors[:, sorted_index]

# Seleccionar los primeros k componentes principales
k = 50
eigenvectors = eigenvectors[:, :k]

# Proyectar las imágenes específicas en el nuevo espacio
faces_test_std = scaler.transform(faces_test)
faces_test_pca = faces_test_std @ eigenvectors

# Reconstruir las imágenes a partir de los componentes principales
faces_reconstructed = faces_test_pca @ eigenvectors.T
faces_reconstructed = scaler.inverse_transform(faces_reconstructed)

# Calcular el MSE para cada imagen (implementación propia)
mse_reconstructed = [mean_squared_error(faces_test[i], faces_reconstructed[i]) for i in range(len(faces_test))]
average_mse_reconstructed = np.mean(mse_reconstructed)
print(f"Error medio cuadrático de las imágenes reconstruidas (propia implementación): {average_mse_reconstructed}")

# Utilizar PCA de scikit-learn para comparación
pca = PCA(n_components=k)
pca.fit(faces_train_std)
faces_test_pca_sklearn = pca.transform(faces_test_std)
faces_reconstructed_sklearn = pca.inverse_transform(faces_test_pca_sklearn)
faces_reconstructed_sklearn = scaler.inverse_transform(faces_reconstructed_sklearn)

# Calcular el MSE para cada imagen (sklearn)
mse_reconstructed_sklearn = [mean_squared_error(faces_test[i], faces_reconstructed_sklearn[i]) for i in range(len(faces_test))]
average_mse_reconstructed_sklearn = np.mean(mse_reconstructed_sklearn)
print(f"Error medio cuadrático de las imágenes reconstruidas (sklearn): {average_mse_reconstructed_sklearn}")

# Visualizar las imágenes originales y reconstruidas
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

for i in range(2):
    ax[i, 0].imshow(faces_test[i].reshape((100, 100)), cmap='gray')
    ax[i, 0].set_title('Original')
    ax[i, 0].axis('off')

    ax[i, 1].imshow(faces_reconstructed[i].reshape((100, 100)), cmap='gray')
    ax[i, 1].set_title('Reconstruida')
    ax[i, 1].axis('off')

plt.show()

# Visualizar las imágenes reconstruidas con scikit-learn PCA
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

for i in range(2):
    ax[i, 0].imshow(faces_test[i].reshape((100, 100)), cmap='gray')
    ax[i, 0].set_title('Original')
    ax[i, 0].axis('off')

    ax[i, 1].imshow(faces_reconstructed_sklearn[i].reshape((100, 100)), cmap='gray')
    ax[i, 1].set_title('Reconstruida (sklearn)')
    ax[i, 1].axis('off')

plt.show()

explained_variance_ratio = np.cumsum(eigenvalues / np.sum(eigenvalues))

plt.figure(figsize=(8, 5))
plt.plot(explained_variance_ratio, marker='o')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Proporción de Varianza Explicada')
plt.title('Proporción de Varianza Explicada por los Componentes Principales')
plt.grid(True)
plt.show()

# Visualización de los primeros componentes principales
num_components_to_show = 10
fig, axes = plt.subplots(1, num_components_to_show, figsize=(20, 4))
for i in range(num_components_to_show):
    ax = axes[i]
    ax.imshow(eigenvectors[:, i].reshape(100, 100), cmap='gray')
    ax.set_title(f'Componente {i+1}')
    ax.axis('off')
plt.suptitle('Primeros Componentes Principales')
plt.show()

# Comparación de las matrices de covarianza
cov_matrix_test = np.cov(faces_test_std.T)
cov_matrix_reconstructed = np.cov(scaler.transform(faces_reconstructed).T)
cov_matrix_reconstructed_sklearn = np.cov(scaler.transform(faces_reconstructed_sklearn).T)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(cov_matrix_test, cmap='hot', interpolation='nearest')
plt.title('Covarianza Original')
plt.colorbar()
plt.subplot(1, 3, 2)
plt.imshow(cov_matrix_reconstructed, cmap='hot', interpolation='nearest')
plt.title('Covarianza Reconstruida (Propia)')
plt.colorbar()
plt.subplot(1, 3, 3)
plt.imshow(cov_matrix_reconstructed_sklearn, cmap='hot', interpolation='nearest')
plt.title('Covarianza Reconstruida (sklearn)')
plt.colorbar()
plt.suptitle('Comparación de Matrices de Covarianza')
plt.show()
