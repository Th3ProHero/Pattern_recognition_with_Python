import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance

def preprocess_image(image):
    # Convertir a espacio de color HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Aplicar un filtro Gaussiano para suavizar la imagen
    blurred_image = cv2.GaussianBlur(hsv_image, (7, 7), 0)

    return blurred_image

def apply_morphology(mask):
    # Definir un kernel para las operaciones morfológicas
    kernel = np.ones((5, 5), np.uint8)

    # Cerrar pequeños huecos con una operación de cierre
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Eliminar ruido con una operación de apertura
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    return mask_clean
def mahalanobis_distance(x, y, cov):
    if not cov.size or np.linalg.det(cov) == 0:
        return float('inf')  # Evitar la inversión de una matriz singular
    cov_inv = np.linalg.inv(cov)
    return distance.mahalanobis(x, y, cov_inv)

def select_image():
    # Initialize the tkinter GUI
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename()  # Open file dialog and return the chosen file path
    return file_path

def plot_pca(selected_pixels, title='PCA de Colores'):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(selected_pixels)
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    plt.title(title)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.show()

def plot_covariance_matrix(covariance_matrix, title='Matriz de Covarianza'):
    sns.heatmap(covariance_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                xticklabels=['R', 'G', 'B'], yticklabels=['R', 'G', 'B'])
    plt.title(title)
    plt.show()

def plot_histogram(image):
    color = ('b', 'g', 'r')
    hist_data = []
    for i, col in enumerate(color):
        histr = cv2.calcHist([image], [i], None, [256], [0, 256])
        hist_data.append(histr)
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    for i, histr in enumerate(hist_data):
        plt.plot(255, histr[255], 'ko')
    plt.title('Histograma de colores con píxeles blancos destacados')
    plt.xlabel('Intensidad de pixel')
    plt.ylabel('Cantidad de pixeles')
    plt.show()

def segment_color(image, lower_color, upper_color):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_color, upper_color)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_image = image.copy()
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 3)
    return result_image, mask, contours

def calculate_covariance(pixels):
    if pixels.size == 0:
        # No pixels, return zero-variance and zero-covariance.
        return np.zeros(3), np.zeros((3, 3))
    if pixels.shape[0] == 1:
        # Only one pixel, variance and covariance are necessarily zero.
        return pixels[0], np.zeros((3, 3))
    
    mean_color = np.mean(pixels, axis=0)
    covariance_matrix = np.cov(pixels, rowvar=False)
    
    return mean_color, covariance_matrix

def analyze_image(image_path, color_bounds):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error al cargar la imagen: {image_path}")
        return None

    results = {}
    for object_type, (lower, upper) in color_bounds.items():
        segmented, mask, _ = segment_color(image, lower, upper)
        pixels = image[mask.astype(bool)].reshape(-1, 3)
        mean, cov = calculate_covariance(pixels)
        var = np.var(pixels, axis=0)
        results[object_type] = {'mean': mean, 'covariance': cov, 'variance': var}
    return results

color_bounds = {
    'banana': (np.array([20, 100, 100]), np.array([30, 255, 255])),
    'chili': (np.array([35, 50, 50]), np.array([85, 255, 255])),
    'egg_white': (np.array([0, 0, 200]), np.array([180, 10, 255]))
}

def create_feature_database(image_paths, color_bounds):
    database = {key: {'variance': [], 'covariance': []} for key in color_bounds.keys()}
    print("Creando base de datos de características:")
    for path in image_paths:
        print(f"Procesando {path}...")
        results = analyze_image(path, color_bounds)
        if results:
            for key in results:
                database[key]['variance'].append(results[key]['variance'])
                database[key]['covariance'].append(results[key]['covariance'])
                print(f"{key.capitalize()} - Varianza: {results[key]['variance']}, Covarianza: {results[key]['covariance'].flatten()}")
    for key in database:
        database[key]['variance'] = np.mean(database[key]['variance'], axis=0)
        database[key]['covariance'] = np.mean(database[key]['covariance'], axis=0)
    return database

def compare_with_database(image_path, database, color_bounds):
    results = analyze_image(image_path, color_bounds)
    if not results:
        print("No se pudo analizar la imagen.")
        return

    # Calculando la distancia máxima posible para normalización
    max_distance = 0
    for key in database:
        zero_vector = np.zeros_like(database[key]['variance'])
        max_dist_for_key = mahalanobis_distance(zero_vector, database[key]['variance'], database[key]['covariance'])
        max_distance = max(max_distance, max_dist_for_key)

    print("Comparando con base de datos:")
    found_any = False
    for key in results:
        img_var = results[key]['variance'].flatten()
        db_var = database[key]['variance'].flatten()
        db_cov = database[key]['covariance']

        if img_var.size == 0 or db_var.size == 0 or db_cov.size == 0 or np.linalg.det(db_cov) == 0:
            print(f"No hay datos suficientes para {key} o la matriz de covarianza es singular.")
            continue

        var_dist = mahalanobis_distance(img_var, db_var, db_cov)
        normalized_dist = var_dist / max_distance  # Normalización de la distancia
        similarity = np.exp(-normalized_dist) * 100  # Convertir distancia en similitud

        if similarity < 20:
            print(f"No se encontró {key.capitalize()} significativo en la imagen.")
        else:
            print(f"Distancia de Mahalanobis para {key}: {var_dist:.2f}")
            print(f"Similitud de {key.capitalize()} estimada en: {similarity:.2f}%")
            found_any = True

    if not found_any:
        print("No se detectaron objetos conocidos en la imagen.")  
                     
def segment_and_display(image_path, color_bounds):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error al cargar la imagen: {image_path}")
        return

    for object_type, (lower, upper) in color_bounds.items():
        _, mask, _ = segment_color(image, lower, upper)
        result_image, _, _ = segment_color(image, lower, upper)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Segmented {object_type.capitalize()}')
        plt.show()
        
# Ejemplo de uso
if __name__ == "__main__":
    train_image_paths = [
        'C:/Users/mawis/OneDrive/Escritorio/Patrones/Entrenamiento1.jpg',
        'C:/Users/mawis/OneDrive/Escritorio/Patrones/Entrenamiento2.jpg',
        'C:/Users/mawis/OneDrive/Escritorio/Patrones/Entrenamiento3.jpg',
        'C:/Users/mawis/OneDrive/Escritorio/Patrones/Entrenamiento4.jpg'
    ]
    
    banana_count_per_image = 2
    egg_count_per_image = 3
    chili_count_per_image = 3
    total_images = len(train_image_paths)

    # Calcula el recuento total para cada objeto
    banana_total_count = banana_count_per_image * total_images
    egg_total_count = egg_count_per_image * total_images
    chili_total_count = chili_count_per_image * total_images
    
    feature_database = create_feature_database(train_image_paths, color_bounds)
    total_objects_count = banana_total_count + egg_total_count + chili_total_count
    # Valores a priori
    priors = {
        'banana': banana_total_count / total_objects_count,
        'egg_white': egg_total_count / total_objects_count,
        'chili': chili_total_count / total_objects_count
    }
    
    test_image_path = select_image()
    if test_image_path:
        original_image = cv2.imread(test_image_path)
        preprocessed_image = preprocess_image(original_image)
        compare_with_database(test_image_path, feature_database, color_bounds)
        segment_and_display(test_image_path, color_bounds)
        # Imprimir los valores a priori
        print("\nValores a priori basados en el conjunto de entrenamiento:")
        for object_type, prior in priors.items():
            print(f"{object_type.capitalize()}: {prior:.3f}")
    else:
        print("No image selected.")
