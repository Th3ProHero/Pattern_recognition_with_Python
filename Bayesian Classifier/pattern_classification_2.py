import cv2
import numpy as np
from scipy.stats import multivariate_normal

# Cargar imagen
image_path = 'Entrenamiento1.jpg'
original_img = cv2.imread(image_path)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
display_img = np.copy(original_img)  # Imagen para mostrar en la interfaz

# Inicializar variables
results = []  # Almacenará (media, covarianza) para cada clase
priors = []  # Probabilidades a priori de cada clase
mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
drawing = False

# Asignación de valores de gris para cada clase
class_values = [0, 128, 250]  # Asegúrate de tener tantos valores como clases

def draw_mask(event, x, y, flags, param):
    global drawing, mask, display_img
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(mask, (x, y), 5, (255), -1)  # Dibuja en la máscara
        cv2.circle(display_img, (x, y), 10, (0, 255, 0), -1)  # Dibuja en verde en la imagen de visualización
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def reset_mask():
    global mask, display_img
    mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
    display_img = np.copy(original_img)  # Restaura la imagen de visualización

def classify_pixels():
    if not results:
        print("No hay clases registradas para clasificar.")
        return
    gaussian_models = [multivariate_normal(mean=mean, cov=cov) for mean, cov in results]
    classified_image = np.zeros(original_img.shape[:2], dtype=np.uint8)

    for i in range(original_img.shape[0]):
        for j in range(original_img.shape[1]):
            pixel = original_img[i, j]
            posteriors = [model.pdf(pixel) * prior for model, prior in zip(gaussian_models, priors)]
            class_index = np.argmax(posteriors)
            classified_image[i, j] = class_values[class_index]

    cv2.imshow('Classified Image', classified_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Configuración de OpenCV
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_mask)

while True:
    cv2.imshow('image', display_img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Esc para salir y clasificar
        classify_pixels()
        break
    elif key == 13:  # Enter para procesar la selección
        selected_region = original_img[mask == 255]
        if selected_region.size > 0:
            mean_colors = np.mean(selected_region.reshape(-1, 3), axis=0)
            cov_colors = np.cov(selected_region.reshape(-1, 3), rowvar=False)
            results.append((mean_colors, cov_colors))
            priors.append(np.sum(mask) / mask.size)  # A priori basado en el área seleccionada
            print(f"Clase {len(results)} registrada. Media: {mean_colors}, Covarianza: {cov_colors}")
        reset_mask()  # Preparar para nueva clase
