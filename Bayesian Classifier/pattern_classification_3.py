import cv2
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# Cargar imagen
image_path = 'Entrenamiento1.jpg'
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
display_img = np.copy(img)  # Imagen para mostrar en la interfaz

masks = []  # Lista para guardar máscaras
results = []  # Guardar medias y covarianzas si es necesario
current_mask = np.zeros(img.shape[:2], dtype=np.uint8)  # Máscara actual
drawing = False  # Estado de si está dibujando o no

def draw_mask(event, x, y, flags, param):
    global drawing, current_mask, display_img
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(current_mask, (x, y), 5, (255), -1)
        cv2.circle(display_img, (x, y), 10, (0, 255, 0), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def reset_mask():
    global current_mask, display_img
    current_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    display_img = np.copy(img)

def classify_pixels(X_train, y_train):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    flat_image = img.reshape(-1, 3)
    predictions = gnb.predict(flat_image)
    classified_image = predictions.reshape(img.shape[:2])

    # Convertir classified_image a uint8 y escalar adecuadamente
    classified_image = np.uint8(classified_image)
    unique_labels = np.unique(classified_image)
    scale_factor = 255 / unique_labels.max() if unique_labels.max() != 0 else 1
    classified_image = np.uint8(classified_image * scale_factor)

    cv2.imshow('Classified Image', classified_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Configuración de OpenCV
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_mask)

X_train = []
y_train = []

while True:
    cv2.imshow('image', display_img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Esc para salir y clasificar
        if len(X_train) > 0:
            classify_pixels(X_train, y_train)
        break
    elif key == 13:  # Enter para guardar la selección actual
        if np.any(current_mask):
            masks.append(current_mask)
            selected_pixels = img[current_mask == 255]
            X_train.extend(selected_pixels)
            y_train.extend([len(masks)] * len(selected_pixels))  # Usar el índice de la máscara como etiqueta
            print(f"Clase {len(masks)} registrada con {len(selected_pixels)} píxeles.")
            reset_mask()  # Preparar para nueva clase
