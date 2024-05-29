import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Cargar el modelo entrenado
model = load_model('modelo_perros_gatos.h5')

# Ruta de la imagen a clasificar
img_path = 'prueba/prueba1.jpg'

# Definir el tamaño de la imagen esperado por el modelo
img_height, img_width = 128, 128

# Cargar y procesar la imagen
img = image.load_img(img_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Añadir una dimensión para el batch
img_array /= 255.0  # Normalizar la imagen

# Realizar la predicción
prediction = model.predict(img_array)

# Interpretar la predicción
if prediction[0] < 0.5:
    print(f'La imagen {img_path} es clasificada como: Gato')
else:
    print(f'La imagen {img_path} es clasificada como: Perro')
