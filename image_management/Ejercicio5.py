from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from tkinter import Tk, filedialog

def cargar_y_preparar_imagen(ruta_imagen):
    # Cargar la imagen
    imagen = Image.open(ruta_imagen)
    # Convertir a escala de grises
    imagen_gris = imagen.convert("L")
    # Asegurarse que es cuadrada recortando el lado más largo
    lado_corto = min(imagen_gris.size)
    imagen_cuadrada = imagen_gris.crop((0, 0, lado_corto, lado_corto))
    return imagen_cuadrada

def decimar_imagen(imagen):
    # Convertir la imagen a una matriz numpy
    arr = np.array(imagen)
    # Reducir a la mitad del tamaño mediante el promedio de grupos de 4 pixeles
    arr_decimada = arr.reshape((arr.shape[0]//2, 2, arr.shape[1]//2, 2)).mean(axis=(1, 3))
    # Convertir de nuevo a una imagen PIL
    imagen_decimada = Image.fromarray(arr_decimada.astype(np.uint8))
    return imagen_decimada

#Ruta de la imagen a procesar
ruta_imagen = filedialog.askopenfilename(
    title='Seleccionar archivo de imagen',
    filetypes=[
        ('Archivos de imagen', '*.tif;*.png;*.jpg;*.raw'),
        ('Todos los archivos', '*.*')
    ]
)

#Cargar, preparar y decimar la imagen
imagen_preparada = cargar_y_preparar_imagen(ruta_imagen)
imagen_decimada = decimar_imagen(imagen_preparada)

#Imprimir las resoluciones
print(f"Resolución de la imagen preparada: {imagen_preparada.size} (ancho x alto)")
print(f"Resolución de la imagen decimada: {imagen_decimada.size} (ancho x alto)")

#Mostrar la imagen original y la imagen decimada
imagen_preparada.show()
imagen_decimada.show()

#Guardar la imagen decimada si lo deseas
imagen_decimada.save('imagen_decimada.jpg')