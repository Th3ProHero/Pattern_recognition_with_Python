import cv2
import numpy as np
from tkinter import Tk, filedialog
import os
import pydicom
import pylibjpeg
import matplotlib.pyplot as plt

# Seleccionar archivo desde el explorador de archivos
root = Tk()
root.withdraw()
Imagen = filedialog.askopenfilename(title='Seleccionar archivo de imagen', filetypes=[('Imagen', ['*.jpg', '*.raw', '*.tif', '*.dicom', '*.dcm'])])
root.destroy()

# Verificar si se seleccionó un archivo
if not Imagen:
    print("No se seleccionó ningún archivo.")
else:
    # Obtener información sobre el archivo
    _, ext = os.path.splitext(Imagen)
    file_size = os.path.getsize(Imagen)

    # Mostrar información
    print(f"Nombre del archivo: {os.path.basename(Imagen)}")
    print(f"Extensión del archivo: {ext}")
    print(f"Tamaño del archivo: {file_size} bytes")

    # Manejar archivos DICOM
    if ext.lower() in ['.dcm']:
        # Leer el archivo DICOM
        ds = pydicom.dcmread(Imagen)

        # Obtener información de resolución
        if 'PixelSpacing' in ds:
            print(f"Pixel Spacing: {ds.PixelSpacing}")
        
        if 'Rows' in ds and 'Columns' in ds:
            print(f"Dimensiones de la imagen DICOM: {ds.Rows} x {ds.Columns}")

        # Otros atributos relacionados con la resolución pueden imprimirse según sea necesario

        # Mostrar la imagen (si es relevante para tu caso)
        if 'PixelData' in ds:
            imagen = ds.pixel_array
            # Puedes realizar cualquier operación con la imagen, como mostrarla con plt.imshow() o cv2.imshow()

    elif ext.lower() in ['.raw']:
        # Intentar cargar la imagen RAW con numpy y Matplotlib
        try:
            # Ajustar las dimensiones según tus necesidades (800 x 600)
            ancho, alto = 800, 600
            dtype = np.uint8  # o np.uint16 si la imagen es de 16 bits

            with open(Imagen, 'rb') as f:
                img_raw = np.fromfile(f, dtype=dtype)
                img = img_raw.reshape((ancho, alto))  # o (ancho, alto) dependiendo de la orientación

                # Mostrar la imagen en escala de grises con Matplotlib
                plt.imshow(img, cmap='gray')  # cmap='jet' para una representación en color falsa
                print("Resolucion: ",ancho, alto)
                plt.show()

        except Exception as e:
            print(f"Error al cargar y mostrar la imagen RAW: {e}")

    else:
        # Mostrar imagen con OpenCV y obtener la resolución
        img = cv2.imread(Imagen)
        height, width, _ = img.shape
        print(f"Resolución de la imagen: {width} x {height}")

        cv2.imshow('Imagen', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
