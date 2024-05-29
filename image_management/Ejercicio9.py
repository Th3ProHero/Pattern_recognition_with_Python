import os
import pydicom
import numpy as np
import imageio
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt

# Crear una ventana de tkinter (sin mostrarla)
root = Tk()
root.withdraw()

# Seleccionar la carpeta que contiene los archivos .dcm
folder_path = filedialog.askdirectory(title='Seleccionar carpeta con archivos DICOM')

# Cerrar la ventana de tkinter
root.destroy()

# Verificar si se seleccionó una carpeta
if not folder_path:
    print("No se seleccionó ninguna carpeta.")
else:
    # Obtener la lista de archivos .dcm en la carpeta
    dcm_files = [file for file in os.listdir(folder_path) if file.lower().endswith('.dcm')]

    # Ordenar los archivos por nombre
    dcm_files.sort()

    # Crear una lista para almacenar las imágenes DICOM
    dicom_images = []

    # Leer cada archivo DICOM y agregarlo a la lista
    for dcm_file in dcm_files:
        dcm_path = os.path.join(folder_path, dcm_file)
        ds = pydicom.dcmread(dcm_path)
        dicom_images.append(ds.pixel_array)

    # Convertir la lista de imágenes DICOM en un arreglo numpy 3D
    dicom_array = np.stack(dicom_images, axis=-1)

    # Normalizar los valores de píxeles entre 0 y 255
    dicom_array = ((dicom_array - dicom_array.min()) / (dicom_array.max() - dicom_array.min()) * 255).astype(np.uint8)

    # Guardar el gif
    gif_path = os.path.join(folder_path, 'output.gif')
    imageio.mimsave(gif_path, dicom_array.transpose(2, 0, 1), duration=0.1)  # Transponer para cambiar el orden de los ejes

    # Mostrar el gif utilizando matplotlib
    with imageio.get_reader(gif_path) as reader:
        for i, frame in enumerate(reader):
            plt.imshow(frame, cmap='gray')
            plt.title(f'Frame {i + 1}')
            plt.pause(0.001)  # Tiempo de pausa más corto para una reproducción más rápida
            plt.draw()

    plt.show()