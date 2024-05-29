import cv2
import numpy as np
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
from skimage import io, color

root = Tk()
root.withdraw()
nombre_archivo = filedialog.askopenfilename(title='Seleccionar archivo de imagen', filetypes=[('Imagen', ['*.jpg', '*.raw', '*.tif', '*.dicom', '*.dcm'])])
root.destroy()

image = cv2.imread(nombre_archivo)


image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)


image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


cv2.imshow('Escala de Grises cv2', image_gray)


cv2.imshow('YUV cv2', image_yuv)


cv2.imshow('HSV cv2', image_hsv)


cv2.waitKey(0)
cv2.destroyAllWindows()

#########################################################
#########################################################

#Cargar la imagen .tif
#Suponiendo que el archivo .tif está en un formato RGB estándar.
# Leer la imagen en formato .tif
image = io.imread(nombre_archivo)

# Verificar la forma de la imagen para asegurarse de que sea compatible con rgb2gray
if image.ndim == 2:  # La imagen ya está en escala de grises
    image_gray = image
elif image.ndim == 3 and image.shape[2] in [3, 4]:  # La imagen es RGB o RGBA
    # Convertir a escala de grises
    image_gray = color.rgb2gray(image[..., :3])  # Ignorar el canal alfa si existe
else:
    raise ValueError(f"La imagen tiene una forma inesperada: {image.shape}")

# Convertir de RGB a YUV
image_yuv = color.rgb2yuv(image[..., :3])

# Convertir de RGB a HSV
image_hsv = color.rgb2hsv(image[..., :3])

# Crear una figura con tres subgráficos
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Mostrar la imagen en escala de grises en el primer subgráfico
axs[0].imshow(image_gray, cmap='gray')
axs[0].set_title('Escala de Grises')

# Mostrar la imagen en espacio de color YUV en el segundo subgráfico
axs[1].imshow(image_yuv)
axs[1].set_title('YUV')

# Mostrar la imagen en espacio de color HSV en el tercer subgráfico
axs[2].imshow(image_hsv)
axs[2].set_title('HSV')

# Ajustar el diseño y mostrar la figura
plt.tight_layout()
plt.show()