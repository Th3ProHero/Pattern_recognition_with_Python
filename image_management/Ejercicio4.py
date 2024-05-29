import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color
from tkinter import Tk, filedialog

# Crear una ventana de tkinter (sin mostrarla)
root = Tk()
root.withdraw()

# Solicitar al usuario que seleccione una imagen
nombre_archivo = filedialog.askopenfilename(
    title='Seleccionar archivo de imagen',
    filetypes=[
        ('Archivos de imagen', '*.tif;*.png;*.jpg;*.raw'),
        ('Todos los archivos', '*.*')
    ]
)

# Cerrar la ventana de tkinter
root.destroy()

# Verificar si se seleccionó un archivo
if not nombre_archivo:
    print("No se seleccionó ningún archivo.")
else:
    # Cargar la imagen
    imagen = io.imread(nombre_archivo)

    # Verificar si la imagen es en color (3 canales) o en escala de grises (1 canal)
    if imagen.ndim == 2:  # Imagen en escala de grises
        print("La imagen es en escala de grises.")
        paleta_rgb = None  # No hay paleta para escala de grises
    elif imagen.ndim == 3 and imagen.shape[2] == 3:  # Imagen en color (RGB)
        print("La imagen es en color (RGB).")
        paleta_rgb = imagen
    else:
        raise ValueError(f"Formato de imagen no compatible: {imagen.shape}")

    # Visualizar paleta de colores de cada canal RGB por separado
    if paleta_rgb is not None:
        canales_rgb = ['Rojo', 'Verde', 'Azul']

        fig, axs = plt.subplots(1, 4, figsize=(15, 4))

        # Visualizar la imagen completa
        axs[0].imshow(paleta_rgb)
        axs[0].set_title('Imagen completa')
        axs[0].axis('off')

        # Visualizar cada canal RGB por separado con mapa de colores 'gray'
        for i in range(3):
            canal = paleta_rgb[..., i]
            axs[i+1].imshow(canal, cmap='gray')  # Utilizar 'gray' para canales individuales
            axs[i+1].set_title(f'Canal {canales_rgb[i]}')
            axs[i+1].axis('off')

        # Mostrar la barra de colores a la derecha
        axs[-1].axis('off')
        cbar = plt.colorbar(axs[1].imshow(paleta_rgb[..., 0], cmap='gray'), ax=axs[-1])
        cbar.set_label('Valor')

        # Ajustar el diseño y mostrar la figura
        plt.tight_layout()
        plt.show()