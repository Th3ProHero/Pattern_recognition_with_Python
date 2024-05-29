from PIL import Image
import os

# Define la ruta base
base_path = "textura"

# Configuración de las imágenes y carpetas
imagenes_carpetas = {
    "D6.bmp": "Ladrillo",
    "D16.bmp": "Tela",
    "D46.bmp": "Malla",
    "D49.bmp": "Madera",
    "D64.bmp": "Azulejo",
    "D101.bmp": "Circulos",
    "Piedras.jpg": "Roca",
    "Piedras3.jpg": "Guijarro"
}

# Tamaño de los fragmentos
fragment_size = 100

# Número de fragmentos a extraer
num_fragments = 4

# Función para extraer fragmentos y guardarlos
def extraer_fragmentos_y_guardar(imagen, carpeta, nombre_base):
    img = Image.open(os.path.join(base_path, imagen))
    width, height = img.size
    
    # Crea la carpeta si no existe
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
    
    # Extraer fragmentos
    for i in range(num_fragments):
        # Calcular la posición aleatoria del fragmento
        x = (i * fragment_size) % (width - fragment_size)
        y = ((i * fragment_size) // (width - fragment_size)) * fragment_size % (height - fragment_size)

        # Recortar el fragmento
        fragmento = img.crop((x, y, x + fragment_size, y + fragment_size))

        # Guardar el fragmento
        fragmento.save(os.path.join(carpeta, f"{nombre_base}{i + 1}.bmp"))

# Procesar cada imagen
for imagen, carpeta in imagenes_carpetas.items():
    carpeta_destino = os.path.join(base_path, carpeta)
    extraer_fragmentos_y_guardar(imagen, carpeta_destino, "Fragmento")

print("Fragmentos extraídos y guardados correctamente.")
