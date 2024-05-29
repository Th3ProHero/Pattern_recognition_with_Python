import os
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from tkinter import Tk, filedialog

# Configuración de las carpetas e imágenes
base_path = "textura"
carpetas = ["Ladrillo", "Tela", "Malla", "Madera", "Azulejo", "Circulos", "Guijarro", "Roca"]

# Parámetros para las matrices GLCM
distancias = [1, 3]  # Distancias a probar
angulos = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Ángulos a probar
propiedades = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]

# Función para calcular características GLCM
def calcular_caracteristicas_glcm(img, distancias, angulos, propiedades):
    glcm = graycomatrix(img, distances=distancias, angles=angulos, levels=256, symmetric=True, normed=True)
    caracteristicas = []
    for prop in propiedades:
        prop_valores = graycoprops(glcm, prop)
        caracteristicas.extend(prop_valores.flatten().tolist())
    return caracteristicas

# Generar encabezados dinámicamente para las características
columnas_glcm = []
for distancia in distancias:
    for angulo in ['0', '45', '90', '135']:
        for prop in propiedades:
            columnas_glcm.append(f"{prop}_{angulo}deg_d{distancia}")

# Función para calcular la entropía
def calcular_entropia(column):
    probabilidad = column.value_counts(normalize=True)
    return entropy(probabilidad)

# Función para calcular características adicionales (media, entropía, energía)
def calcular_caracteristicas_adicionales(df):
    means = df.mean(axis=0)
    entropies = df.apply(calcular_entropia, axis=0)
    energies = (df ** 2).sum(axis=0)
    stats_df = pd.DataFrame({
        'mean': means,
        'entropy': entropies,
        'energy': energies
    }).T
    return stats_df

# Función para calcular características GLCM y estadísticas adicionales para cada fragmento
def calcular_caracteristicas_completas(fragmento, distancias, angulos, propiedades):
    caracteristicas_glcm = calcular_caracteristicas_glcm(fragmento, distancias, angulos, propiedades)
    df_temp = pd.DataFrame([caracteristicas_glcm], columns=columnas_glcm)
    stats_df = calcular_caracteristicas_adicionales(df_temp)
    return caracteristicas_glcm + stats_df.values.flatten().tolist()

# Extraer características para cada carpeta
datos = []
for carpeta in carpetas:
    carpeta_path = os.path.join(base_path, carpeta)
    if os.path.exists(carpeta_path):
        for archivo in os.listdir(carpeta_path):
            if archivo.lower().endswith('.bmp'):
                img_path = os.path.join(carpeta_path, archivo)
                img = Image.open(img_path).convert('L')
                img_np = np.array(img)

                # Calcular las características de GLCM para las distintas combinaciones de distancia y ángulo
                caracteristicas_glcm = calcular_caracteristicas_glcm(img_np, distancias, angulos, propiedades)
                df_temp = pd.DataFrame([caracteristicas_glcm], columns=columnas_glcm)
                stats_df = calcular_caracteristicas_adicionales(df_temp)
                fila_datos = [carpeta, archivo] + caracteristicas_glcm + stats_df.values.flatten().tolist()
                datos.append(fila_datos)

# Generar los encabezados para el DataFrame
columnas_adicionales = ['mean', 'entropy', 'energy']
columnas_adicionales = [f"{col}_{idx}" for idx in range(len(columnas_glcm)) for col in columnas_adicionales]
columnas = ["Carpeta", "Imagen"] + columnas_glcm + columnas_adicionales

# Crear el DataFrame con los resultados
df = pd.DataFrame(datos, columns=columnas)

# Guardar el DataFrame a un archivo CSV
csv_path = os.path.join(base_path, "caracteristicas_glcm.csv")
df.to_csv(csv_path, index=False)

print(f"Características GLCM guardadas en {csv_path}")

# Paso 1: Preparar los Datos
# Leer datos desde el CSV
df = pd.read_csv(csv_path)

# Suponiendo que la columna 'Carpeta' es el target
X = df.drop(['Carpeta', 'Imagen'], axis=1)
y = df['Carpeta']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Paso 2: Entrenar los Clasificadores
# Inicializar clasificadores
knn = KNeighborsClassifier(n_neighbors=3)
svm = SVC(kernel='linear')

# Entrenar K-NN
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Entrenar SVM
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# Resultados
print("Resultados K-NN:")
print(classification_report(y_test, y_pred_knn))

print("Resultados SVM:")
print(classification_report(y_test, y_pred_svm))

# Función para clasificar los fragmentos de una imagen
def fragmentar_y_clasificar(imagen_path, clasificadores, scaler, distancias, angulos, propiedades, num_caracteristicas_esperadas):
    # Cargar la imagen en escala de grises
    img = Image.open(imagen_path).convert('L')
    img_np = np.array(img)
    height, width = img_np.shape

    # Dividir la imagen en cuatro fragmentos (2x2)
    fragmentos = [
        img_np[0:height//2, 0:width//2],         # Esquina superior izquierda
        img_np[0:height//2, width//2:width],     # Esquina superior derecha
        img_np[height//2:height, 0:width//2],    # Esquina inferior izquierda
        img_np[height//2:height, width//2:width] # Esquina inferior derecha
    ]

    # Clasificar cada fragmento
    resultados = []
    for i, fragmento in enumerate(fragmentos):
        caracteristicas = calcular_caracteristicas_completas(fragmento, distancias, angulos, propiedades)

        # Verificar que el número de características sea el esperado
        if len(caracteristicas) == num_caracteristicas_esperadas:
            # Normalizar las características
            caracteristicas = scaler.transform([caracteristicas])

            # Clasificar con los modelos proporcionados
            resultados.append({
                "Fragmento": i + 1,
                "KNN": clasificadores['knn'].predict(caracteristicas)[0],
                "SVM": clasificadores['svm'].predict(caracteristicas)[0]
            })
        else:
            resultados.append({
                "Fragmento": i + 1,
                "KNN": "Número de características no coincide",
                "SVM": "Número de características no coincide"
            })

    return resultados

# Permitir al usuario elegir la imagen desde el explorador de archivos
Tk().withdraw()  # Ocultar la ventana principal de Tkinter
imagen_path = filedialog.askopenfilename(
    title="Seleccionar Imagen",
    filetypes=[("Imagen PNG", "*.png"), ("Todos los archivos", "*.*")]
)

# Asegurarse de que se ha seleccionado un archivo
if imagen_path:
    # Clasificar fragmentos
    clasificadores = {'knn': knn, 'svm': svm}
    num_caracteristicas_esperadas = X_train.shape[1]
    resultados = fragmentar_y_clasificar(imagen_path, clasificadores, scaler, distancias, angulos, propiedades, num_caracteristicas_esperadas)

    # Mostrar los resultados
    for resultado in resultados:
        print(f"Fragmento {resultado['Fragmento']}:")
        print(f"  - KNN Predicción: {resultado['KNN']}")
        print(f"  - SVM Predicción: {resultado['SVM']}")
