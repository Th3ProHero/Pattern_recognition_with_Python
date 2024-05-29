import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define la ruta del archivo iris.data en la ubicación actual del script
#ruta_actual = 'EjerciciosIris/iris.data'  # Ruta del archivo en la carpeta EjerciciosIris
ruta_actual = 'iris.data'  # Ruta del archivo en la carpeta EjerciciosIris

# Usa una de las siguientes líneas, comentando o descomentando según sea necesario
iris_df = pd.read_csv(ruta_actual, header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

# Extraer características y etiquetas directamente del DataFrame
X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris_df['species']

# Crear un DataFrame de pandas
iris_data = pd.DataFrame(data=np.c_[X, y], columns=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Class"])


print("Ejercicio 3.1")
# Mostrar la descripción de los datos
print("Descripción de los datos:")
print(iris_data.head(20))
# Crear un DataFrame de pandas
iris_data = pd.DataFrame(data=np.c_[X, y], columns=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Class"])


print("Ejercicio 3.1")
# Mostrar la descripción de los datos
print("Descripción de los datos:")
print(iris_data.head(20))


print("Ejercicio 3.2")
# Imprimir las llaves y el número de filas y columnas
print("\nLlaves:")
print(iris_data.keys())
print("\nNúmero de filas y columnas:")
print(iris_data.shape)


print("Ejercicio 3.3")
# Obtener el número de muestras faltantes o NaN
print("\nNúmero de muestras faltantes:")
print(iris_data.isnull().sum())


print("Ejercicio 3.4")
# Paso 3.4: Crear un arreglo 2-D de tamaño 5x5 con unos en la diagonal y ceros en el resto
array_2d = np.eye(5)
# Convertir el arreglo a una matriz dispersa de Scipy en formato CRS
sparse_matrix = csr_matrix(array_2d)
# Mostrar la matriz dispersa
print("\nMatriz dispersa:")
print(sparse_matrix)

# Paso 3.5: Mostrar estadísticas básicas utilizando describe
print("\nEjercicio 3.5")
print(iris_df.describe().loc[['mean', 'std']])


print("Ejercicio 3.6")
# Paso 3.6: Obtener el número de muestras para cada clase
samples_per_class = iris_data['Class'].value_counts()
print("\nNúmero de muestras por clase:")
print(samples_per_class)


print("Ejercicio 3.7")
# Paso 3.7: Añadir un encabezado a los datos usando los nombres en iris.names
header = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Class']
iris_data.columns = header
print(iris_data)


print("Ejercicio 3.8")
# Paso 3.8: Imprimir las diez primeras filas y las dos primeras columnas del data frame
print("\nLas diez primeras filas y las dos primeras columnas:")
print(iris_data.iloc[:10, :2])