import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar los datos de Iris
#iris_df = pd.read_csv('iris.data', header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
iris_df = pd.read_csv('EjerciciosIris/iris.data', header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

# 3.9 Crear una gráfica de barras para la media, mínimo y máximo de todos los datos
iris_describe = iris_df.describe()
iris_describe.loc[['mean', 'min', 'max']].plot(kind='bar', figsize=(10, 6))
plt.title('Media, Mínimo y Máximo de características del Iris')
plt.ylabel('Valores')
plt.xlabel('Estadísticas')
plt.show()

# 3.10 Mostrar la frecuencia de las tres especies como una gráfica de pastel
iris_df['species'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(8, 8))
plt.title('Distribución de especies en el dataset Iris')
plt.ylabel('')
plt.show()

# 3.11 Gráfica de la relación entre la longitud y ancho del sépalo para las tres especies
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=iris_df)
plt.title('Relación entre longitud y ancho del sépalo por especie')
plt.show()

# 3.12 Histogramas de las variables SepalLength, SepalWidth, PetalLength y PetalWidth
iris_df.drop('species', axis=1).hist(bins=15, figsize=(10, 7))
plt.suptitle('Histogramas de características del Iris')
plt.show()

# 3.13 Gráficas de dispersión con pairplot de seaborn
sns.pairplot(iris_df, hue='species')
plt.show()

# 3.14 Gráfica de dispersión entre longitud y ancho del sépalo con jointplot
sns.jointplot(x='sepal_length', y='sepal_width', data=iris_df, kind='scatter', color='green')
plt.suptitle('Dispersión entre longitud y ancho del sépalo')
plt.show()

# 3.15 Repetir el ejercicio anterior usando kind="hex"
sns.jointplot(x='sepal_length', y='sepal_width', data=iris_df, kind='hex', color='green')
plt.suptitle('Hexbin de longitud y ancho del sépalo')
plt.show()