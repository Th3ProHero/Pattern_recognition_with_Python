import tensorflow as tf
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
from tensorflow.python import pywrap_tensorflow as _pywrap_tensorflow
import scipy  # Asegúrate de importar scipy

# Definir el directorio del dataset
train_dir = 'dataset/train'
test_dir = 'dataset/test'

# Verificar que los directorios existan
if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    raise ValueError("Los directorios especificados no existen. Verifica las rutas.")

# Definir parámetros para el generador de datos
img_height, img_width = 128, 128
batch_size = 32

# Crear el generador de datos para el entrenamiento con aumento de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Crear el generador de datos para el test (sin aumento de datos)
test_datagen = ImageDataGenerator(rescale=1./255)

# Cargar las imágenes del directorio de entrenamiento
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Cargar las imágenes del directorio de test
validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Definir la arquitectura del modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Verificar que el modelo se ha creado correctamente
print("debugg model")
print(model)

# Compilar el modelo
model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

# Resumen del modelo
model.summary()

# Entrenar el modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,  # Reducimos el número de épocas para demostración
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Evaluar el modelo con el conjunto de test
test_loss, test_acc = model.evaluate(validation_generator)
print(f'\nTest accuracy: {test_acc:.4f}')

# Guardar el modelo entrenado
model.save('modelo_perros_gatos.h5')
