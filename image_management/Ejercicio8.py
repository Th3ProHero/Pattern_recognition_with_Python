import numpy as np
import cv2
import matplotlib.pyplot as plt

# Leer el video
cap = cv2.VideoCapture('0X2A8498756C4D6E82_corto.avi')

# Verificar si el video se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir el archivo de video.")
    exit()

# Leer y mostrar cada frame
while True:
    ret, frame = cap.read()

    # Verificar si se ha llegado al final del video
    if not ret:
        print("Fin del video.")
        break

    # Mostrar el frame
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.pause(0.03)  # Ajusta el tiempo de pausa según tu preferencia
    plt.draw()

# Liberar el objeto de captura de video
cap.release()

# Mantener la ventana abierta hasta que se cierre manualmente
plt.show()
