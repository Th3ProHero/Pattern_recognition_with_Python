import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import cv2
from skimage import io
from PIL import Image

Imagen = 'perrito.jpg'

# METODO 1 CON MATPLOTLIB
img = mpimg.imread(Imagen)
plt.imshow(img)
plt.title('Matplotlib')
plt.show()
time.sleep(2)
plt.close()

# OPENCV
img = cv2.imread(Imagen)
cv2.imshow('OpenCV - Imagen', img)
cv2.waitKey(2000)  # 2000 milisegundos = 2 segundos
cv2.destroyAllWindows()

# SCIKIT-IMAGE
img = io.imread(Imagen)
io.imshow(img)
plt.title('Scikit-Image')
io.show()
time.sleep(2)
io.show()

# PIL (Pillow)
img = Image.open(Imagen)
img.show()
img.title = 'PIL (Pillow)'
time.sleep(2)
img.close()
