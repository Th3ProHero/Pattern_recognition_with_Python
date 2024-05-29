from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

#Path to the original TIFF image
image_path = 'peppers_color.tif'

##Load the image using skimage, which can handle various image formats including TIFF
image_skimage = io.imread(image_path)

#Assuming the image has an unusual shape (e.g., (512, 512, 2)),
#and using the first channel as a grayscale representation
gray_image_from_first_channel = image_skimage[:, :, 0]

#Crop coordinates (x1, y1, x2, y2) - adjust these based on the desired area
#These placeholder values may need to be adjusted to accurately capture one of the green peppers
crop_coordinates = (180, 220, 420, 470)

#Crop the image using the selected coordinates
cropped_image = gray_image_from_first_channel[crop_coordinates[1]:crop_coordinates[3], crop_coordinates[0]:crop_coordinates[2]]

#Convert the cropped image to PIL Image format for saving
cropped_image_pil = Image.fromarray(cropped_image)

#Specify the path where the cropped image will be saved in JPEG format
cropped_image_path = 'cropped_pepper.jpg'

#Save the cropped image in JPEG format
cropped_image_pil.save(cropped_image_path, 'JPEG')

print(f'Cropped image saved as {cropped_image_path}')#