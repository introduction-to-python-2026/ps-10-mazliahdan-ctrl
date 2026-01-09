from PIL import Image
import numpy as np
from scipy.ndimage import convolve



def load_image(file_path):
    my_cat = "my_cat.jpeg"
    cat = Image.open(my_cat).convert("RGB")
    image_array = np.array(cat)
    return image_array

image_path = "/content/my_cat.jpeg"
img_array = load_image(image_path)

print(type(img_array))
print(img_array.shape)
print(img_array.dtype)



def edge_detection(image_array):
    
    grayscale_image = np.mean(image_array, axis=2)

    
    kY = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    
    kX = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    
    edgeY = convolve(grayscale_image, kY, mode='constant', cval=0)
    edgeX = convolve(grayscale_image, kX, mode='constant', cval=0)

   
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    return edgeMAG


edge_magnitudes = edge_detection(img_array)

print(f"Shape of edgeMAG: {edge_magnitudes.shape}")
print(f"Data type of edgeMAG: {edge_magnitudes.dtype}")
print(f"Min value of edgeMAG: {np.min(edge_magnitudes)}")
print(f"Max value of edgeMAG: {np.max(edge_magnitudes)}")
