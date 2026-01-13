from skimage.filters import median
from skimage.morphology import ball 

from image_utils import load_image, edge_detection




image = load_image("my_cat.jpeg")



clean_image = median(image, ball(3))
edge_img = edge_detection(clean_image)



edge_binary = edge_img > 100

plt.imshow('my_edges-2.png', edge_binary, cmap='gray')
