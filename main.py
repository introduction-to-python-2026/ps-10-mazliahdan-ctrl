import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from skimage.filters import median
from skimage.morphology import ball

from image_utils import load_image, edge_detection



image_path = "/content/my_cat.jpeg"
image = load_image(image_path)


clean_image = median(image, ball(3))


edgeMAG = edge_detection(clean_image)


plt.figure()
plt.hist(edgeMAG.ravel(), bins=256)
plt.title("Edge magnitude histogram")
plt.show()

threshold = 50
edge_binary = edgeMAG > threshold   # boolean array

plt.figure()
plt.imshow(edge_binary, cmap="gray")
plt.axis("off")
plt.show()

edge_binary_uint8 = (edge_binary.astype(np.uint8)) * 255
edge_image = Image.fromarray(edge_binary_uint8)
edge_image.save("my_edges.png")
