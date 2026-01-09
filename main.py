import matplotlib.pyplot as plt
from image_utils import load_image, edge_detection
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
image_path = "/content/my_cat.jpeg"
img_array = load_image(image_path)
clean_image = median(img_array, ball(3))
edge_magnitudes = edge_detection(clean_image)
plt.figure(figsize=(5,3))
plt.hist(edge_magnitudes.ravel(), bins=255);
plt.imshow(edge_magnitudes, cmap='gray')

plt.show()
threshold = 100
edge_binary = edge_magnitudes > threshold
edge_binary = edge_binary.astype(np.uint8) * 255
edge_image = Image.fromarray(edge_binary)
edge_image.save('my_edges.png')
