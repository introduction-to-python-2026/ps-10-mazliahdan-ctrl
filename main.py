from skimage.filters import median
from skimage.morphology import ball
from image_utils import load_image, edge_detection
import numpy as np


image = load_image('.tests/lena.jpg')


image = median(image, ball(3))


edge = edge_detection(image)


edge_binary = edge > 50


true = load_image('.tests/lena_edges.png')
if true.ndim == 3:
    true = np.mean(true, axis=2)  


area = true.shape[0] * true.shape[1]
score = np.sum(true == edge_binary) / area
assert score > 0.9
