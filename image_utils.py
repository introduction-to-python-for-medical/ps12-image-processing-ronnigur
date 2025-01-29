from PIL import Image
import numpy as np
from scipy.signal import convolve2d
from skimage.filters import median
from skimage.morphology import disk

def load_image(path):
    img = Image.open(path)
    img = img.convert("L")  # Convert to grayscale
    img = np.array(img)
    return img

def edge_detection(img):
    # Sobel-like kernels for edge detection
    horizontal_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    vertical_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    
    # Apply median filter to reduce noise
    clear_image = median(img, disk(1))
    
    # Convolve with horizontal and vertical kernels
    edge_x = convolve2d(clear_image, horizontal_kernel, mode='same', boundary='fill', fillvalue=0)
    edge_y = convolve2d(clear_image, vertical_kernel, mode='same', boundary='fill', fillvalue=0)
    
    # Compute the gradient magnitude
    edge_mag = np.sqrt(edge_x**2 + edge_y**2)
    
    # Dynamic threshold using mean or Otsu method (can be adjusted)
    threshold = np.mean(edge_mag) * 1.5  # Can experiment with the multiplier
    
    # Apply the threshold to create a binary image
    edge_mag_binary = (edge_mag > threshold).astype(np.uint8) * 255  # Scale to 0-255
    
    return edge_mag_binary



