from PIL import Image
import numpy as np
from scipy.signal import convolve2d  # שימוש בפונקציה הנכונה

def load_image(path):
    """Load an image and convert it to a NumPy array."""
    img = Image.open(path)  # Open the image
    img_array = np.array(img)  # Convert image to a NumPy array
    return img_array

def edge_detection(image):
    # 1. Convert image to grayscale
    grayscale = np.mean(image, axis=2)

    # 2. Define Sobel filters for edge detection
    kernelY = np.array([[ 1,  2,  1],
                        [ 0,  0,  0],
                        [-1, -2, -1]])  # Vertical edges

    kernelX = np.array([[-1,  0,  1],
                        [-2,  0,  2],
                        [-1,  0,  1]])  # Horizontal edges

    # 3. Apply convolution using convolve2d
    edgeX = convolve2d(grayscale, kernelX, mode='same', boundary='fill', fillvalue=0)  # Horizontal edges
    edgeY = convolve2d(grayscale, kernelY, mode='same', boundary='fill', fillvalue=0)  # Vertical edges

    # 4. Compute the edge magnitude
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    return edgeMAG  # Return the edge-detected image
