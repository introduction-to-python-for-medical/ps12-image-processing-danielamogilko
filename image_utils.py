from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    img = Image.open(path)  # Open the image
    img_array = np.array(img)    # Convert image to a NumPy array
    return img_array

def edge_detection(image):
    # 1. המרת התמונה לאפור (גרייסקייל) ע"י ממוצע של 3 הערוצים (RGB)
    grayscale = np.mean(image, axis=2)

    # 2. יצירת מסננים לזיהוי גבולות
    kernelY = np.array([[ 1,  2,  1],
                         [ 0,  0,  0],
                         [-1, -2, -1]])  # Vertical edges

    kernelX = np.array([[-1,  0,  1],
                         [-2,  0,  2],
                         [-1,  0,  1]])  # Horizontal edges

    # 3. קונבולוציה של המסננים על התמונה האפורה
    edgeX = convolve(grayscale, kernelX, mode='constant', cval=0.0)  # Horizontal edges
    edgeY = convolve(grayscale, kernelY, mode='constant', cval=0.0)  # Vertical edges

    # 4. חישוב גודל הגבול
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    return edgeMAG  # מחזירים את התמונה עם הגבולות
