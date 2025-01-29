import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
from image_utils import load_image, edge_detection

# 1. טעינת התמונה המקורית
image_path = "path/to/your/image.jpg"  # יש לשנות לנתיב לתמונה שלך
image = load_image(image_path)

# 2. הפחתת רעשים באמצעות פילטר מדיאן
clean_image = median(image, ball(3))

# 3. זיהוי קצוות בתמונה לאחר הפחתת רעשים
edges = edge_detection(clean_image)

# 4. יצירת תמונה בינארית על פי סף
threshold = np.mean(edges)  # חישוב הסף על בסיס ממוצע הפיקסלים
edge_binary = edges > threshold

# 5. הצגת ושמירת התמונה עם הגבולות
plt.figure(figsize=(8, 6))
plt.imshow(edge_binary, cmap='gray')
plt.title("Binary Edge Detection")
plt.axis("off")
plt.show()

# שמירת התוצאה כקובץ PNG
edge_image = Image.fromarray((edge_binary * 255).astype(np.uint8))
edge_image.save("edge_detected.png")

print("✅ Edge detection completed and saved as 'edge_detected.png'.")

