from google.colab import files
uploaded = files.upload()
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from PIL import Image

# Step 1: Load the image
image_path = "one.png"
print(f"Loading image from: {image_path}")

try:
    image = Image.open(image_path).convert('L')  # Convert to grayscale
except FileNotFoundError:
    print(f"Error: The file at {image_path} was not found.")
    exit()

# Step 2: Convert image to numpy array
image_array = np.array(image)

# Step 3: Apply Gaussian filter
smoothed_image = gaussian_filter(image_array, sigma=2)

# Step 4: Display both images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_array, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Smoothed Image")
plt.imshow(smoothed_image, cmap='gray')
plt.axis('off')

plt.show()
