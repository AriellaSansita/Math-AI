import streamlit as st

uploaded_file = st.file_uploader("Upload a file")
if uploaded_file is not None:
    # process the file
    content = uploaded_file.read()

from google.colab import files
uploaded = files.upload()
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from PIL import Image

image_path = "one.png"
print(f"Loading image from: {image_path}")

try:
    image = Image.open(image_path).convert('L') 
except FileNotFoundError:
    print(f"Error: The file at {image_path} was not found.")
    exit()

image_array = np.array(image)

smoothed_image = gaussian_filter(image_array, sigma=2)

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
