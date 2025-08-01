import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from PIL import Image
import io

st.title("Gaussian Smoothing of Uploaded Image")

# Upload image
uploaded_file = st.file_uploader("Upload a grayscale image (e.g., PNG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open image
    try:
        image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()

    image_array = np.array(image)

    # Apply Gaussian filter
    smoothed_image = gaussian_filter(image_array, sigma=2)

    # Display images side by side
    st.subheader("Original vs Smoothed Image")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(image_array, cmap='gray')
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(smoothed_image, cmap='gray')
    ax[1].set_title("Smoothed Image")
    ax[1].axis('off')

    st.pyplot(fig)
