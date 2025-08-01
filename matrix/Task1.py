import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from PIL import Image

st.title("Gaussian Smoothing of Grayscale Image")

# Upload image
uploaded_file = st.file_uploader("Upload a grayscale image (PNG or JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Convert to grayscale
    try:
        image = Image.open(uploaded_file).convert('L')
        st.success("Image uploaded successfully!")
    except Exception as e:
        st.error(f"Failed to load image: {e}")
        st.stop()

    # Convert image to array
    image_array = np.array(image)

    # Apply Gaussian filter
    smoothed_image = gaussian_filter(image_array, sigma=2)

    # Show both images
    st.subheader("Original vs Smoothed Image")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(image_array, cmap='gray')
    ax[0].set_title("Original")
    ax[0].axis('off')

    ax[1].imshow(smoothed_image, cmap='gray')
    ax[1].set_title("Smoothed")
    ax[1].axis('off')

    st.pyplot(fig)
else:
    st.info("Please upload an image file to begin.")
