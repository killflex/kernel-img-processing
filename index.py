import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load original image (replace 'image.jpg' with your image path)
image = cv2.imread('ferry.jpg')

# Convert BGR to RGB for displaying
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. Convert image to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Function to plot histogram
def plot_histogram(image, title):
    plt.hist(image.ravel(), bins=256, range=[0, 256])
    plt.title(title)
    plt.show()

# 4. Apply histogram equalization for image enhancement
image_equalized = cv2.equalizeHist(image_gray)

# 6. Apply reflect padding
image_padded = cv2.copyMakeBorder(image_rgb, 50, 50, 50, 50, cv2.BORDER_REFLECT)

# 7. Edge detection using Laplacian
laplacian_edges = cv2.Laplacian(image_gray, cv2.CV_64F)

# 8. Edge detection using Sobel
sobelx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)  # x direction
sobely = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)  # y direction
sobel_edges = cv2.magnitude(sobelx, sobely)

# 9. Negative or Invert Transformation (without using library)
def invert_image(image):
    return 255 - image

image_inverted = invert_image(image_gray)

# Create the grid layout with 5x2 subplots
fig, axes = plt.subplots(5, 2, figsize=(15, 20))

# 1. Original image
axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis('off')

# 2. Grayscale image
axes[0, 1].imshow(image_gray, cmap='gray')
axes[0, 1].set_title("Grayscale Image")
axes[0, 1].axis('off')

# 3. Histogram of grayscale image
axes[1, 0].hist(image_gray.ravel(), bins=256, range=[0, 256])
axes[1, 0].set_title("Histogram of Grayscale Image")

# 4. Equalized image
axes[1, 1].imshow(image_equalized, cmap='gray')
axes[1, 1].set_title("Equalized Image")
axes[1, 1].axis('off')

# 5. Histogram of equalized image
axes[2, 0].hist(image_equalized.ravel(), bins=256, range=[0, 256])
axes[2, 0].set_title("Histogram of Equalized Image")

# 6. Padded image (reflect)
axes[2, 1].imshow(image_padded)
axes[2, 1].set_title("Padded Image (Reflect)")
axes[2, 1].axis('off')

# 7. Laplacian edge detection
axes[3, 0].imshow(laplacian_edges, cmap='gray')
axes[3, 0].set_title("Laplacian Edge Detection")
axes[3, 0].axis('off')

# 8. Sobel edge detection
axes[3, 1].imshow(sobel_edges, cmap='gray')
axes[3, 1].set_title("Sobel Edge Detection")
axes[3, 1].axis('off')

# 9. Inverted image
axes[4, 0].imshow(image_inverted, cmap='gray')
axes[4, 0].set_title("Inverted Image")
axes[4, 0].axis('off')

# 10. Histogram of inverted image
axes[4, 1].hist(image_inverted.ravel(), bins=256, range=[0, 256])
axes[4, 1].set_title("Histogram of Inverted Image")

# Adjust layout and display
plt.tight_layout()
plt.show()
