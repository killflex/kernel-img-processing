import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

# Step 1: Load image
img_rgb = cv.imread('ferry.jpg')  # Replace with your image path

# Step 2: Convert RGB to Grayscale
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

# Step 3: Apply Padding (border reflect) with 10% of the image size
padded_img = cv.copyMakeBorder(img_gray, 44, 44, 39, 39, cv.BORDER_REFLECT)

# Step 4: Histogram Equalization (applied after grayscale conversion)
equalized_img = cv.equalizeHist(padded_img)

# Step 5: Apply edge detection with different kernels
kernel1 = np.array([[-1, -1, -1],
                    [-1,  8, -1],
                    [-1, -1, -1]])
kernel2 = np.array([[1, 0, -1],
                    [1, 0, -1],
                    [1, 0, -1]])

# Edge detection using Laplacian kernel
edge1 = cv.filter2D(equalized_img, -1, kernel1)

# Edge detection using Sobel kernel
edge2 = cv.filter2D(equalized_img, -1, kernel2)

# Step 6: Plot in a 3x3 Grid Layout
fig, axs = plt.subplots(3, 3, figsize=(12, 12))

# Top row: Original image, Grayscale, and Padded image
axs[0, 0].imshow(cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB))
axs[0, 0].set_title('Citra RGB')
axs[0, 0].axis('off')

axs[0, 1].imshow(img_gray, cmap='gray')
axs[0, 1].set_title('Citra Gray')
axs[0, 1].axis('off')

axs[0, 2].imshow(padded_img, cmap='gray')
axs[0, 2].set_title('Padded Image (Border Reflect)')
axs[0, 2].axis('off')

# Middle row: Equalized image, Edge detection with kernel 1, Edge detection with kernel 2
axs[1, 0].imshow(equalized_img, cmap='gray')
axs[1, 0].set_title('Equalized Image')
axs[1, 0].axis('off')

axs[1, 1].imshow(edge1, cmap='gray')
axs[1, 1].set_title('Extract Edge Kernel 1')
axs[1, 1].axis('off')

axs[1, 2].imshow(edge2, cmap='gray')
axs[1, 2].set_title('Extract Edge Kernel 2')
axs[1, 2].axis('off')

# Bottom row: Histograms for grayscale image, equalized image, and edge 2 result
axs[2, 0].hist(img_gray.ravel(), bins=256, range=[0, 256])
axs[2, 0].set_title('Histogram Citra Gray')

axs[2, 1].hist(equalized_img.ravel(), bins=256, range=[0, 256])
axs[2, 1].set_title('Histogram Equalized Image')

axs[2, 2].hist(edge2.ravel(), bins=256, range=[0, 256])
axs[2, 2].set_title('Histogram Edge Kernel 2')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
