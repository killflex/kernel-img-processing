import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img_rgb = cv.imread('ferry.jpg')  # Ganti dengan path citra Anda
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

# Define two kernels
kernel1 = np.array([[-1, -1, -1], 
                    [-1,  8, -1], 
                    [-1, -1, -1]])  # Edge detection kernel 1

kernel2 = np.array([[ 1,  0, -1], 
                    [ 0,  0,  0], 
                    [-1,  0,  1]])  # Edge detection kernel 2

# Apply convolution using the two kernels
edge1 = cv.filter2D(img_gray, -1, kernel1)
edge2 = cv.filter2D(img_gray, -1, kernel2)

# Calculate histograms
hist_gray = cv.calcHist([img_gray], [0], None, [256], [0, 256])
hist_edge2 = cv.calcHist([edge2], [0], None, [256], [0, 256])

# Plot the results
plt.figure(figsize=(12, 8))

# Display the original RGB image
plt.subplot(231)
plt.imshow(cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB))
plt.title('Citra RGB')
plt.axis('off')

# Display the grayscale image
plt.subplot(232)
plt.imshow(img_gray, cmap='gray')
plt.title('Citra Gray')
plt.axis('off')

# Display the result of edge detection using kernel 1
plt.subplot(233)
plt.imshow(edge1, cmap='gray')
plt.title('Extract Edge kernel 1')
plt.axis('off')

# Display the result of edge detection using kernel 2
plt.subplot(234)
plt.imshow(edge2, cmap='gray')
plt.title('Extract Edge kernel 2')
plt.axis('off')

# Plot histogram of the grayscale image
plt.subplot(235)
plt.plot(hist_gray, color='black')
plt.title('Histogram Citra Gray')
plt.xlim([0, 256])
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

# Plot histogram of the result from kernel 2
plt.subplot(236)
plt.plot(hist_edge2, color='black')
plt.title('Histogram Edge Kernel 2')
plt.xlim([0, 256])
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
