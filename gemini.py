import cv2
import numpy as np
import matplotlib.pyplot as plt

# Membaca citra asli
img = cv2.imread('ferry.jpg', 0)  # Baca sebagai grayscale

# Menampilkan histogram citra asli
plt.hist(img.ravel(), 256, [0, 256])
plt.show()

# Histogram equalization
equ = cv2.equalizeHist(img)

# Menampilkan histogram setelah equalization
plt.hist(equ.ravel(), 256, [0, 256])
plt.show()

# Padding reflect
padded_img = cv2.copyMakeBorder(equ, 10, 10, 10, 10, cv2.BORDER_REFLECT)

# Deteksi tepi Laplacian
laplacian = cv2.Laplacian(padded_img, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

# Deteksi tepi Sobel
sobelx = cv2.Sobel(padded_img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(padded_img, cv2.CV_64F, 0, 1, ksize=5)
sobelx = np.uint8(np.absolute(sobelx))
sobely = np.uint8(np.absolute(sobely))

# Transformasi negatif
inverted_img = 255 - equ

# Menampilkan histogram setelah transformasi negatif
plt.hist(inverted_img.ravel(), 256, [0, 256])
plt.show()

# Menampilkan hasil
cv2.imshow('Original', img)
cv2.imshow('Histogram Equalized', equ)
cv2.imshow('Laplacian', laplacian)
cv2.imshow('Sobel X', sobelx)
cv2.imshow('Sobel Y', sobely)
cv2.imshow('Inverted', inverted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()