import numpy as np
from matplotlib import pyplot as plt

import cv2

h = 100
w = 100

center = (int(w/2), int(h/2))
radius = min(center[0], center[1], w-center[0], h-center[1]) // 3

Y, X = np.ogrid[:h, :w]
dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

mask = dist_from_center <= radius

plt.figure(), plt.imshow(mask, cmap='gray')


img_circle = np.ones((h, w), dtype = "uint8") 
cv2.circle(img_circle, center, radius, 255, -1)
img_circle[img_circle != 255] = 0

plt.figure(), plt.imshow(img_circle, cmap='gray')

kernel = np.ones((3,3), np.uint8)
kernel[0, 0] = 0
kernel[2, 2] = 0
kernel[2, 0] = 0
kernel[0, 2] = 0
print(kernel)