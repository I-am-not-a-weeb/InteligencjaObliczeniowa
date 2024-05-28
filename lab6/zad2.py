import cv2
import numpy as np

# Load the image
image = cv2.imread('parrot.png')

weights2 = [0.114, 0.587, 0.299]

# Split the image into its RGB channels
B, G, R = cv2.split(image)

weights = [(R + G + B)/3]
weighted_channels2 = [
    np.multiply(R, weights2[0]),
    np.multiply(G, weights2[1]),
    np.multiply(B, weights2[2])
]

# Sum up the weighted channels to get the grayscale image
gray_image1 = sum(weights)
gray_image2 = sum(weighted_channels2)

# Convert to uint8 (8-bit) data type
gray_image1 = np.uint8(gray_image1)
gray_image2 = np.uint8(gray_image2)

# Save the grayscale image to a file
cv2.imwrite('wrong_wages.jpg', gray_image1)
cv2.imwrite('improved_wages.jpg', gray_image2)

print("Grayscale image saved successfully.")