import cv2
import numpy as np

def enhance_smoky_image(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive histogram equalization to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

    # Apply morphological operations (closing) to further reduce noise
    kernel = np.ones((5,5),np.uint8)
    closed = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)

    return closed

# Example usage:
image_path ='smoky image2.jpg'
enhanced_image = enhance_smoky_image(image_path)

# Display the original and enhanced images
cv2.imshow('Original Image', cv2.imread(image_path))
cv2.imshow('Enhanced Image', enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()