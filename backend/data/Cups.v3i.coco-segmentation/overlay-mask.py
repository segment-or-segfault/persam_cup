import cv2
import numpy as np

# Load the original image and mask
image_path = "/Users/tangxiaohan/Desktop/2025 school/csc490/Cups.v3i.coco-segmentation/test/Images/cup-186-_jpg.rf.815ba93b39bff98a48e090ceaab7f006.jpg"
mask_path = "/Users/tangxiaohan/Desktop/2025 school/csc490/Cups.v3i.coco-segmentation/test/Annotations/cup-186-_jpg.rf.815ba93b39bff98a48e090ceaab7f006.png"
image = cv2.imread(image_path)  # your original image
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # your binary mask (0 or 255)

# Ensure they have the same dimensions
mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

# Choose a color for the mask overlay (e.g., red)
color = np.array([0, 0, 255], dtype=np.uint8)  # BGR format for red

# Create a color version of the mask
mask_color = np.zeros_like(image, dtype=np.uint8)
mask_color[mask > 0] = color

# Overlay with transparency
alpha = 0.5  # transparency level
overlay = cv2.addWeighted(image, 1 - alpha, mask_color, alpha, 0)

# Display the result
cv2.imshow("Overlay", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save it
cv2.imwrite("overlay_result.jpg", overlay)
