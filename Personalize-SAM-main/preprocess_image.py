import cv2
import numpy as np
import matplotlib.pyplot as plt

def single_scale_retinex(img, sigma=80):
    """Retinex illumination normalization (applied per channel)."""
    result = np.zeros_like(img)
    for c in range(3):
        blur = cv2.GaussianBlur(img[..., c], (0, 0), sigma)
        retinex = np.log10(img[..., c] + 1.0) - np.log10(blur + 1.0)
        retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min() + 1e-8)
        result[..., c] = retinex
    return result

def gray_equalized(img_rgb):
    """Convert RGB float image (0-1) → grayscale uint8 → equalize histogram."""
    res = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    # res = cv2.bilateralFilter(res, 9, 50, 50)
    gray_eq = cv2.equalizeHist(res)
    # normalize back to 0-1 for visualization
    res = gray_eq.astype(np.float32)
    return res

# ---- load + apply ----
image = cv2.imread('/Users/tangxiaohan/Desktop/2025 school/csc490/Cups.v3i.coco-segmentation/test/Images/cup-4-_jpg.rf.6c9466c1e5948f23e40a9d006533fab2.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.astype(np.float32) / 255.0

# Option 1: Retinex (illumination/ reflection suppression)
image_retinex = single_scale_retinex(image, sigma=80)

# Option 2: Convert to gray + equalize (texture/color suppression)
image_gray = gray_equalized(image)

both = gray_equalized(image_retinex)

# ---- visualize ----
plt.figure(figsize=(10,5))
plt.subplot(1,4,1); plt.imshow(image); plt.title("Original"); plt.axis('off')
plt.subplot(1,4,2); plt.imshow(image_retinex); plt.title("Retinex"); plt.axis('off')
plt.subplot(1,4,3); plt.imshow(image_gray, cmap='gray'); plt.title("gray"); plt.axis('off')
plt.subplot(1,4,4); plt.imshow(both, cmap='gray'); plt.title("Both"); plt.axis('off')
plt.show()
