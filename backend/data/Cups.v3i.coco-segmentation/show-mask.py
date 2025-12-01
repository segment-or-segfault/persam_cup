import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

img_path = "test/images/cup-4-_jpg.rf.6c9466c1e5948f23e40a9d006533fab2.jpg"
mask_path = "test/masks/cup-4-_jpg.rf.6c9466c1e5948f23e40a9d006533fab2_mask.png"

img = np.array(Image.open(img_path))
mask = np.array(Image.open(mask_path))

plt.figure(figsize=(6,6))
plt.imshow(img)
plt.imshow(mask, cmap="jet", alpha=0.5)  # colored overlay
plt.axis("off")
plt.show()
