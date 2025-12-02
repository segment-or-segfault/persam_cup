import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# paths to the two result directories
dir1 = "backend/outputs/cups/persam"
dir2 = "backend/outputs/cups/persamf"

# where the original image live
images_dir = "backend/data/Cups.v3i.coco-segmentation/test/Images"
masks_dir = "backend/data/Cups.v3i.coco-segmentation/test/Annotations"


# list all mask files in first dir
mask_files = [
    f for f in os.listdir(dir1)
    # only show the masks
    if f.lower().endswith(('.png'))
]

for mask_filename in mask_files:
    original_mask_path = os.path.join(masks_dir, mask_filename)
    mask1_path = os.path.join(dir1, mask_filename)
    mask2_path = os.path.join(dir2, mask_filename)
    prior_path = os.path.join(dir2, "prior_" + mask_filename).replace(".png", ".jpg")
    vis_path = os.path.join(dir2, "vis_mask_" + mask_filename).replace(".png", ".jpg")

    if not os.path.exists(mask2_path) or not os.path.exists(mask1_path):
        print(f"⚠️  Missing {mask_filename} in {dir1} or {dir2}, skipping.")
        continue

    original_m = np.array(Image.open(original_mask_path).convert("L")) > 0
    m1 = np.array(Image.open(mask1_path).convert("L")) > 0
    m2 = np.array(Image.open(mask2_path).convert("L")) > 0
    prior = np.array(Image.open(prior_path))
    vis_mask = np.array(Image.open(vis_path))

    intersection_m1 = np.logical_and(m1, original_m).sum()
    union_m1 = np.logical_or(m1, original_m).sum()
    iou_m1 = intersection_m1 / union_m1 if union_m1 > 0 else 0

    intersection_m2 = np.logical_and(m2, original_m).sum()
    union_m2 = np.logical_or(m2, original_m).sum()
    iou_m2 = intersection_m2 / union_m2 if union_m2 > 0 else 0

    # load both masks
    mask1 = np.array(Image.open(mask1_path).convert("RGB"))
    mask2 = np.array(Image.open(mask2_path).convert("RGB"))

    # load original image for context
    image_path = os.path.join(images_dir, mask_filename.replace(".png", ".jpg"))
    if os.path.exists(image_path):
        img = np.array(Image.open(image_path).convert("RGB"))
    else:
        img = np.zeros_like(mask1)

    intersection = np.logical_and(m1, m2).sum()

    # side-by-side view
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 5, 1)
    plt.imshow(img)
    plt.title("Original"); plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.imshow(mask1)
    plt.title(f"Persam Result with IoU = {iou_m1:.3f}"); plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.imshow(mask2)
    plt.title(f"Persam_f Result with IoU = {iou_m2:.3f}"); plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.imshow(prior)
    plt.title(f"Prior Result with IoU = {iou_m2:.3f}"); plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.imshow(vis_mask)
    plt.title(f"Vis Mask Result with IoU = {iou_m2:.3f}"); plt.axis("off")

    plt.suptitle(mask_filename, fontsize=14)
    plt.tight_layout()
    plt.show()
