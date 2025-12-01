import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tabulate import tabulate
import csv

# Methods and their prediction folders
methods = {
    "persamf": "backend/outputs/cups/persamf",
    "our_persamf": "backend/outputs/cups/our_persamf",
    "our_persamf_augment_rotate_90": "backend/outputs/cups/our_persamf_augment_rotate_90",
    "our_persamf_augment_rotate_270": "backend/outputs/cups/our_persamf_augment_rotate_270",
}

images_dir = "backend/data/Cups.v3i.coco-segmentation/test/Images"
masks_dir = "backend/data/Cups.v3i.coco-segmentation/test/Annotations"

gt_masks = [f for f in os.listdir(masks_dir) if f.lower().endswith(".png")]

# IoU threshold for “correct” prediction
IOU_THRESH = 0.7

scores       = {m: 0   for m in methods.keys()}  # count of IoU >= IOU_THRESH
ious_all     = {m: []  for m in methods.keys()}  # IoUs over *all* images
ious_passed  = {m: []  for m in methods.keys()}  # IoUs only for IoU >= IOU_THRESH


# Helper: undo rotation for rotated models
def undo_rotation(pred_mask_img, method_name):
    """
    pred_mask_img: numpy array (H, W) from prediction PNG
    method_name: key in `methods`, used to decide how to rotate back
    """
    if "rotate_90" in method_name:
        # model saw 90°-rotated images => prediction is rotated 90° CW
        # to match original GT, rotate 90° CCW
        return cv2.rotate(pred_mask_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif "rotate_270" in method_name:
        # model saw 270°-rotated images => prediction is rotated 270° CW (= 90° CCW)
        # so to match GT, rotate 90° CW
        return cv2.rotate(pred_mask_img, cv2.ROTATE_90_CLOCKWISE)
    else:
        return pred_mask_img


#  FIRST PASS — compute IoUs, no images shown
count_seen = 0

for mask_filename in gt_masks:
    count_seen += 1
    original_mask_path = os.path.join(masks_dir, mask_filename)
    original_mask = np.array(Image.open(original_mask_path).convert("L")) > 0

    for name, directory in methods.items():
        pred_path = os.path.join(directory, mask_filename)
        if not os.path.exists(pred_path):
            continue

        pred = np.array(Image.open(pred_path).convert("L"))

        #fix orientation for rotated models
        pred = undo_rotation(pred, name)

        # resize to GT size
        pred = cv2.resize(
            pred,
            (original_mask.shape[1], original_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        pred_mask = pred > 0

        inter = np.logical_and(pred_mask, original_mask).sum()
        union = np.logical_or(pred_mask, original_mask).sum()
        iou = inter / union if union > 0 else 0.0

        # store IoU for all images
        ious_all[name].append(iou)

        # check if this image is “correct”
        if iou >= IOU_THRESH:
            scores[name] += 1
            ious_passed[name].append(iou)


print("\n\n====== FINAL RESULTS (NO PLOTS YET) ======")
table = []
for name in methods.keys():
    mean_iou_all = np.mean(ious_all[name]) if len(ious_all[name]) > 0 else 0.0
    mean_iou_pass = (
        np.mean(ious_passed[name]) if len(ious_passed[name]) > 0 else 0.0
    )
    accuracy = scores[name] / count_seen if count_seen > 0 else 0.0
    table.append(
        [
            name,
            mean_iou_all,     # average IoU over all images
            accuracy,         # fraction with IoU >= IOU_THRESH
            mean_iou_pass,    # average IoU only over correctly predicted images
            scores[name],     # how many passed
            len(ious_all[name]),
        ]
    )

print(
    tabulate(
        table,
        headers=[
            "Method",
            "Mean IoU (all)",
            f"IoU>={IOU_THRESH} Accuracy",
            f"Mean IoU (IoU>={IOU_THRESH})",
            "Passed Count",
            "Images",
        ],
        floatfmt=".4f",
    )
)

# Save CSV
with open("backend/results_cups.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "Method",
            "Mean IoU (all)",
            "Accuracy",
            f"Mean IoU (IoU>={IOU_THRESH})",
            "Passed",
            "Total",
        ]
    )
    for row in table:
        writer.writerow(row)

print("\nResults saved to results_cups.csv")

# still choose “best” by overall mean IoU (you can change to mean_iou_pass if you want)
best_method = max(table, key=lambda r: r[1])
print(f"\nBest method by Mean IoU (all): {best_method[0]} (mean IoU = {best_method[1]:.3f})")


#  SECOND PASS — SHOW ALL qualitative visualizations
print("\nShowing ALL qualitative results...\n")

for mask_filename in gt_masks:
    original_mask_path = os.path.join(masks_dir, mask_filename)
    original_mask = np.array(Image.open(original_mask_path).convert("L")) > 0

    # corresponding RGB image
    jpg_name = mask_filename.replace(".png", ".jpg")
    image_path = os.path.join(images_dir, jpg_name)
    if os.path.exists(image_path):
        img = np.array(Image.open(image_path).convert("RGB"))
    else:
        img = np.zeros((*original_mask.shape, 3), dtype=np.uint8)

    # only keep methods that actually have this file
    available_methods = [
        (name, directory)
        for name, directory in methods.items()
        if os.path.exists(os.path.join(directory, mask_filename))
    ]
    if not available_methods:
        continue

    plt.figure(figsize=(18, 6))
    plt.suptitle(mask_filename)

    # show original
    plt.subplot(1, len(available_methods) + 1, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    col = 2
    for name, directory in available_methods:
        pred_path = os.path.join(directory, mask_filename)
        pred = np.array(Image.open(pred_path).convert("L"))

        # 🔁 fix orientation for this method
        pred = undo_rotation(pred, name)

        pred = cv2.resize(
            pred,
            (original_mask.shape[1], original_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        pred_mask = pred > 0

        inter = np.logical_and(pred_mask, original_mask).sum()
        union = np.logical_or(pred_mask, original_mask).sum()
        iou = inter / union if union > 0 else 0.0

        plt.subplot(1, len(available_methods) + 1, col)
        plt.imshow(pred_mask, cmap="Reds")
        plt.title(f"{name}\nIoU={iou:.3f}")
        plt.axis("off")
        col += 1

    plt.tight_layout()
    plt.show()
