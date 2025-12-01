import json
import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import cv2

# --- paths ---
json_path = "valid/_annotations.coco.json"
image_dir = "valid/Images"
mask_dir = "valid/Annotations"

import shutil
import os

directory_to_remove = mask_dir

# Check if the directory exists before attempting to remove it
if os.path.isdir(directory_to_remove):
    try:
        shutil.rmtree(directory_to_remove)
        print(f"Directory '{directory_to_remove}' and its contents removed successfully.")
    except OSError as e:
        print(f"Error: {e.filename} - {e.strerror}.")
else:
    print(f"Directory '{directory_to_remove}' does not exist.")

    
os.makedirs(mask_dir, exist_ok=True)

# --- load COCO annotations ---
coco = COCO(json_path)

for img_id in coco.getImgIds():
    img_info = coco.loadImgs(img_id)[0]
    img_name = img_info["file_name"]
    width, height = img_info["width"], img_info["height"]

    # empty mask (0 = background)
    mask = np.zeros((height, width), dtype=np.uint8)

    # get annotations for this image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    for ann in anns:
        if isinstance(ann["segmentation"], list):
            rle = maskUtils.frPyObjects(ann["segmentation"], height, width)
            rle = maskUtils.merge(rle)
        else:
            rle = ann["segmentation"]

        m = maskUtils.decode(rle)
        mask[m > 0] = 255

    # save binary mask (white = object, black = background)
    mask_filename = os.path.splitext(img_name)[0] + ".png"
    mask_path = os.path.join(mask_dir, mask_filename)
    cv2.imwrite(mask_path, mask)

    print(f"✅ Saved binary mask for {img_name} at {mask_path}")
