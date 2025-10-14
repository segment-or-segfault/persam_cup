# persam_cup
Persam on cup datasets. 

The Personalize-SAM directory is downloaded from the github repo
https://github.com/ZrrSkywalker/Personalize-SAM. 

All model weights have been put to gitignore due to large file size. Please use the above link and include the following files after you cloned this repo:
- Personalize-SAM-main/weights/mobile_sam/.pt
- sam_vit_h_4b8939.pth

# Dataset preprocessing

Persam requires the dataset to be in the form /Images and /Annotations where /Images contains the original imgaes and /Annotations contains the masks. The test directory under Cups.v3i.coco-segmentation/ have been changed to this from.

If you want to play around with /train and /valid, run the script convert-coco-to-mask.py to convert them into the correct form. Note that you need to manually change the following variables inside the convert-coco-to-mask.py to match the directory.
```
json_path = "test/_annotations.coco.json"
image_dir = "test/Images"
mask_dir = "test/Annotations"
```

# Persam

To run persam on the cup datasets, use (change the ouput directory as you prefer. If you are using cups/persam, remember to create the directory output/cups first.):
```
python Personalize-SAM-main/persam.py --data Cups.v3i.coco-segmentation/test --outdir cups/persam --single true
```

# Persam-f

To run persam_f on the cup datasets, use:
```
python Personalize-SAM-main/persam_f.py --data Cups.v3i.coco-segmentation/test --outdir cups/persam --single true
```

# Compare results

To compare results from persam and persam_f, use compare-cup-reuslts.py