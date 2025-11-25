from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import os
import cv2
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from show import *
from per_segment_anything import sam_model_registry, SamPredictor
from per_segment_anything.adaptive_fusion import AdaptiveMaskFusion


device = "cuda" if torch.cuda.is_available() else \
         "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading DINOv2...")
dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
dino_model = AutoModel.from_pretrained("facebook/dinov2-large").to(device)
dino_model.eval()

@torch.no_grad()
def extract_dino_features(img_np):
    """
    img_np: RGB uint8 numpy array [H, W, 3]
    returns:
        feat_hwD: [H_feat, W_feat, D]
        feat_flat: [HW_feat, D]
        (H_feat, W_feat)
    """

    img_pil = Image.fromarray(img_np)
    inputs = dino_processor(images=img_pil, return_tensors="pt").to(device)

    out = dino_model(**inputs)
    # remove CLS token: out.last_hidden_state: [1, 1+HW, D]
    feat_flat = out.last_hidden_state[:, 1:, :]        # [1, HW, D]
    feat_flat = F.normalize(feat_flat, dim=-1)         # cosine norm
    feat_flat = feat_flat.squeeze(0)                   # [HW, D]

    N, D = feat_flat.shape
    H_feat = W_feat = int(N**0.5)

    feat_hwD = feat_flat.reshape(H_feat, W_feat, D)

    return feat_hwD, feat_flat, (H_feat, W_feat)

# Save original interpolate
_original_interpolate = F.interpolate

def safe_interpolate(input, size=None, scale_factor=None, mode='bicubic', align_corners=False):
    # If MPS: convert unsupported bicubic → bilinear
    if mode == 'bicubic':
        mode = 'bilinear'
    # Call PyTorch's real interpolate, not the monkey-patched one
    return _original_interpolate(input, size=size, scale_factor=scale_factor,
                                 mode=mode, align_corners=align_corners)

# Monkey patch
F.interpolate = safe_interpolate


def get_arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--outdir', type=str, default='persam_f')
    parser.add_argument('--ckpt', type=str, default='./sam_vit_h_4b8939.pth')
    parser.add_argument('--sam_type', type=str, default='vit_h')
    parser.add_argument('--single', type=str, default='false')
    
    parser.add_argument('--lr', type=float, default=1e-3) 
    parser.add_argument('--train_epoch', type=int, default=1000)
    parser.add_argument('--log_epoch', type=int, default=200)
    parser.add_argument('--ref_idx', type=str, default='00')
    
    args = parser.parse_args()
    return args


def main():

    args = get_arguments()
    print("Args:", args)

    images_path = args.data + '/Images/'
    masks_path = args.data + '/Annotations/'
    output_path = './outputs/' + args.outdir

    if not os.path.exists('./outputs/'):
        os.mkdir('./outputs/')
    
    if args.single == "false":
        for obj_name in os.listdir(images_path):
            if ".DS" not in obj_name:
                referenceImageName = 00
                persam_f(args, obj_name, images_path, masks_path, referenceImageName, output_path)
    else:
        files = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not files:
            print("⚠️ No images found!")
        else:
            referenceImage = os.path.join(images_path, files[0])
            base = os.path.basename(referenceImage) 
            referenceImageName = os.path.splitext(base)[0] 
            print("Reference image:", referenceImageName)
            persam_f(args, None, images_path, masks_path, referenceImageName, output_path)


def persam_f(args, obj_name, images_path, masks_path, referenceImageName, output_path):
    if obj_name is None:
        print("\n------------> Segment " + "Cups")
    else:
        print("\n------------> Segment " + obj_name)

    global device
    
    # Path preparation
    ref_image_path = os.path.join(images_path, obj_name or '', referenceImageName + '.jpg')
    print("ref_image_path",ref_image_path)
    ref_mask_path = os.path.join(masks_path, obj_name or '', referenceImageName + '.png')
    test_images_path = os.path.join(images_path, obj_name or '')

    output_path = os.path.join(output_path, obj_name or '')
    os.makedirs(output_path, exist_ok=True)

    # Load images and masks
    ref_image = cv2.imread(ref_image_path)
    if ref_image is None:
        raise FileNotFoundError(f"Could not read reference image: {ref_image_path}. Verify --data, object name, index, and extension.")
    # Convert to grayscale
    preprocessed_image = cv2.GaussianBlur(ref_image, (5, 5), sigmaX=0)

    # Equalize histogram to enhance contrast / normalize lighting
    # gray_eq = cv2.equalizeHist(gray)
    # preprocessed_image = np.stack([gray]*3, axis=-1)
    ori_ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

    ref_mask = cv2.imread(ref_mask_path)
    if ref_mask is None:
        raise FileNotFoundError(f"Could not read reference mask: {ref_mask_path}. Verify the mask exists and path is correct.")
    ori_ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)

    gt_mask = torch.tensor(ref_mask)[:, :, 0] > 0 
    gt_mask = gt_mask.float().unsqueeze(0).flatten(1).to(device)

    
    print("======> Load SAM" )
    if args.sam_type == 'vit_h':
        sam_type, sam_ckpt = 'vit_h', args.ckpt
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device)
    elif args.sam_type == 'vit_t':
        sam_type, sam_ckpt = 'vit_t', 'weights/mobile_sam.pt'
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
        sam.eval()
    
    
    for name, param in sam.named_parameters():
        param.requires_grad = False
    predictor = SamPredictor(sam)
    
    angles = [0]
    sims = []
    target_feats = {}
    for ang in angles:
        # Rotate image and mask
        ref_image_rot = rotate_image(ori_ref_image, ang)
        ref_mask_rot = rotate_image(ori_ref_mask, ang)
        print("======> Obtain Self Location Prior" )
        edge = get_color_outline(ref_image_rot)
        img_for_similarity = combine_edges_with_image(ref_image_rot, edge)
        # DINO
        feat_hwD, feat_flat, (H_feat, W_feat) = extract_dino_features(img_for_similarity)
        # ref_mask = binary mask [H, W]

        # Resize mask to match DINO feature resolution
        ref_mask_gray = ref_mask_rot[:, :, 0] > 0   # shape [H, W]
        ref_mask_small = cv2.resize(ref_mask_gray.astype(np.uint8),
                                    (W_feat, H_feat),
                                    interpolation=cv2.INTER_NEAREST)
        ref_mask_small = torch.tensor(ref_mask_small, device=device)
        # Convert to torch on same device as DINO feat
        ref_mask_small = torch.tensor(ref_mask_small, device=device, dtype=torch.bool)

        # Extract prototype handle feature
        proto_dino = feat_hwD[ref_mask_small > 0].mean(0)    # [D]
        proto_dino = F.normalize(proto_dino, dim=0)


        # SAM Image features encoding
        # side effect: computes and stores the model’s image features (embedding) in predictor.features.
        # return a mask that you can later use to compute target features.
        ref_mask = predictor.set_image(img_for_similarity, ref_mask_rot)
        ref_feat = predictor.features.squeeze().permute(1, 2, 0)

        ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
        ref_mask = ref_mask.squeeze()[0]

        # Target feature extraction describing the handle appearance
        target_feat = ref_feat[ref_mask > 0]
        # Averages all feature vectors inside the masked region
        # Captures the overall, smooth, dominant appearance of the object (stable against noise and small variations)
        target_feat_mean = target_feat.mean(0)
        # Takes the elementwise maximum along each feature channel
        # Highlights the most distinctive or strongest activations among those features (edges, colors, textures that are particularly characteristic)
        target_feat_max = torch.max(target_feat, dim=0)[0]
        # Blends representativeness (mean) with discriminativeness (max)
        target_feat = (target_feat_max / 2 + target_feat_mean / 2).unsqueeze(0)

        # Cosine similarity between target feature and all image features
        h, w, C = ref_feat.shape
        # Normalize features
        target_feat = target_feat / target_feat.norm(dim=-1, keepdim=True)
        ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
        ref_feat = ref_feat.permute(2, 0, 1).reshape(C, h * w)
        # gives a cosine similarity map between the target feature and every pixel feature in the image
        D = target_feat.shape[-1] 
        sim = target_feat @ ref_feat / (D ** 0.5)
        target_feats[ang] = target_feat

    sim = sim.reshape(1, 1, h, w)
    sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
    sim = predictor.model.postprocess_masks(
                    sim,
                    input_size=predictor.input_size,
                    original_size=predictor.original_size).squeeze()

    sim = (sum(sims)) / len(angles)
    # Positive location point on the reference object.
    topk_xy, topk_label = point_selection(sim, topk=1)

    # Save reference location prior as a heatmap overlay on the reference image
    try:
        sim_np = sim.detach().cpu().numpy() if isinstance(sim, torch.Tensor) else np.array(sim)
        prior_vis_ref = os.path.join(output_path, 'location_prior_ref.jpg')
        plt.figure(figsize=(8, 8))
        plt.imshow(ref_image)
        plt.imshow(sim_np, cmap='jet', alpha=0.5)
        # Mark the prompt location with a white x
        prompt_xy = topk_xy[0]  # shape (2,) - [y,x]
        # Use [x,y] for visualization since plt.scatter expects x,y order
        plt.scatter([prompt_xy[0]], [prompt_xy[1]], c='white', marker='x', s=100, linewidths=2)
        plt.title('Location Prior (reference)')
        plt.axis('off')
        plt.savefig(prior_vis_ref, bbox_inches='tight')
        plt.close()
    except Exception:
        # non-fatal: if plotting fails, continue
        pass


    print('======> Start Training')
    # Learnable mask weights
    # mask_weights = Mask_Weights().to(device)
    # mask_weights.train()
    
    # optimizer = torch.optim.AdamW(mask_weights.parameters(), lr=args.lr, eps=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.train_epoch)

    fusion = AdaptiveMaskFusion().to(device)
    fusion.train()

    optimizer = torch.optim.AdamW(fusion.parameters(), lr=args.lr, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.train_epoch)

    # Train combination weights
    for train_idx in range(args.train_epoch):

        # Run the decoder
        # Uses the point we computed to tell SAM where to look
        masks, scores, logits, logits_high = predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            multimask_output=True)
        
        # logits_high = logits_high.flatten(1)

        # Weighted sum three-scale masks
        # weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
        # logits_high = logits_high * weights
        # logits_high = logits_high.sum(0).unsqueeze(0)

        # dice_loss = calculate_dice_loss(logits_high, gt_mask)
        # focal_loss = calculate_sigmoid_focal_loss(logits_high, gt_mask)
        fused_logits = fusion(logits_high).flatten(1)

        # [H,W]
        # fused_logits = fused_logits.unsqueeze(0).flatten(1)  # [1, H*W]

        dice_loss = calculate_dice_loss(fused_logits, gt_mask)
        focal_loss = calculate_sigmoid_focal_loss(fused_logits, gt_mask)
        loss = dice_loss + focal_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()


        if train_idx % args.log_epoch == 0:
            print('Train Epoch: {:} / {:}'.format(train_idx, args.train_epoch))
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Dice_Loss: {:.4f}, Focal_Loss: {:.4f}'.format(current_lr, dice_loss.item(), focal_loss.item()))


    # mask_weights.eval()

    # weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
    # weights_np = weights.detach().cpu().numpy()
    # print('======> Mask weights:\n', weights_np)
    fusion.eval()
    learned_w = fusion.learnable_weights.detach().cpu().numpy()
    print("======> Learned Edge-aware Fusion Weights =", learned_w)

    print('======> Start Testing')
    test_images = os.listdir(test_images_path)
    for test_idx in tqdm(range(len(os.listdir(test_images_path)))):

        # Load test image
        if obj_name is None:
            # single-folder mode: use actual filenames from the directory
            test_image_path = os.path.join(test_images_path, test_images[test_idx])
        else:
            # object folder mode: images are named as 00.jpg, 01.jpg, ...
            test_idx_str = '%02d' % test_idx
            test_image_path = os.path.join(test_images_path, test_idx_str + '.jpg')

        test_sims = []
        for ang in angles:
            ori_test_image = cv2.imread(test_image_path)
            test_image = rotate_image(ori_test_image, ang)
            if test_image is None:
                # Provide extra diagnostics to help debugging missing/corrupt files
                try:
                    exists = os.path.exists(test_image_path)
                    fsize = os.path.getsize(test_image_path) if exists else None
                except Exception:
                    exists = False
                    fsize = None
                print(f"[Warn] Missing or unreadable test image, skipping: {test_image_path} (exists={exists}, size={fsize})")
                continue
            test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

            edge = get_color_outline(test_image)
            img_for_similarity = combine_edges_with_image(test_image, edge)
            # DINO
            test_dino_hwD, _, (hD, wD) = extract_dino_features(img_for_similarity)
            sim_dino = torch.einsum("c,hwc->hw", proto_dino, test_dino_hwD)
            sim_dino_up = F.interpolate(sim_dino.unsqueeze(0).unsqueeze(0),
                            size=predictor.original_size,
                            mode="bilinear")[0,0]


            # SAM Image feature encoding
            predictor.set_image(img_for_similarity)
            test_feat = predictor.features.squeeze()

            # Cosine similarity
            C, h, w = test_feat.shape
            test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
            test_feat = test_feat.reshape(C, h * w)
            # For each test image, it finds the spatial location whose feature vector 
            # best matches the reference prototype
            target_feat = target_feats[ang]
            D = target_feat.shape[-1] 
            sim = target_feat @ test_feat / (D ** 0.5)

            sim = sim.reshape(1, 1, h, w)
            sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
            sim = predictor.model.postprocess_masks(
                            sim,
                            input_size=predictor.input_size,
                            original_size=predictor.original_size).squeeze()
            sim = rotate_sim_back(sim, ang)
            test_sims.append(sim)
        sim_sam = (sum(test_sims)) / len(angles)

        sim_dino_norm = (sim_dino_up - sim_dino_up.min()) / (sim_dino_up.max() - sim_dino_up.min() + 1e-8)
        sim_sam_norm  = (sim_sam      - sim_sam.min())      / (sim_sam.max()      - sim_sam.min() + 1e-8)

        fused_sim = torch.sqrt(sim_dino_norm * sim_sam_norm)

        # Positive location prior
        # gives the prompt for SAM on the test image
        topk_xy, topk_label = point_selection(fused_sim, topk=1)
        # print("topyk_xy before neg:", topk_xy)
        # print("topyk_label before neg:", topk_label)

        # add negative points
        neg_xy, neg_label = negative_point_selection(topk_xy, fused_sim)
        # print("neg_xy:", neg_xy)
        # print("neg_label:", neg_label)

        topk_xy = np.concatenate((topk_xy, neg_xy), axis=0)
        topk_label = np.concatenate((topk_label, neg_label), axis=0)
        # print("topyk_xy after neg:", topk_xy)
        # print("topyk_label after neg:", topk_label)

        # Save test-image location prior as a heatmap overlay
        try:
            vis_test_image = os.path.splitext(os.path.basename(test_image_path))[0]
            sim_np = sim.detach().cpu().numpy() if isinstance(sim, torch.Tensor) else np.array(sim)
            prior_vis_path = os.path.join(output_path, f'prior_{vis_test_image}.jpg')
            plt.figure(figsize=(8, 8))
            plt.imshow(test_image)
            plt.imshow(sim_np, cmap='jet', alpha=1.0)
            show_points(topk_xy, topk_label, plt.gca())
            plt.title(f'Location Prior ({vis_test_image})')
            plt.axis('off')
            plt.savefig(prior_vis_path, bbox_inches='tight')
            plt.close()
        except Exception:
            pass

        # First-step prediction
        masks, scores, logits, logits_high = predictor.predict(
                    point_coords=topk_xy,
                    point_labels=topk_label,
                    multimask_output=True)

        # 1. persam_ cascaded refinement (no fusion)

        # use the highest-score sam  as starting mask
        base_best = int(np.argmax(scores))
        base_mask0 = masks[base_best].astype(np.uint8)      # [H_mask, W_mask]

        # cascade step 1
        y0, x0 = np.nonzero(base_mask0)
        if len(x0) == 0 or len(y0) == 0:
            # nothing there, just skip this image
            continue

        x_min, x_max = x0.min(), x0.max()
        y_min, y_max = y0.min(), y0.max()
        input_box = np.array([x_min, y_min, x_max, y_max])

        masks_ref1, scores_ref1, logits_ref1, _ = predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            box=input_box[None, :],
            mask_input=logits[base_best:base_best+1, :, :],
            multimask_output=True
        )
        best1 = int(np.argmax(scores_ref1))
        base_mask1 = masks_ref1[best1].astype(np.uint8)

        # cascade step 2
        y1, x1 = np.nonzero(base_mask1)
        if len(x1) == 0 or len(y1) == 0:
            continue

        x_min, x_max = x1.min(), x1.max()
        y_min, y_max = y1.min(), y1.max()
        input_box = np.array([x_min, y_min, x_max, y_max])

        masks_ref2, scores_ref2, logits_ref2, _ = predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            box=input_box[None, :],
            mask_input=logits_ref1[best1:best1+1, :, :],
            multimask_output=True
        )
        best2 = int(np.argmax(scores_ref2))
        base_final = masks_ref2[best2].astype(np.uint8)     # [H_mask, W_mask]

        # keep only the component that touches the positive point
        prompt_xy = topk_xy[0]   # (x_img, y_img)
        base_final = keep_component_with_point(base_final, prompt_xy, test_image.shape)

        kernel = np.ones((5, 5), np.uint8)
        base_dilated = cv2.dilate(base_final, kernel, iterations=1)

        # 2. adaptiveMaskFusion + post-processing
        with torch.no_grad():
            # fuse multi-scale logits
            logits_high_t = logits_high.to(device=device, dtype=torch.float32)
            fused = fusion(logits_high_t)                   # [1, H_mask, W_mask]

        # smooth logits to kill spikes
        fused = F.avg_pool2d(
            fused.unsqueeze(0), kernel_size=3, stride=1, padding=1
        ).squeeze(0)                                        # [1, H_mask, W_mask]

        prob = fused.sigmoid()
        fused_low = (prob.squeeze(0) > 0.2).cpu().numpy().astype(np.uint8)

        # morphology cleanup
        fused_low = cv2.morphologyEx(fused_low, cv2.MORPH_CLOSE, kernel, iterations=2)
        fused_low = cv2.dilate(fused_low, kernel, iterations=1)

        # clamp fusion to stay near baseline
        fused_low = np.logical_and(fused_low, base_dilated).astype(np.uint8)

        overlap = np.logical_and(fused_low, base_final).astype(np.uint8)

        if overlap.sum() == 0:
            # fused mask does not touch the baseline handle => reject
            fused_low = np.zeros_like(base_final)
        else:
            # keep only the overlapping region
            fused_low = overlap


        # 3. if fused mask looks bad => fall back to baseline
        fused_area = float(fused_low.sum())
        base_area  = float(base_final.sum())

        iou_fb = mask_iou(fused_low, base_final)

        # if fused is empty OR overlaps too little OR is too big → baseline
        if fused_low.sum() == 0 or iou_fb < 0.5 or fused_low.sum() > 1.3 * base_final.sum():
            final_low = base_final
        else:
            final_low = fused_low

        # upsample whichever one we chose to original image size
        final_mask_up = cv2.resize(
            final_low,
            (test_image.shape[1], test_image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # visual
        plt.figure(figsize=(10, 10))
        plt.imshow(test_image)
        show_mask(final_mask_up, plt.gca())
        show_points(topk_xy, topk_label, plt.gca())
        plt.title("Final Mask (fusion + fallback)", fontsize=18)
        plt.axis('off')

        base = os.path.basename(test_image_path)
        vis_test_image = os.path.splitext(base)[0]
        vis_mask_output_path = os.path.join(output_path, f"vis_mask_{vis_test_image}.jpg")
        plt.savefig(vis_mask_output_path, format='jpg')
        plt.close()

        mask_colors = np.zeros_like(test_image)
        mask_colors[final_mask_up > 0] = [255, 0, 0]
        mask_output_path = os.path.join(output_path, vis_test_image + '.png')
        cv2.imwrite(mask_output_path, mask_colors)



class Mask_Weights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)

def rotate_image(img, angle):
    # angle must be 0, 90, 180, or 270
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if angle == 0:
        return img
    elif angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
def rotate_sim_back(sim, angle):
    # sim is torch tensor H×W
    sim_np = sim.cpu().numpy()

    if angle == 0:
        sim_rot = sim_np
    elif angle == 90:
        sim_rot = cv2.rotate(sim_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        sim_rot = cv2.rotate(sim_np, cv2.ROTATE_180)
    elif angle == 270:
        sim_rot = cv2.rotate(sim_np, cv2.ROTATE_90_CLOCKWISE)
    
    return torch.tensor(sim_rot).to(sim.device)

def get_edge_image(img, blur_ksize=5, canny_lo=50, canny_hi=150, dilate_iter=1):
    """
    Extract a clean edge drawing from an RGB image.
    Returns a 3-channel uint8 edge image suitable for SAM.
    """
    # to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # smooth noise
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1.2)

    # Canny edges
    edges = cv2.Canny(blur, canny_lo, canny_hi)

    # Optional: thicken edges to give SAM stronger cues
    if dilate_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=dilate_iter)

    # Convert to 3-channel (SAM expects RGB input)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return edges_rgb

import cv2
import numpy as np

import cv2
import numpy as np

def get_color_outline(img, k=3, morph=3):
    """
    Extracts the OUTER CONTOUR of the cup based on color clusters.
    Returns a clean outline (3-channel RGB) suitable for SAM.
    """

    # 1. Downsample for speed 
    h, w = img.shape[:2]
    small = cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_AREA)

    # 2. Lab colorspace -> better color separation
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    Z = lab.reshape((-1,3))
    Z = np.float32(Z)

    # 3. K-means segmentation
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)

    segmented = centers[labels.flatten()].reshape(lab.shape).astype(np.uint8)

    # 4. Choose cluster with brightest L-channel (likely the cup)
    L, A, B = cv2.split(segmented)
    cluster_id = np.argmax([np.mean(L[labels.reshape(h//2, w//2)==i]) for i in range(k)])
    cup_mask_small = (labels.reshape(h//2, w//2) == cluster_id).astype(np.uint8)*255

    # 5. Resize mask back to original size
    cup_mask = cv2.resize(cup_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

    # 6. Morphological cleanup
    kernel = np.ones((morph, morph), np.uint8)
    cup_mask = cv2.morphologyEx(cup_mask, cv2.MORPH_CLOSE, kernel)
    cup_mask = cv2.morphologyEx(cup_mask, cv2.MORPH_OPEN, kernel)

    # 7. Find contour
    contours, _ = cv2.findContours(cup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(outline, contours, -1, 255, 2)  # thickness = 2 px

    # 8. Convert to 3 channel
    outline_rgb = cv2.cvtColor(outline, cv2.COLOR_GRAY2BGR)

    return outline_rgb


def combine_edges_with_image(img, edge_img, alpha=0):
    """
    Blend edges with original RGB image.
    Both must be uint8 and same size.
    """
    # ensure uint8
    img = img.astype(np.uint8)
    edge_img = edge_img.astype(np.uint8)

    blended = cv2.addWeighted(edge_img, alpha, img, 1 - alpha, 0)
    cv2.imwrite("blended.jpg", blended)
    return blended


def point_selection(mask_sim, topk=1):
    w, h = mask_sim.shape
    # Get both values and indices of top k
    values, indices = mask_sim.flatten(0).topk(topk)
    # print(f"\nShape of similarity map: {mask_sim.shape}")
    # print(f"Max similarity value: {mask_sim.max().item():.4f}")
    # print(f"Selected values: {values.cpu().numpy()}")
    
    # Find the 2D coordinates of the max value directly
    # max_pos = mask_sim.argmax()
    # max_x = (max_pos // h).item()
    # max_y = (max_pos % h).item()
    # print(f"Direct max position: ({max_x}, {max_y})")
    
    # Original calculation
    topk_xy = indices
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    print(f"Calculated position: ({topk_x.item()}, {topk_y.item()})")
    
    # Get the similarity value at the calculated position
    if torch.is_tensor(mask_sim):
        calc_sim = mask_sim[topk_x.item(), topk_y.item()].item()
    else:
        calc_sim = mask_sim[topk_x.item(), topk_y.item()]
    print(f"Similarity at calculated position: {calc_sim:.4f}")
    
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()
    print("topk_xy:", topk_xy)
    return topk_xy, topk_label

def negative_point_selection(pos_xy, mask_sim, threshold=0.85, step=5000, window=100):
    # unpack positive point (pos_xy is [[px, py]])
    px, py = pos_xy[0]

    # convert to numpy
    mask_np = mask_sim.cpu().numpy()
    sim_min = mask_np.min()
    sim_max = mask_np.max()
    sim_norm = (mask_np - sim_min) / (sim_max - sim_min + 1e-8)

    # find all negative points (before filtering)
    ys, xs = np.where(sim_norm < threshold)

    # --- LOCAL WINDOW FILTER ---
    half = window // 2
    x_low, x_high = px - half, px + half
    y_low, y_high = py - half, py + half

    # boolean mask selecting points inside the window
    in_window = (
        (xs >= x_low) & (xs <= x_high) &
        (ys >= y_low) & (ys <= y_high)
    )

    xs = xs[in_window]
    ys = ys[in_window]

    # fallback if none
    if len(xs) == 0:
        y, x = np.unravel_index(mask_np.argmin(), mask_np.shape)
        return np.array([[x, y]]), np.array([0])

    # stack
    neg_xy = np.stack((xs, ys), axis=-1)

    # subsample (optional)
    neg_xy = neg_xy[::step]

    neg_label = np.zeros(len(neg_xy), dtype=np.int32)
    return neg_xy, neg_label


def calculate_dice_loss(inputs, targets, num_masks = 1):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def calculate_sigmoid_focal_loss(inputs, targets, num_masks = 1, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks

def keep_component_with_point(mask, point_xy_img, img_shape):
    """
    Keep only the connected component that touches the prompt point.
    Handles resolution mismatch between SAM mask and full image.
    """
    Hm, Wm = mask.shape       # mask resolution (256x256 typically)
    Hi, Wi = img_shape[:2]    # image resolution (e.g., 600x600)

    # map image coords => mask coords
    sx = Wm / Wi
    sy = Hm / Hi

    px, py = point_xy_img     # (x_img, y_img)

    mx = int(round(px * sx))
    my = int(round(py * sy))

    mx = np.clip(mx, 0, Wm - 1)
    my = np.clip(my, 0, Hm - 1)

    # CC based on mask resolution
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    label_at_point = labels[my, mx]

    if label_at_point == 0:
        return mask     # if prompt falls on background

    return (labels == label_at_point).astype(np.uint8)


def mask_iou(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0
 
if __name__ == '__main__':
    main()