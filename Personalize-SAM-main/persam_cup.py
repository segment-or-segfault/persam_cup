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
    test_masks_path = os.path.join(masks_path, obj_name or '')

    output_path = os.path.join(output_path, obj_name or '')
    os.makedirs(output_path, exist_ok=True)

    # Load images and masks
    ref_image = cv2.imread(ref_image_path)
    if ref_image is None:
        raise FileNotFoundError(f"Could not read reference image: {ref_image_path}. Verify --data, object name, index, and extension.")
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
        sim = rotate_sim_back(sim, ang)
        sims.append(sim)

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
    mask_weights = Mask_Weights().to(device)
    mask_weights.train()
    
    optimizer = torch.optim.AdamW(mask_weights.parameters(), lr=args.lr, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.train_epoch)

    # Train combination weights
    for train_idx in range(args.train_epoch):

        # Run the decoder
        # Uses the point we computed to tell SAM where to look
        masks, scores, logits, logits_high = predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            multimask_output=True)
        logits_high = logits_high.flatten(1)

        # Weighted sum three-scale masks
        weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
        logits_high = logits_high * weights
        logits_high = logits_high.sum(0).unsqueeze(0)

        dice_loss = calculate_dice_loss(logits_high, gt_mask)
        focal_loss = calculate_sigmoid_focal_loss(logits_high, gt_mask)
        loss = dice_loss + focal_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if train_idx % args.log_epoch == 0:
            print('Train Epoch: {:} / {:}'.format(train_idx, args.train_epoch))
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Dice_Loss: {:.4f}, Focal_Loss: {:.4f}'.format(current_lr, dice_loss.item(), focal_loss.item()))


    mask_weights.eval()
    weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
    weights_np = weights.detach().cpu().numpy()
    print('======> Mask weights:\n', weights_np)

    accuracies = {}
    for alpha in [0]:
        print('======> Start Testing')
        correct = 0
        test_images = os.listdir(test_images_path)
        for test_idx in tqdm(range(len(os.listdir(test_images_path)))):

            # Load test image
            if obj_name is None:
                # single-folder mode: use actual filenames from the directory
                test_image_path = os.path.join(test_images_path, test_images[test_idx])
                test_mask_path = os.path.join(test_masks_path, test_images[test_idx].replace(".jpg", ".png"))
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

            fused_sim = (sim_dino_norm ** alpha) * (sim_sam_norm ** (1 - alpha))

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

            # Weighted sum three-scale masks
            logits_high = logits_high * weights.unsqueeze(-1)
            logit_high = logits_high.sum(0)
            mask = (logit_high > 0).detach().cpu().numpy()

            logits = logits * weights_np[..., None]
            logit = logits.sum(0)

            # Cascaded Post-refinement-1
            y, x = np.nonzero(mask)
            x_min = x.min()
            x_max = x.max()
            y_min = y.min()
            y_max = y.max()
            input_box = np.array([x_min, y_min, x_max, y_max])
            masks, scores, logits, _ = predictor.predict(
                point_coords=topk_xy,
                point_labels=topk_label,
                box=input_box[None, :],
                mask_input=logit[None, :, :],
                multimask_output=True)
            best_idx = np.argmax(scores)

            # Cascaded Post-refinement-2
            y, x = np.nonzero(masks[best_idx])
            x_min = x.min()
            x_max = x.max()
            y_min = y.min()
            y_max = y.max()
            input_box = np.array([x_min, y_min, x_max, y_max])
            masks, scores, logits, _ = predictor.predict(
                point_coords=topk_xy,
                point_labels=topk_label,
                box=input_box[None, :],
                mask_input=logits[best_idx: best_idx + 1, :, :],
                multimask_output=True)
            best_idx = np.argmax(scores)
            
            # Save masks
            plt.figure(figsize=(10, 10))
            plt.imshow(test_image)
            show_mask(masks[best_idx], plt.gca())
            show_points(topk_xy, topk_label, plt.gca())
            plt.title(f"Mask {best_idx}", fontsize=18)
            plt.axis('off')
            base = os.path.basename(test_image_path) 
            vis_test_image = os.path.splitext(base)[0] 
            vis_mask_output_path = os.path.join(output_path, f'vis_mask_{vis_test_image}.jpg')
            with open(vis_mask_output_path, 'wb') as outfile:
                plt.savefig(outfile, format='jpg')

            final_mask = masks[best_idx]
            real_mask = np.array(Image.open(test_mask_path).convert("L")) > 0

            threshold = 0.7
            iou = evaluate_iou(final_mask, real_mask)
            print(f"Test Image: {test_image_path}, IoU: {iou:.4f}")
            if iou >= threshold:
                correct += 1
            print(f"test path: {test_mask_path}, correct: {correct}")

            mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
            mask_colors[final_mask, :] = np.array([[0, 0, 128]])
            mask_output_path = os.path.join(output_path, vis_test_image + '.png')
            cv2.imwrite(mask_output_path, mask_colors)

        accuracy = correct / (len(os.listdir(test_images_path)))
        accuracies[alpha] = accuracy
        print(f"Accuracy: {accuracy:.4f}")
    print("Accuracies for different weights:", accuracies)


class Mask_Weights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)

def evaluate_iou(mask_pred, mask_gt):
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    iou = intersection / union if union > 0 else 0
    return iou

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
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()
    print("topk_xy:", topk_xy)
    return topk_xy, topk_label

def negative_point_selection(pos_xy, mask_sim, threshold=0.80, step=5000, window=100):
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


if __name__ == '__main__':
    main()