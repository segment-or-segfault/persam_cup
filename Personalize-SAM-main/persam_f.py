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

    ref_mask = cv2.imread(ref_mask_path)
    if ref_mask is None:
        raise FileNotFoundError(f"Could not read reference mask: {ref_mask_path}. Verify the mask exists and path is correct.")
    ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)

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
    

    print("======> Obtain Self Location Prior" )
    # Image features encoding
    # side effect: computes and stores the model’s image features (embedding) in predictor.features.
    # return a mask that you can later use to compute target features.
    predictor.set_image(preprocessed_image)
    preprocessed_feat = predictor.features.squeeze().permute(1, 2, 0)

    ref_mask = predictor.set_image(ref_image, ref_mask)
    ref_feat = predictor.features.squeeze().permute(1, 2, 0)

    ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
    ref_mask = ref_mask.squeeze()[0]

    topk_xy = np.empty((0, 2), dtype=np.int64)
    topk_label = np.empty((0,), dtype=np.int64)
    count = 0
    for feat in [preprocessed_feat, ref_feat]:
        count += 1
        # Target feature extraction describing the handle appearance
        target_feat = feat[ref_mask > 0]
        # Averages all feature vectors inside the masked region
        # Captures the overall, smooth, dominant appearance of the object (stable against noise and small variations)
        target_feat_mean = target_feat.mean(0)
        # Takes the elementwise maximum along each feature channel
        # Highlights the most distinctive or strongest activations among those features (edges, colors, textures that are particularly characteristic)
        target_feat_max = torch.max(target_feat, dim=0)[0]
        # Blends representativeness (mean) with discriminativeness (max)
        target_feat = (target_feat_max / 2 + target_feat_mean / 2).unsqueeze(0)

        # Cosine similarity between target feature and all image features
        h, w, C = feat.shape
        # Normalize features
        target_feat = target_feat / target_feat.norm(dim=-1, keepdim=True)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        feat = feat.permute(2, 0, 1).reshape(C, h * w)
        # gives a cosine similarity map between the target feature and every pixel feature in the image
        sim = target_feat @ feat

        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = predictor.model.postprocess_masks(
                        sim,
                        input_size=predictor.input_size,
                        original_size=predictor.original_size).squeeze()

        # Positive location point on the reference object.
        xy, label = point_selection(sim, topk=1)

        xy = np.asarray(xy, dtype=np.int64).reshape(-1, 2)
        label = np.asarray(label, dtype=np.int64).reshape(-1)

        # Concatenate as numpy arrays
        topk_xy = np.concatenate((topk_xy, xy), axis=0)
        topk_label = np.concatenate((topk_label, label), axis=0)

        # Save reference location prior as a heatmap overlay on the reference image
        try:
            sim_np = sim.detach().cpu().numpy() if isinstance(sim, torch.Tensor) else np.array(sim)
            prior_vis_ref = os.path.join(output_path, 'location_prior_ref_{}.jpg'.format(count))
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

        test_image = cv2.imread(test_image_path)
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

        test_preprocessed_image = cv2.GaussianBlur(test_image, (5, 5), sigmaX=0)
        # test_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)

        # Equalize histogram to enhance contrast / normalize lighting
        # test_gray_eq = cv2.equalizeHist(test_gray)
        # test_preprocessed_image = np.stack([test_gray]*3, axis=-1)

        # Gray test Image feature encoding
        predictor.set_image(test_preprocessed_image)
        test_gray_feat = predictor.features.squeeze()

        # Image feature encoding
        predictor.set_image(test_image)
        test_feat = predictor.features.squeeze()
        
        # Cosine similarity for the test image
        C, h, w = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
        test_feat = test_feat.reshape(C, h * w)
        # For each test image, it finds the spatial location whose feature vector
        # best matches the reference prototype
        test_sim = target_feat @ test_feat

        # Cosine similarity for the gray test image
        C, h, w = test_gray_feat.shape
        test_gray_feat = test_gray_feat / test_gray_feat.norm(dim=0, keepdim=True)
        test_gray_feat = test_gray_feat.reshape(C, h * w)
        # For each test image, it finds the spatial location whose feature vector
        # best matches the reference prototype
        test_gray_sim = target_feat @ test_gray_feat

        topk_xy = np.empty((0, 2), dtype=np.int64)
        topk_label = np.empty((0,), dtype=np.int64)

        for sim in [test_gray_sim, test_sim]:
            sim = sim.reshape(1, 1, h, w)
            sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
            sim = predictor.model.postprocess_masks(
                            sim,
                            input_size=predictor.input_size,
                            original_size=predictor.original_size).squeeze()
            
            # gives the prompt for SAM on the test image
            xy, label = point_selection(sim, topk=1)

            xy = np.asarray(xy, dtype=np.int64).reshape(-1, 2)
            label = np.asarray(label, dtype=np.int64).reshape(-1)

            # Concatenate as numpy arrays
            topk_xy = np.concatenate((topk_xy, xy), axis=0)
            topk_label = np.concatenate((topk_label, label), axis=0)

        # Positive location prior
        # Save test-image location prior as a heatmap overlay
        try:
            vis_test_image = os.path.splitext(os.path.basename(test_image_path))[0]
            sim_np = sim.detach().cpu().numpy() if isinstance(test_sim, torch.Tensor) else np.array(sim)
            prior_vis_path = os.path.join(output_path, f'prior_{vis_test_image}.jpg')
            plt.figure(figsize=(8, 8))
            plt.imshow(test_image)
            plt.imshow(sim_np, cmap='jet', alpha=0.5)
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
        mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
        mask_colors[final_mask, :] = np.array([[0, 0, 128]])
        mask_output_path = os.path.join(output_path, vis_test_image + '.png')
        cv2.imwrite(mask_output_path, mask_colors)


class Mask_Weights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)

def point_selection(mask_sim, topk=1):
    # Top-1 point selection
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
    print("Final selection point (x, y):", topk_xy)
    print("Final selection label:", topk_label)

    return topk_xy, topk_label


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
