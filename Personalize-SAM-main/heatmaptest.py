from persam_f import *
def print_heatmap_selection_point(sim):
    topk_xy, topk_label = point_selection(sim, topk=1)
    print("sim:", sim)

    sim_np = sim.detach().cpu().numpy() if isinstance(sim, torch.Tensor) else np.array(sim)
    prior_vis_ref = os.path.join('location_prior_ref.jpg')
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

if __name__ == "__main__":
    print("======> Load SAM" )
    sam_type, sam_ckpt = 'vit_h', '../sam_vit_h_4b8939.pth'
    sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device)
    
    for name, param in sam.named_parameters():
        param.requires_grad = False
    predictor = SamPredictor(sam)

    ref_image_path = '/Users/tangxiaohan/Desktop/2025 school/csc490/Cups.v3i.coco-segmentation/test/Images/cup-108-_jpg.rf.fd1eee9e0f61ca9b8e9c27fd4b389624.jpg'
    ref_mask_path = '/Users/tangxiaohan/Desktop/2025 school/csc490/Cups.v3i.coco-segmentation/test/Annotations/cup-108-_jpg.rf.fd1eee9e0f61ca9b8e9c27fd4b389624.png'
    # Load images and masks
    ref_image = cv2.imread(ref_image_path)
    if ref_image is None:
        raise FileNotFoundError(f"Could not read reference image: {ref_image_path}. Verify --data, object name, index, and extension.")
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

    ref_mask = cv2.imread(ref_mask_path)
    if ref_mask is None:
        raise FileNotFoundError(f"Could not read reference mask: {ref_mask_path}. Verify the mask exists and path is correct.")
    ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)

    print("======> Obtain Self Location Prior" )
    # Image features encoding
    # side effect: computes and stores the model’s image features in predictor.features.
    # returns a mask-like tensor (a prompt / processed mask) which we assign back to ref_mask.
    ref_mask = predictor.set_image(ref_image, ref_mask)
    ref_feat = predictor.features.squeeze().permute(1, 2, 0)

    ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
    ref_mask = ref_mask.squeeze()[0]

    # Target feature extraction describing the handle appearance
    target_feat = ref_feat[ref_mask > 0]
    target_feat_mean = target_feat.mean(0)
    target_feat_max = torch.max(target_feat, dim=0)[0]
    target_feat = (target_feat_max / 2 + target_feat_mean / 2).unsqueeze(0)

    # Cosine similarity
    h, w, C = ref_feat.shape
    target_feat = target_feat / target_feat.norm(dim=-1, keepdim=True)
    ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
    ref_feat = ref_feat.permute(2, 0, 1).reshape(C, h * w)
    sim = target_feat @ ref_feat

    sim = sim.reshape(1, 1, h, w)
    sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
    sim = predictor.model.postprocess_masks(
                    sim,
                    input_size=predictor.input_size,
                    original_size=predictor.original_size).squeeze()
    print_heatmap_selection_point(sim)