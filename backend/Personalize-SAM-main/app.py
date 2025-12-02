from flask import Flask, request, send_file, jsonify
import numpy as np
import cv2
import torch
from PIL import Image
import tempfile
import os
from flask_cors import CORS

# ===== your imports =====
from transformers import AutoImageProcessor, AutoModel
from per_segment_anything import sam_model_registry, SamPredictor
from per_segment_anything.adaptive_fusion import AdaptiveMaskFusion
from show import *
from persam_f import (
    extract_dino_features,
    get_color_outline,
    combine_edges_with_image,
    point_selection,
    negative_point_selection,
    rotate_image,
    keep_component_with_point,
    mask_iou,
)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# =============================
#          LOAD MODELS
# =============================
device = "cuda" if torch.cuda.is_available() else \
         "mps" if torch.backends.mps.is_available() else "cpu"

dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
dino_model = AutoModel.from_pretrained("facebook/dinov2-large").to(device)
dino_model.eval()

sam = sam_model_registry["vit_h"](checkpoint="backend/sam_vit_h_4b8939.pth").to(device)
predictor = SamPredictor(sam)

fusion = AdaptiveMaskFusion().to(device)
fusion.eval()     # we use trained weights or init weights


# ===========================================================================
#                              MAIN ENDPOINT
# ===========================================================================
@app.post("/segment")
def segment():

    # =============================
    #        READ INPUT FILES
    # =============================
    ref_image = cv2.imdecode(np.frombuffer(
        request.files["ref_image"].read(), np.uint8), cv2.IMREAD_COLOR)
    ref_mask = cv2.imdecode(np.frombuffer(
        request.files["ref_mask"].read(), np.uint8), cv2.IMREAD_COLOR)
    query_image = cv2.imdecode(np.frombuffer(
        request.files["test_image"].read(), np.uint8), cv2.IMREAD_COLOR)

    if ref_image is None or ref_mask is None or query_image is None:
        return jsonify({"error": "Invalid images"}), 400

    ref_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    query_rgb = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)

    # ======================================================
    #      1. Build edge-enhanced images
    # ======================================================
    ref_edge = get_color_outline(ref_rgb)
    ref_in = combine_edges_with_image(ref_rgb, ref_edge)

    query_edge = get_color_outline(query_rgb)
    query_in = combine_edges_with_image(query_rgb, query_edge)

    # ======================================================
    #      2. Compute REF DINO prototype
    # ======================================================
    ref_dino_hw, ref_dino_flat, (Hf, Wf) = extract_dino_features(ref_in)

    ref_mask_gray = ref_mask[:, :, 0] > 0
    ref_mask_small = cv2.resize(ref_mask_gray.astype(np.uint8),
                                (Wf, Hf),
                                interpolation=cv2.INTER_NEAREST)
    ref_mask_small = torch.tensor(ref_mask_small, device=device, dtype=torch.bool)

    proto_dino = ref_dino_hw[ref_mask_small].mean(0)
    proto_dino = proto_dino / proto_dino.norm()

    # ======================================================
    #      3. Compute REF SAM prototype
    # ======================================================
    predictor.set_image(ref_in)
    ref_feat = predictor.features.squeeze().permute(1, 2, 0)     # H×W×C
    Hs, Ws, C = ref_feat.shape

    ref_mask_sam = cv2.resize(ref_mask_gray.astype(np.uint8),
                              (Ws, Hs),
                              interpolation=cv2.INTER_NEAREST)

    ref_feat_masked = ref_feat[ref_mask_sam > 0]
    f_mean = ref_feat_masked.mean(0)
    f_max = ref_feat_masked.max(0)[0]
    target_feat = (f_mean + f_max) / 2
    target_feat = target_feat.unsqueeze(0)
    target_feat = target_feat / target_feat.norm()

    # ======================================================
    #      4. Compute Query DINO similarity
    # ======================================================
    q_dino_hw, _, (Hq, Wq) = extract_dino_features(query_in)
    sim_dino = torch.einsum("c,hwc->hw", proto_dino, q_dino_hw)
    sim_dino_up = cv2.resize(sim_dino.cpu().numpy(),
                             (query_rgb.shape[1], query_rgb.shape[0]),
                             interpolation=cv2.INTER_LINEAR)

    # ======================================================
    #      5. Compute Query SAM similarity
    # ======================================================
    predictor.set_image(query_in)
    test_feat = predictor.features.squeeze()                # C×H×W
    C, Hs, Ws = test_feat.shape
    test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)

    test_feat_flat = test_feat.reshape(C, Hs * Ws)
    D = target_feat.shape[-1]
    sim_sam = (target_feat @ test_feat_flat / (D**0.5)).reshape(Hs, Ws)

    sim_sam_up = predictor.model.postprocess_masks(
        sim_sam.unsqueeze(0).unsqueeze(0),
        input_size=predictor.input_size,
        original_size=predictor.original_size
    )[0, 0].cpu().numpy()

    # ======================================================
    #      6. Fuse similarities
    # ======================================================
    d = sim_dino_up
    s = sim_sam_up
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    fused = np.sqrt(d * s)
    fused_t = torch.tensor(fused).to(device)

    # ======================================================
    #      7. Select points
    # ======================================================
    pos_xy, pos_label = point_selection(fused_t, topk=1)
    neg_xy, neg_label = negative_point_selection(pos_xy, fused_t)

    pts_xy = np.concatenate([pos_xy, neg_xy], axis=0)
    pts_lb = np.concatenate([pos_label, neg_label], axis=0)

    # ======================================================
    #      8. SAM First prediction
    # ======================================================
    masks, scores, logits, logits_high = predictor.predict(
        point_coords=pts_xy,
        point_labels=pts_lb,
        multimask_output=True,
    )

    base_best = int(np.argmax(scores))
    base_mask0 = masks[base_best].astype(np.uint8)

    # ======================================================
    #      9. Cascade refinement step 1
    # ======================================================
    y0, x0 = np.nonzero(base_mask0)
    x_min, x_max = x0.min(), x0.max()
    y_min, y_max = y0.min(), y0.max()
    box1 = np.array([x_min, y_min, x_max, y_max])

    masks1, scores1, logits1, _ = predictor.predict(
        point_coords=pts_xy,
        point_labels=pts_lb,
        box=box1[None, :],
        mask_input=logits[base_best:base_best+1],
        multimask_output=True,
    )
    best1 = int(np.argmax(scores1))
    mask1 = masks1[best1].astype(np.uint8)

    # ======================================================
    #      10. Cascade refinement step 2
    # ======================================================
    y1, x1 = np.nonzero(mask1)
    x_min2, x_max2 = x1.min(), x1.max()
    y_min2, y_max2 = y1.min(), y1.max()
    box2 = np.array([x_min2, y_min2, x_max2, y_max2])

    masks2, scores2, logits2, _ = predictor.predict(
        point_coords=pts_xy,
        point_labels=pts_lb,
        box=box2[None, :],
        mask_input=logits1[best1:best1+1],
        multimask_output=True,
    )
    best2 = int(np.argmax(scores2))
    base_final = masks2[best2].astype(np.uint8)

    # ======================================================
    #      11. Keep component touching positive point
    # ======================================================
    final_base = keep_component_with_point(
        base_final,
        pos_xy[0],   # positive point
        query_rgb.shape,
    )

    kernel = np.ones((5, 5), np.uint8)
    base_dil = cv2.dilate(final_base, kernel, iterations=1)

    # ======================================================
    #      12. Adaptive Mask Fusion
    # ======================================================
    logits_high_t = logits_high.to(device=device, dtype=torch.float32)
    fused_logits = fusion(logits_high_t)      # [1, H_mask, W_mask]

    fused_logits = torch.nn.functional.avg_pool2d(
        fused_logits.unsqueeze(0),
        kernel_size=3,
        stride=1,
        padding=1
    ).squeeze(0)

    prob = fused_logits.sigmoid()
    fused_low = (prob.squeeze(0) > 0.2).cpu().numpy().astype(np.uint8)

    fused_low = cv2.morphologyEx(fused_low, cv2.MORPH_CLOSE, kernel, iterations=2)
    fused_low = cv2.dilate(fused_low, kernel, iterations=1)

    fused_low = np.logical_and(fused_low, base_dil).astype(np.uint8)

    overlap = np.logical_and(fused_low, final_base).astype(np.uint8)

    if overlap.sum() == 0:
        fused_low = np.zeros_like(final_base)
    else:
        fused_low = overlap

    # ======================================================
    #      13. Fallback logic
    # ======================================================
    iou_fb = mask_iou(fused_low, final_base)

    if fused_low.sum() == 0 or iou_fb < 0.5 \
        or fused_low.sum() > 1.3 * final_base.sum():
        final_low = final_base
    else:
        final_low = fused_low

    # ======================================================
    #      14. Upsample mask to original image size
    # ======================================================
    final_mask_up = cv2.resize(
        final_low,
        (query_rgb.shape[1], query_rgb.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # ======================================================
    #      15. Return result
    # ======================================================
    color_mask = np.zeros_like(query_rgb)
    color_mask[final_mask_up > 0] = [255, 0, 0]

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(tmp.name, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))

    return send_file(tmp.name, mimetype="image/png")


# ===========================================================================
#                               RUN SERVER
# ===========================================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
