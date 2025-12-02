import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AdaptiveMaskFusion(nn.Module):
    def __init__(self, threshold=0.0, alpha=0.42):
        super().__init__()
        self.threshold = threshold
        self.alpha = alpha  # balance: edge vs contrast
        self.learnable_weights = nn.Parameter(torch.rand(3) * 0.1)  # correction term

    def forward(self, logits):
        # logits: [3,H,W] or [1,3,H,W]
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits).to(self.learnable_weights.device)
        if logits.dim() == 3:
            logits = logits.unsqueeze(0)
        B, S, H, W = logits.shape
        probs = logits.sigmoid()

        # sobel kernels
        sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], 
                               dtype=torch.float32, device=logits.device).view(1,1,3,3)
        sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],
                               dtype=torch.float32, device=logits.device).view(1,1,3,3)

        # laplacian
        lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],
                           dtype=torch.float32, device=logits.device).view(1,1,3,3)

        edge_scores = []
        contrast_scores = []

        for i in range(S):
            m = probs[:, i:i+1, :, :]  # [B,1,H,W]

            # sobel edges
            gx = F.conv2d(m, sobel_x, padding=1)
            gy = F.conv2d(m, sobel_y, padding=1)
            sobel_mag = torch.sqrt(gx**2 + gy**2)

            # laplacian edge
            laplace = torch.abs(F.conv2d(m, lap, padding=1))
            edge = sobel_mag + 0.5 * laplace

            # local contrast okay handle has high local contrast
            local_mean = F.avg_pool2d(m, kernel_size=7, stride=1, padding=3)
            local_var = F.avg_pool2d((m - local_mean) ** 2, kernel_size=7, stride=1, padding=3)

            # = edge_strength + contrast_strength
            score = self.alpha * edge.sum((1,2,3)) + (1 - self.alpha) * local_var.sum((1,2,3))
            edge_scores.append(score)


        edge_scores = torch.stack(edge_scores, dim=1)  # [B, S]

        # normalize and add learnable correction
        w_edge = edge_scores / (edge_scores.sum(1, keepdim=True) + 1e-6)
        w_learnable = torch.softmax(self.learnable_weights, dim=0)   # [3]
        weights = (w_edge + w_learnable) / 2
        weights = weights.view(B, S, 1, 1)
        #TODO: need training and update 


        # wighted logits fusion
        fused = (logits * weights).sum(dim=1)   # [B,H,W]
        return fused
