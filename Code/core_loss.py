"""Sub-centre ArcFace loss for the MS-EEGNet framework (Chapter 5).

Implements Eq. 5.2 from Chapter 5.  Each class is represented by K
learnable sub-centre weight vectors; the cosine similarity to the
closest sub-centre is used for the angular margin computation.  The
additive angular margin m is applied exclusively to the ground-truth
class logit before rescaling all logits by s.

Default hyperparameters match the published study:
    K = 2 sub-centres per class
    m = 0.3 radians additive angular margin
    s = 30 logit scale

The forward pass returns (loss, predictions) so that the training loop
can track accuracy without a separate forward pass.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubCentreArcFaceLoss(nn.Module):
    """Sub-centre ArcFace loss with configurable K, margin, and scale.

    Args:
        num_classes:     Number of output classes N.
        feature_dim:     Embedding dimensionality D_e.
        num_sub_centres: Sub-centres per class K (default 2).
        scale:           Logit scale s (default 30.0).
        margin:          Additive angular margin m in radians (default 0.3).
    """

    def __init__(
        self,
        num_classes:     int,
        feature_dim:     int,
        num_sub_centres: int   = 2,
        scale:           float = 30.0,
        margin:          float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes     = num_classes
        self.feature_dim     = feature_dim
        self.num_sub_centres = num_sub_centres
        self.scale           = scale

        # Sub-centre weight matrix: (N, K, D_e)
        self.sub_centres = nn.Parameter(
            torch.empty(num_classes, num_sub_centres, feature_dim)
        )
        nn.init.xavier_uniform_(self.sub_centres)

        # Pre-compute margin constants for angle addition identity
        self.register_buffer("cos_m", torch.cos(torch.tensor(margin)))
        self.register_buffer("sin_m", torch.sin(torch.tensor(margin)))

    def forward(
        self,
        features: torch.Tensor,
        labels:   torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute loss and return predictions.

        Args:
            features: L2-normalised embeddings of shape (B, D_e).
            labels:   Ground-truth class indices of shape (B,).

        Returns:
            (loss, predictions) — scalar loss and predicted class tensor.
        """
        device = features.device

        # Normalise embeddings and sub-centres
        feat = F.normalize(features, dim=1, eps=1e-8)             # (B, D_e)
        wts  = F.normalize(
            self.sub_centres.to(device), dim=2, eps=1e-8
        )                                                          # (N, K, D_e)

        # Cosine similarity to every sub-centre: (B, N, K)
        cos_sim = torch.einsum("bd,nkd->bnk", feat, wts)

        # Max cosine similarity per class: (B, N)
        max_cos, _ = cos_sim.max(dim=2)
        max_cos = max_cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Angular margin via angle addition identity: cos(θ + m)
        sin_theta = (1.0 - max_cos ** 2).clamp(min=1e-14).sqrt()
        cos_with_margin = max_cos * self.cos_m - sin_theta * self.sin_m

        # Apply margin only to ground-truth class
        logits = max_cos.clone()
        batch_idx = torch.arange(features.size(0), device=device)
        logits[batch_idx, labels] = cos_with_margin[batch_idx, labels]

        loss        = F.cross_entropy(self.scale * logits, labels)
        predictions = logits.argmax(dim=1)
        return loss, predictions