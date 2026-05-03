"""MS-EEGNet architecture for the phonetic similarity study (Chapter 5).

The model processes single-channel EEG through two stages:

1. WaveletDecomp — frozen db4 convolutional filters decompose the signal
   into F=6 frequency bands (5 detail + 1 approximation), aligned to a
   common temporal length and stacked into (B, 1, F, T).

2. MSEEGNet — an EEGNet variant adapted to spectral-temporal dynamics.
   Block 1 applies temporal convolution across all bands, followed by
   depthwise convolution that collapses the band dimension (spectral
   mixing). Block 2 applies separable convolution. A two-layer MLP
   projection head produces the embedding e ∈ R^{D_e} passed to
   SubCentreArcFaceLoss.

The model follows the same channel-wise input convention as Chapter 4:
each EEG channel is processed independently as a (B, 1, T) tensor, so
the spatial axis is absent and depthwise convolution operates on the
wavelet band axis instead.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Wavelet decomposition ─────────────────────────────────────────────────────

# Daubechies-4 filter coefficients (reversed for conv1d correlation).
_DB4_HI = [-0.2304,  0.7148, -0.6309, -0.0280,  0.1870,  0.0308, -0.0329, -0.0106]
_DB4_LO = [-0.0106,  0.0329,  0.0308, -0.1870, -0.0280,  0.6309,  0.7148,  0.2304]


class WaveletDecomp(nn.Module):
    """Multi-scale decomposition via frozen db4 convolutional filters.

    Applies *levels* pairs of high/low-pass filters recursively, passing
    the low-pass output as input to the next level.  All detail and the
    final approximation are aligned via adaptive average pooling and
    stacked into (B, 1, levels+1, T).

    Args:
        levels:    Number of decomposition levels (default 5 → F=6 bands).
        learnable: If True, filters are trainable (default False — frozen).
    """

    def __init__(self, levels: int = 5, learnable: bool = False) -> None:
        super().__init__()
        self.levels = levels
        hi = torch.tensor(_DB4_HI, dtype=torch.float32).view(1, 1, -1)
        lo = torch.tensor(_DB4_LO, dtype=torch.float32).view(1, 1, -1)
        k, pad = hi.shape[-1], hi.shape[-1] // 2

        self.blocks = nn.ModuleList()
        for _ in range(levels):
            hp = nn.Conv1d(1, 1, k, padding=pad, bias=False)
            lp = nn.Conv1d(1, 1, k, padding=pad, bias=False)
            hp.weight = nn.Parameter(hi.clone(), requires_grad=learnable)
            lp.weight = nn.Parameter(lo.clone(), requires_grad=learnable)
            self.blocks.append(nn.ModuleDict({"hi": hp, "lo": lp}))

        self.act = nn.LeakyReLU(0.1)

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, T)

        Returns:
            (B, 1, levels+1, T) — stacked multi-scale bands.
        """
        x = x.view(x.size(0), 1, -1)
        bands, cur = [], x
        for blk in self.blocks:
            hi_out = self.act(F.layer_norm(blk["hi"](cur), blk["hi"](cur).shape[1:]))
            lo_out = self.act(F.layer_norm(blk["lo"](cur), blk["lo"](cur).shape[1:]))
            bands.append(hi_out)
            cur = lo_out
        bands.append(cur)   # final approximation

        T = max(b.shape[-1] for b in bands)
        aligned = [F.adaptive_avg_pool1d(b, T) for b in bands]
        return torch.stack(aligned, dim=2).unsqueeze(1)  # (B, 1, F, T)


# ── MS-EEGNet ─────────────────────────────────────────────────────────────────

class MSEEGNet(nn.Module):
    """Multi-Scale EEGNet for single-channel wavelet-decomposed EEG.

    Architecture (from Eq. 5.1 in Chapter 5):

    Block 1:
        Temporal conv  (F1 × 1 × 1 × κ)  → BN
        Depthwise conv (F1·D × 1 × F × 1) → BN → ELU → AvgPool(1,4) → Dropout

    Block 2:
        Separable conv (depthwise 16 + pointwise 1×1) → BN → ELU
        AvgPool(1,8) → Dropout

    Projection head:
        Flatten → Linear(hidden) → ReLU → Linear(proj_dim) → BN1d
        Output e ∈ R^{proj_dim} is passed to SubCentreArcFaceLoss.

    Args:
        n_classes:   Number of output classes (used only for MODEL_REGISTRY).
        data_length: Input temporal length T (samples at 256 Hz; default 1280 = 5s).
        input_dim:   Input channels (always 1 for channel-wise processing).
        config:      Optional dict overriding defaults below.

    Config keys
    -----------
    wd_levels     int    Wavelet decomposition levels (default 5 → F=6).
    F1            int    Temporal conv output channels (default 8).
    D             int    Depth multiplier for depthwise conv (default 2).
    F2            int    Pointwise output channels (default F1*D).
    kern_length   int    Temporal conv kernel width κ (default 64).
    dropout       float  Dropout rate (default 0.5).
    projection_dim int   Embedding dimension D_e (default 128).
    """

    _DEFAULTS: Dict = {
        "wd_levels":     5,
        "F1":            8,
        "D":             2,
        "F2":            None,   # resolved to F1*D
        "kern_length":   64,
        "dropout":       0.5,
        "projection_dim": 128,
    }

    def __init__(
        self,
        n_classes:   int,
        data_length: int = 1280,
        input_dim:   int = 1,
        config:      Optional[Dict] = None,
    ) -> None:
        super().__init__()
        cfg = {k: (config or {}).get(k, v) for k, v in self._DEFAULTS.items()}
        if cfg["F2"] is None:
            cfg["F2"] = cfg["F1"] * cfg["D"]

        F1, D, F2       = cfg["F1"], cfg["D"], cfg["F2"]
        kern            = cfg["kern_length"]
        drop            = cfg["dropout"]
        wd_levels       = cfg["wd_levels"]
        n_bands         = wd_levels + 1     # F = 6 by default
        proj_dim        = cfg["projection_dim"]

        self.wavelet = WaveletDecomp(levels=wd_levels, learnable=False)

        # Block 1
        self.temp_conv   = nn.Conv2d(input_dim, F1,
                                     kernel_size=(1, kern),
                                     padding=(0, kern // 2), bias=False)
        self.bn1         = nn.BatchNorm2d(F1)
        self.spatial_conv = nn.Conv2d(F1, F1 * D,
                                      kernel_size=(n_bands, 1),
                                      groups=F1, bias=False)
        self.bn2         = nn.BatchNorm2d(F1 * D)
        self.elu1        = nn.ELU()
        self.pool1       = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1       = nn.Dropout(drop)

        # Block 2
        self.depth_conv  = nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16),
                                     padding=(0, 8), groups=F1 * D, bias=False)
        self.point_conv  = nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
        self.bn3         = nn.BatchNorm2d(F2)
        self.elu2        = nn.ELU()
        self.pool2       = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2       = nn.Dropout(drop)

        # Projection head
        flat_dim = self._flat_dim(data_length, input_dim)
        hidden   = max(proj_dim * 2, 256)
        self.proj = nn.Sequential(
            nn.Linear(flat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(drop) if drop > 0 else nn.Identity(),
            nn.Linear(hidden, proj_dim),
            nn.BatchNorm1d(proj_dim),
        )

    def _flat_dim(self, data_length: int, input_dim: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, input_dim, data_length)
            return self._backbone(dummy).flatten(1).shape[1]

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        x = self.wavelet(x)          # (B, 1, F, T)
        x = self.bn1(self.temp_conv(x))
        x = self.bn2(self.spatial_conv(x))
        x = self.drop1(self.pool1(self.elu1(x)))
        x = self.bn3(self.point_conv(self.depth_conv(x)))
        x = self.drop2(self.pool2(self.elu2(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, T)

        Returns:
            e: (B, proj_dim) — L2-normalised embedding.
        """
        return self.proj(self._backbone(x).flatten(1))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw backbone features (before projection head)."""
        return self._backbone(x).flatten(1)


MODEL_REGISTRY: Dict[str, type] = {"MSEEGNet": MSEEGNet}