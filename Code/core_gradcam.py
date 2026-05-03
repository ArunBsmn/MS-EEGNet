"""Grad-CAM and Smooth Grad-CAM for the MS-EEGNet framework (Chapter 5).

Implements the saliency analysis described in Section 5.3.4.  Hooks are
registered on the target layer (default: backbone.drop2) to capture
feature-map activations and their gradients.  Two heatmap variants are
provided:

generate_heatmap        — standard single-pass Grad-CAM.
generate_heatmap_smoothed — averages heatmaps over V=25 noisy copies of
                            the input, scaled by η=0.1 × σ_x (Eq. 5.3).

GradCAMPlotter extends GradCAM with per-class visualisation across all
trials, used for the cross-task saliency figures in Chapter 5.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

plt.rcParams.update({
    "font.family":        "Arial",
    "font.size":          12,
    "axes.titlesize":     14,
    "axes.titleweight":   "bold",
    "axes.labelsize":     12,
    "axes.labelweight":   "bold",
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "figure.dpi":         300,
    "lines.linewidth":    1.5,
    "axes.grid":          True,
    "grid.linestyle":     "--",
    "grid.alpha":         0.7,
})


class GradCAM:
    """Grad-CAM with optional Smooth-GradCAM wrapper.

    Registers forward and backward hooks on *target_layer* to capture
    activations and their gradients.  Supports feature maps of shape
    (B, C, 1, T) (2-D conv output) or (B, C, T) (1-D conv output).

    Args:
        model:        Trained MSEEGNet model in eval mode.
        target_layer: Layer name string or ``nn.Module`` instance.
                      Chapter 5 default: ``"backbone.drop2"`` or
                      ``model.drop2``.
        device:       Compute device string or ``torch.device``.
    """

    def __init__(
        self,
        model:        nn.Module,
        target_layer: Union[str, nn.Module],
        device:       Union[str, torch.device] = "cpu",
    ) -> None:
        self.device      = torch.device(device)
        self.model       = model.to(self.device).eval()
        self.activations: Optional[torch.Tensor] = None
        self.gradients:   Optional[torch.Tensor] = None
        self._set_target(target_layer)
        self._register_hooks()

    # ── Hook setup ────────────────────────────────────────────────────────

    def _set_target(self, layer: Union[str, nn.Module]) -> None:
        self.target_layer = (
            dict(self.model.named_modules())[layer]
            if isinstance(layer, str) else layer
        )

    def _register_hooks(self) -> None:
        self.target_layer.register_forward_hook(
            lambda m, _i, o: setattr(self, "activations", o.detach())
        )
        self.target_layer.register_full_backward_hook(
            lambda m, _gi, go: setattr(self, "gradients", go[0].detach())
        )

    # ── Core heatmap ──────────────────────────────────────────────────────

    def generate_heatmap(
        self,
        x:            torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Standard single-pass Grad-CAM.

        Handles feature maps of shape (B, C, 1, T_out) or (B, C, T_out).

        Args:
            x:            Input tensor of shape (1, 1, T).
            target_class: Class index to explain; defaults to predicted class.

        Returns:
            1-D heatmap array of length T (interpolated to input length).
        """
        T_in = x.shape[-1]
        x    = x.to(self.device).requires_grad_(True)

        out = self.model(x)
        if isinstance(out, tuple):
            out = out[0]
        out_flat = out.view(out.size(0), -1)
        if target_class is None:
            target_class = int(out_flat[0].argmax().item())

        self.model.zero_grad(set_to_none=True)
        one_hot = torch.zeros_like(out_flat)
        one_hot[0, target_class] = 1.0
        out.backward(gradient=one_hot.view_as(out))

        g = self.gradients[0]                          # (C, [1,] T_out)
        a = self.activations[0]
        w = g.mean(dim=tuple(range(1, g.ndim)))        # (C,)
        cam = (a * w.view(-1, *[1] * (g.ndim - 1))).sum(dim=0).squeeze()
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-10)

        # Interpolate to input length
        cam_np = cam.cpu().float().numpy()
        if cam_np.shape[0] != T_in:
            x_old = np.linspace(0, 1, cam_np.shape[0])
            x_new = np.linspace(0, 1, T_in)
            cam_np = np.interp(x_new, x_old, cam_np)
        return cam_np

    def generate_heatmap_smoothed(
        self,
        x:            torch.Tensor,
        target_class: Optional[int] = None,
        n_samples:    int   = 25,
        noise_sigma:  float = 0.1,
    ) -> np.ndarray:
        """Smooth Grad-CAM (Eq. 5.3 in Chapter 5).

        Averages standard Grad-CAM maps over *n_samples* noisy copies of
        the input.  Noise is scaled by η × σ_x where η = noise_sigma.

        Args:
            x:            Input tensor of shape (1, 1, T).
            target_class: Class index; defaults to predicted class on clean input.
            n_samples:    Number of noisy copies V (default 25).
            noise_sigma:  Relative noise scale η (default 0.1).

        Returns:
            Smoothed 1-D heatmap of length T.
        """
        if target_class is None:
            with torch.no_grad():
                out = self.model(x.to(self.device))
                if isinstance(out, tuple):
                    out = out[0]
                target_class = int(out.view(out.size(0), -1)[0].argmax().item())

        sigma = float(x.std().item()) * noise_sigma
        maps  = []
        for _ in range(n_samples):
            x_noisy = x + torch.randn_like(x) * sigma
            maps.append(self.generate_heatmap(x_noisy, target_class))
        return np.mean(maps, axis=0)


# ── Visualisation helper ──────────────────────────────────────────────────────

class GradCAMPlotter(GradCAM):
    """Extends GradCAM with per-class trial-averaged heatmap visualisation.

    Used to generate the cross-task saliency figures (Fig. 5.3) comparing
    B4 and MC models on the same EEG signal.
    """

    def plot_subject_class_heatmaps(
        self,
        data:         np.ndarray,
        labels:       np.ndarray,
        subject_name: str,
        out_dir:      str,
        suffix:       str,
        use_smooth:   bool  = True,
        n_samples_sg: int   = 25,
        noise_sigma:  float = 0.1,
        cmap:         str   = "cividis",
        signal_color: str   = "white",
        signal_alpha: float = 0.7,
        signal_lw:    float = 1.5,
    ) -> None:
        """Generate and save per-class heatmap figures for one subject.

        Args:
            data:         Array of shape (N, 1, T) — channel-wise windows.
            labels:       Integer label array of shape (N,).
            subject_name: Subject identifier for file naming.
            out_dir:      Output directory for SVG files.
            suffix:       Task suffix appended to file names (e.g. 'B4', 'MC').
            use_smooth:   Use Smooth Grad-CAM if True (default).
            n_samples_sg: Noisy samples for Smooth Grad-CAM (default 25).
            noise_sigma:  Relative noise scale η (default 0.1).
            cmap:         Matplotlib colourmap name (default 'cividis').
            signal_color: Signal line colour (default 'white').
            signal_alpha: Signal line opacity (default 0.7).
            signal_lw:    Signal line width (default 1.5).
        """
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        for cls in sorted(np.unique(labels)):
            idx = np.where(labels == cls)[0]
            if idx.size == 0:
                continue

            cams:    List[np.ndarray] = []
            signals: List[np.ndarray] = []

            for i in idx:
                x = torch.from_numpy(
                    data[i:i + 1].astype(np.float32)
                ).to(self.device)
                hm = (
                    self.generate_heatmap_smoothed(
                        x, target_class=int(cls),
                        n_samples=n_samples_sg, noise_sigma=noise_sigma,
                    )
                    if use_smooth else
                    self.generate_heatmap(x, target_class=int(cls))
                )
                cams.append(hm)
                signals.append(data[i, 0])

            self._plot_class(
                cams, signals, subject_name, cls, suffix,
                out_dir, cmap, signal_alpha, signal_color, signal_lw,
            )

    def _plot_class(
        self,
        cams:         List[np.ndarray],
        signals:      List[np.ndarray],
        subject_name: str,
        cls:          int,
        suffix:       str,
        out_dir:      str,
        cmap:         str,
        signal_alpha: float,
        signal_color: str,
        signal_lw:    float,
    ) -> None:
        avg_cam    = np.mean(cams, axis=0)
        avg_signal = np.mean(signals, axis=0)
        T          = len(avg_signal)

        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(
            avg_cam[np.newaxis, :], aspect="auto", cmap=cmap, alpha=0.85,
            extent=[0, T, float(avg_signal.min()), float(avg_signal.max())],
        )
        ax.plot(avg_signal, color=signal_color, alpha=signal_alpha,
                linewidth=signal_lw, label="Signal")
        plt.colorbar(im, ax=ax, label="Grad-CAM intensity")
        ax.set_title(
            f"{subject_name} | Class {cls} | {suffix} "
            f"({'Smooth ' if len(cams) > 1 else ''}Grad-CAM, n={len(cams)} trials)"
        )
        ax.set_xlabel("Time steps")
        ax.set_ylabel("Amplitude")
        plt.tight_layout()

        fname = os.path.join(out_dir, f"{subject_name}_cls{cls}_{suffix}.svg")
        plt.savefig(fname, format="svg", bbox_inches="tight")
        plt.close(fig)