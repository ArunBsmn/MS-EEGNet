"""Shared utilities for the LSTM-CNN imagined speech framework.

Covers reproducibility seeding, normalisation-statistics persistence,
timing helpers, and model-summary serialisation.  All functions are
stateless and carry no dependency on other ``core_*`` modules.
"""
from __future__ import annotations

import os
import pickle
import random
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ── Reproducibility ───────────────────────────────────────────────────────────

def set_all_seeds(seed: int = 37) -> None:
    """Fix seeds for Python, NumPy, and PyTorch for reproducible runs.

    Sets ``torch.backends.cudnn.deterministic = True`` and
    ``torch.backends.cudnn.benchmark = False`` so that GPU operations
    are deterministic at the cost of a small speed penalty.

    Args:
        seed: Integer seed value. Default matches the published study (37).
    """
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ── Timing ────────────────────────────────────────────────────────────────────

def format_time(seconds: float) -> str:
    """Format an elapsed duration as a human-readable string.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        String of the form ``'Xh Ym Zs'``, omitting leading zero fields
        (e.g. ``'3m 7s'`` for 187 seconds).
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m or h:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


class Timer:
    """Context manager for measuring elapsed wall-clock time.

    Usage::

        with Timer() as t:
            run_experiment()
        print(f"Done in {t.elapsed_str}")
    """

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed: float = time.perf_counter() - self._start
        self.elapsed_str: str = format_time(self.elapsed)


# ── Model summary ─────────────────────────────────────────────────────────────

def save_model_summary(
    model:        nn.Module,
    input_shape:  Tuple[int, ...],
    path:         str,
    device:       Optional[torch.device] = None,
) -> None:
    """Write a layer-by-layer parameter summary to a plain-text file.

    Counts trainable and non-trainable parameters per named module and
    appends a grand total.  Does not require torchinfo or torchsummary.

    Args:
        model:       Model to summarise (moved to *device* temporarily).
        input_shape: Shape of a single input sample, e.g. ``(1, 1280)``.
                     A batch dimension of 1 is prepended automatically.
        path:        Destination ``.txt`` file path.
        device:      Device for the dummy forward pass (CPU if None).
    """
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    lines = [f"{'Module':<50} {'Trainable':>12} {'Frozen':>12}"]
    lines.append("-" * 76)

    total_train = total_frozen = 0
    for name, module in model.named_modules():
        if name == "":
            continue
        t = sum(p.numel() for p in module.parameters(recurse=False)
                if p.requires_grad)
        f = sum(p.numel() for p in module.parameters(recurse=False)
                if not p.requires_grad)
        if t or f:
            lines.append(f"{name:<50} {t:>12,} {f:>12,}")
            total_train += t
            total_frozen += f

    lines.append("-" * 76)
    lines.append(f"{'TOTAL':<50} {total_train:>12,} {total_frozen:>12,}")

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def save_metadata(metadata: dict, path: str) -> None:
    """Serialise an arbitrary metadata dict to a plain key=value text file.

    Useful for recording hyperparameters, dataset info, and git hashes
    alongside saved model weights.

    Args:
        metadata: Dict of string-serialisable key-value pairs.
        path:     Destination ``.txt`` file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        for k, v in metadata.items():
            fh.write(f"{k} = {v}\n")