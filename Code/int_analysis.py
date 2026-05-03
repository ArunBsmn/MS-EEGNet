"""Cross-task saliency overlap analysis for the MS-EEGNet framework.

Implements the quantitative overlap analysis described in Section 5.3.5:

1. Binary thresholding of Grad-CAM maps at the top-P% most salient
   time points (default P=15).
2. Jaccard similarity between thresholded maps from two tasks on the
   same EEG signal.
3. Statistical significance via Wilcoxon signed-rank test and paired
   permutation test.
4. Random baseline generation for Phase-3 control analysis.
5. Per-subject result persistence as .npy files for downstream loading.

Public API
----------
threshold_top_p      — binary mask of top-P% saliency indices.
jaccard              — Jaccard index between two binary masks.
compute_subject_jaccards — per-trial Jaccards for one subject.
group_stats          — Wilcoxon + permutation test on a list of overlaps.
permutation_test     — two-sided test: H0 = mean overlaps == 0.
permutation_test_pairwise — one-sided test: H0 = real <= random.
save_jaccards        — persist per-subject scores as .npy.
load_jaccards        — load all *_jac.npy files from a directory.
random_baseline      — generate random mask Jaccard scores.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import wilcoxon


# ── Saliency thresholding ─────────────────────────────────────────────────────

def threshold_top_p(heatmap: np.ndarray, p: float = 0.15) -> np.ndarray:
    """Binary mask of the top-P fraction of salient time points.

    Args:
        heatmap: 1-D saliency map of shape (T,).
        p:       Fraction to retain (default 0.15 — top 15%).

    Returns:
        Boolean array of shape (T,); True at retained positions.
    """
    k = max(1, int(np.ceil(p * len(heatmap))))
    mask = np.zeros(len(heatmap), dtype=bool)
    mask[np.argpartition(heatmap, -k)[-k:]] = True
    return mask


def jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Jaccard index between two binary masks.

    Args:
        mask_a: Boolean array of shape (T,).
        mask_b: Boolean array of shape (T,).

    Returns:
        Jaccard similarity in [0, 1].
    """
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / (float(union) + 1e-10)


# ── Per-subject Jaccard computation ──────────────────────────────────────────

def compute_subject_jaccards(
    heatmaps_a: np.ndarray,
    heatmaps_b: np.ndarray,
    top_p:      float = 0.15,
) -> np.ndarray:
    """Compute per-trial Jaccard scores between two sets of saliency maps.

    Both arrays must correspond to the same set of EEG trials, computed
    from models trained on two different tasks (e.g. B4 and MC).

    Args:
        heatmaps_a: Array of shape (N, T) — saliency maps from task A.
        heatmaps_b: Array of shape (N, T) — saliency maps from task B.
        top_p:      Threshold fraction (default 0.15).

    Returns:
        Array of shape (N,) with per-trial Jaccard scores.
    """
    assert heatmaps_a.shape == heatmaps_b.shape, (
        "heatmaps_a and heatmaps_b must have the same shape."
    )
    scores = np.array([
        jaccard(
            threshold_top_p(heatmaps_a[i], top_p),
            threshold_top_p(heatmaps_b[i], top_p),
        )
        for i in range(len(heatmaps_a))
    ])
    return scores


# ── Statistical tests ─────────────────────────────────────────────────────────

def permutation_test(values: np.ndarray, n_perm: int = 1000) -> float:
    """Two-sided permutation test: H0 = mean(values) == 0.

    Args:
        values: Per-subject overlap scores (e.g. mean Jaccard per subject).
        n_perm: Number of random sign permutations (default 1000).

    Returns:
        p-value.
    """
    values = np.asarray(values)
    orig   = np.abs(np.mean(values))
    count  = sum(
        np.abs(np.mean(values * np.random.choice([-1, 1], size=len(values))))
        >= orig
        for _ in range(n_perm)
    )
    return (count + 1) / (n_perm + 1)


def permutation_test_pairwise(
    x:      np.ndarray,
    y:      np.ndarray,
    n_perm: int = 10_000,
) -> float:
    """One-sided paired permutation test: H0 = mean(x - y) <= 0.

    Args:
        x:      Real overlap scores (per subject).
        y:      Baseline/random scores (per subject).
        n_perm: Number of random sign permutations (default 10 000).

    Returns:
        p-value (one-sided, right tail).
    """
    x, y     = np.asarray(x), np.asarray(y)
    observed = np.mean(x - y)
    diffs    = [
        np.mean((x - y) * np.random.choice([-1, 1], size=len(x)))
        for _ in range(n_perm)
    ]
    return float(np.mean(np.array(diffs) >= observed))


def group_stats(
    overlaps: List[float],
) -> Tuple[float, float]:
    """Wilcoxon signed-rank and permutation test on per-subject overlaps.

    Args:
        overlaps: Per-subject mean Jaccard scores.

    Returns:
        (p_wilcoxon, p_permutation)
    """
    arr = np.array(overlaps)
    p_w    = float(wilcoxon(arr).pvalue)
    p_perm = permutation_test(arr)
    return p_w, p_perm


# ── Persistence helpers ───────────────────────────────────────────────────────

def save_jaccards(
    subject:    str,
    scores:     np.ndarray,
    output_dir: str,
    suffix:     str = "jac",
) -> None:
    """Save per-trial Jaccard scores for one subject as a .npy file.

    Args:
        subject:    Subject identifier.
        scores:     Array of shape (N,) with per-trial scores.
        output_dir: Destination directory.
        suffix:     File suffix (default 'jac').
    """
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{subject}_{suffix}.npy"), scores)


def load_jaccards(data_dir: str, suffix: str = "jac") -> Dict[str, np.ndarray]:
    """Load all *_{suffix}.npy files from *data_dir*.

    Args:
        data_dir: Directory containing saved .npy files.
        suffix:   File suffix to match (default 'jac').

    Returns:
        Dict mapping subject identifier to score array.
    """
    result: Dict[str, np.ndarray] = {}
    for fname in os.listdir(data_dir):
        if fname.endswith(f"_{suffix}.npy"):
            subj = fname.replace(f"_{suffix}.npy", "")
            result[subj] = np.load(os.path.join(data_dir, fname))
    return result


# ── Random baseline ───────────────────────────────────────────────────────────

def random_baseline(
    n_trials:   int,
    signal_len: int = 1280,
    top_p:      float = 0.15,
    seed:       Optional[int] = None,
) -> np.ndarray:
    """Jaccard scores between pairs of randomly generated binary masks.

    Used in Phase 3 to quantify the chance-level overlap against which
    real Jaccard scores are compared.

    Args:
        n_trials:   Number of trial-level random pairs to generate.
        signal_len: Length T of each mask (default 1280 = 5s × 256Hz).
        top_p:      Fraction of True positions per mask (default 0.15).
        seed:       Optional RNG seed for reproducibility.

    Returns:
        Array of shape (n_trials,) with random Jaccard scores.
    """
    rng   = np.random.default_rng(seed)
    k     = max(1, int(np.ceil(top_p * signal_len)))
    scores = []
    for _ in range(n_trials):
        m1 = np.zeros(signal_len, dtype=bool)
        m2 = np.zeros(signal_len, dtype=bool)
        m1[rng.choice(signal_len, k, replace=False)] = True
        m2[rng.choice(signal_len, k, replace=False)] = True
        scores.append(jaccard(m1, m2))
    return np.array(scores)