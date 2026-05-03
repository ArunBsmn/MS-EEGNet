"""Post-hoc statistical tests and visualisation for the LSTM-CNN framework.

Loads per-subject test metrics saved by the training pipeline and
applies the following statistical procedures described in Chapter 4:

* Friedman test across model variants (LSTMCNN, CNNLSTM, LSTMOnly,
  CNNOnly) to detect a significant main effect of architecture.
* Nemenyi post-hoc test for pairwise comparison when Friedman is
  significant.
* Wilcoxon signed-rank test for pairwise comparisons between the
  proposed LSTMCNN and each alternative variant.
* Box-plot generation for accuracy distributions per variant.

Public API
----------
run   — aggregate metrics, run tests, and save results.
"""
from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ── Friedman test ─────────────────────────────────────────────────────────────

def friedman_test(
    data: np.ndarray,
) -> Tuple[float, float]:
    """Friedman test for a repeated-measures design.

    Args:
        data: Array of shape ``(n_subjects, n_treatments)`` where each
              row is one subject's accuracy across all model variants.

    Returns:
        Tuple ``(statistic, p_value)``.
    """
    result = stats.friedmanchisquare(*[data[:, j] for j in range(data.shape[1])])
    return float(result.statistic), float(result.pvalue)


# ── Nemenyi post-hoc test ─────────────────────────────────────────────────────

def nemenyi_test(
    data: np.ndarray,
) -> np.ndarray:
    """Nemenyi post-hoc test via scikit-posthocs.

    Args:
        data: Array of shape ``(n_subjects, n_treatments)``.

    Returns:
        Symmetric p-value matrix of shape ``(n_treatments, n_treatments)``.
    """
    try:
        import scikit_posthocs as sp  # type: ignore[import]
        df = pd.DataFrame(data)
        return sp.posthoc_nemenyi_friedman(df).to_numpy()
    except ImportError:
        warnings.warn(
            "scikit-posthocs is not installed; Nemenyi test skipped.  "
            "Install with: pip install scikit-posthocs"
        )
        return np.full((data.shape[1], data.shape[1]), np.nan)


# ── Wilcoxon signed-rank test ─────────────────────────────────────────────────

def wilcoxon_vs_proposed(
    data:            np.ndarray,
    proposed_index:  int = 0,
    alpha:           float = 0.05,
) -> List[Dict[str, Any]]:
    """Wilcoxon signed-rank test: proposed vs. each other variant.

    Args:
        data:           Array of shape ``(n_subjects, n_treatments)``.
        proposed_index: Column index of the proposed method (LSTMCNN).
        alpha:          Significance level.

    Returns:
        List of dicts with keys ``variant_index``, ``statistic``,
        ``p_value``, and ``significant``.
    """
    results = []
    proposed = data[:, proposed_index]
    for j in range(data.shape[1]):
        if j == proposed_index:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p = stats.wilcoxon(proposed, data[:, j])
        results.append({
            "variant_index": j,
            "statistic":     float(stat),
            "p_value":       float(p),
            "significant":   bool(p < alpha),
        })
    return results


# ── Box-plot generation ───────────────────────────────────────────────────────

def save_boxplot(
    data:       np.ndarray,
    labels:     List[str],
    title:      str,
    out_path:   str,
) -> None:
    """Save a box-plot of accuracy distributions to *out_path*.

    Args:
        data:     Array of shape ``(n_subjects, n_variants)``.
        labels:   Variant names for the x-axis.
        title:    Plot title.
        out_path: Destination SVG path.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available; box-plot skipped.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(
        [data[:, j] for j in range(data.shape[1])],
        labels    = labels,
        patch_artist = True,
    )
    ax.set_title(title)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Architecture variant")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, format="svg")
    plt.close(fig)


# ── Main entry point ──────────────────────────────────────────────────────────

_VARIANT_LABELS: List[str] = ["LSTMCNN", "CNNLSTM", "LSTMOnly", "CNNOnly"]


def run(cfg: Dict[str, Any] | None = None) -> None:
    """Load per-subject metrics and run all post-hoc procedures.

    Expects CSV files written by core_train under RESULTS_ROOT with
    the naming convention ``<dataset>_<task>/<subject>/test_summary.csv``.

    Args:
        cfg: Optional dict with keys ``dataset`` and ``task``.
    """
    from int_karaone import RESULTS_ROOT as K1_ROOT
    from int_asu     import RESULTS_ROOT as ASU_ROOT

    cfg = cfg or {}
    dataset = cfg.get("dataset", "karaone")
    task    = cfg.get("task",    "MC")

    results_root = K1_ROOT if dataset.lower() == "karaone" else ASU_ROOT
    task_dir     = os.path.join(results_root, task)

    # Collect per-subject per-variant accuracies.
    subject_rows: Dict[str, Dict[str, float]] = {}
    for variant in _VARIANT_LABELS:
        summary_path = os.path.join(task_dir, "all_subjects",
                                    f"{variant}_test_summary.csv")
        if not os.path.exists(summary_path):
            continue
        df = pd.read_csv(summary_path)
        for _, row in df.iterrows():
            subj = str(row["subject"])
            if subj not in subject_rows:
                subject_rows[subj] = {}
            subject_rows[subj][variant] = float(row["accuracy"])

    if not subject_rows:
        print("[int_posthoc] No test summaries found; skipping post-hoc tests.")
        return

    subjects = sorted(subject_rows.keys())
    data     = np.array([
        [subject_rows[s].get(v, np.nan) for v in _VARIANT_LABELS]
        for s in subjects
    ])

    # Drop rows with any NaN (incomplete data).
    mask = ~np.isnan(data).any(axis=1)
    data = data[mask]

    if data.shape[0] < 2:
        print("[int_posthoc] Insufficient complete subjects for tests.")
        return

    out_dir = os.path.join(task_dir, "posthoc")
    os.makedirs(out_dir, exist_ok=True)

    # Friedman test.
    f_stat, f_p = friedman_test(data)
    print(f"[Friedman] χ²={f_stat:.4f}, p={f_p:.4e}")

    # Nemenyi post-hoc.
    nem_mat = nemenyi_test(data)

    # Wilcoxon vs proposed.
    wilcox = wilcoxon_vs_proposed(data, proposed_index=0)

    # Save summary.
    with open(os.path.join(out_dir, "posthoc_results.txt"), "w") as f:
        f.write(f"Friedman χ² = {f_stat:.6f},  p = {f_p:.6e}\n\n")
        f.write("Nemenyi p-value matrix:\n")
        f.write(",".join(_VARIANT_LABELS) + "\n")
        for i, row in enumerate(nem_mat):
            f.write(_VARIANT_LABELS[i] + "," +
                    ",".join(f"{v:.6f}" for v in row) + "\n")
        f.write("\nWilcoxon (LSTMCNN vs. others):\n")
        for res in wilcox:
            j     = res["variant_index"]
            label = _VARIANT_LABELS[j]
            sig   = "*" if res["significant"] else ""
            f.write(f"  {label}: W={res['statistic']:.1f},  "
                    f"p={res['p_value']:.4e}  {sig}\n")

    # Box-plot.
    save_boxplot(
        data,
        labels   = _VARIANT_LABELS,
        title    = f"{dataset.upper()} {task} — Accuracy by Architecture",
        out_path = os.path.join(out_dir, f"boxplot_{dataset}_{task}.svg"),
    )
    print(f"[int_posthoc] Results saved to {out_dir}")
