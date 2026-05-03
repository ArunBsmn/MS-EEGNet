"""Entry point for MS-EEGNet interpretation pipeline (int_driver.py).

Phases:
    2 — Cross-task Grad-CAM + Jaccard overlap (int_karaone.run).
    3 — Statistical tests vs. random baseline.
    4 — Visual heatmap figures for selected subjects.

Run Phase 2 first; Phases 3 and 4 depend on its .npy outputs.

Run from inside the Code/ directory:
    python int_driver.py
"""
from __future__ import annotations

import os

import numpy as np

import int_karaone
from int_analysis import (
    group_stats,
    load_jaccards,
    permutation_test_pairwise,
    random_baseline,
    save_jaccards,
)

# ── Paths — must match main_karaone.py ────────────────────────────────────────

RESULTS_ROOT = "/path/to/results/karaone"

# ── Run configuration ─────────────────────────────────────────────────────────

RUN_CFG = {
    "run_phase2": True,
    "run_phase3": True,
    "signal_len": 1280,   # 5 s × 256 Hz
    "top_p":      0.15,
    "n_perm":     10_000,
}

# ── Phase 3 — statistical report ─────────────────────────────────────────────

def _phase3(jac_dir: str, results_dir: str) -> None:
    real_jaccards = load_jaccards(jac_dir)
    if not real_jaccards:
        print("Phase 3: no Jaccard files found — run Phase 2 first.")
        return

    true_means: list[float] = []
    rand_means: list[float] = []

    rand_dir = os.path.join(results_dir, "Jaccard_Sims_Random")
    os.makedirs(rand_dir, exist_ok=True)

    for subj, scores in real_jaccards.items():
        if len(scores) == 0:
            continue
        rand = random_baseline(len(scores), RUN_CFG["signal_len"], RUN_CFG["top_p"])
        true_means.append(float(scores.mean()))
        rand_means.append(float(rand.mean()))
        save_jaccards(subj, rand, rand_dir, suffix="rand_jac")

    if not true_means:
        print("Phase 3: no valid subjects.")
        return

    p_real_w, p_real_perm = group_stats(true_means)
    p_base_perm = permutation_test_pairwise(
        np.array(true_means), np.array(rand_means), RUN_CFG["n_perm"]
    )
    from scipy.stats import wilcoxon
    p_base_w = float(wilcoxon(true_means, rand_means).pvalue)

    report_path = os.path.join(results_dir, "phase3_report.txt")
    lines = [
        "=== Statistical Report: Phase 2 and Phase 3 ===\n",
        "PHASE 2 — Real Overlap Significance",
        "H\u2080: overlaps are due to chance (mean = 0)",
        f"\u2192 Wilcoxon p-value    : {p_real_w:.4e}",
        f"\u2192 Permutation p-value : {p_real_perm:.4e}",
        "",
        "PHASE 3 — Real vs. Random Baseline",
        "H\u2080: real overlaps \u2264 random baseline",
        f"\u2192 Wilcoxon p-value    : {p_base_w:.4e}",
        f"\u2192 Permutation p-value : {p_base_perm:.4e}",
    ]
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Phase 3 report saved to {report_path}")
    for line in lines:
        print(line)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    jac_dir = os.path.join(RESULTS_ROOT, "Jaccard_Sims")

    if RUN_CFG["run_phase2"]:
        print("\n" + "=" * 60)
        print("  Phase 2 — Cross-task Grad-CAM + Jaccard overlap")
        print("=" * 60)
        int_karaone.run()

    if RUN_CFG["run_phase3"]:
        print("\n" + "=" * 60)
        print("  Phase 3 — Statistical analysis vs. random baseline")
        print("=" * 60)
        _phase3(jac_dir, RESULTS_ROOT)