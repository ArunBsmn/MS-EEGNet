"""Interpretation pipeline — KARAOne dataset.

Generates Grad-CAM heatmaps, applies VBTM and VATM masking at the
optimal threshold, and writes per-subject interpretation outputs.
Requires trained weights saved by main_karaone.py.

Configure WEIGHTS_ROOT, RESULTS_ROOT, and DATA_PATH below, then run
via int_driver.py.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from core_dataset  import make_channelwise, make_loaders, preprocess_trials
from core_gradcam  import batch_heatmaps
from core_loaders  import load_karaone
from core_model    import MODEL_REGISTRY
from core_train    import evaluate
from core_utils    import set_all_seeds
from int_signal    import analyse_subject, ttest_iws

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_PATH    = "/path/to/KARAOne"
WEIGHTS_ROOT = "/path/to/results/karaone"
RESULTS_ROOT = "/path/to/results/karaone/interpretation"

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_CFG: Dict[str, Any] = {
    "subjects":  ["MM05", "MM08", "MM09", "MM10", "MM11",
                  "MM12", "MM14", "MM15", "MM16", "MM18",
                  "MM19", "MM20", "MM21", "P02"],
    "td":        5,
    "task":      "MC",
    "load_rest": True,
}

MODEL_CFG: Dict[str, Any] = {
    "arch":          "LSTMCNN",
    "lstm_dims":     [32, 64, 64, 32, 16],
    "cnn_channels":  [32, 64],
    "cnn_kernels":   [3, 3],
    "fc_dims":       [],
    "bidirectional": False,
    "dropout":       0.0,
}

INT_CFG: Dict[str, Any] = {
    "seed":          37,
    "batch_size":    64,
    "sfreq":         256,
    "n_thresh_steps": 100,   # number of threshold steps to sweep
    "arch_variants": ["LSTMCNN", "CNNLSTM", "LSTMOnly", "CNNOnly"],
}

_TASK_CLASSES: Dict[str, List[int]] = {
    "B1": [1, 2],  "B2": [3, 4],  "B3": [5, 6],
    "B4": [7, 8],  "B5": [9, 10],
    "MC": list(range(1, 12)),
}


def _filter_task(data, targets, task):
    classes   = _TASK_CLASSES[task]
    mask      = np.isin(targets, classes)
    data      = data[mask]
    targets   = targets[mask]
    label_map = {c: i for i, c in enumerate(classes)}
    targets   = np.vectorize(label_map.__getitem__)(targets)
    return data, targets


# ── Masking helpers ────────────────────────────────────────────────────────────

def _apply_mask(
    signals:   np.ndarray,
    heatmaps:  np.ndarray,
    threshold: float,
    mode:      str,
) -> np.ndarray:
    """Apply VBTM or VATM masking to signals using heatmap thresholds.

    Args:
        signals:   Raw signals of shape ``(N, samples)``.
        heatmaps:  Grad-CAM maps of shape ``(N, samples)``, values in [0,1].
        threshold: Masking threshold in [0, 1].
        mode:      ``"vbtm"`` masks values below threshold;
                   ``"vatm"`` masks values above threshold.

    Returns:
        Masked signal array of same shape as *signals*.
    """
    masked = signals.copy()
    if mode == "vbtm":
        masked[heatmaps < threshold] = 0.0
    elif mode == "vatm":
        masked[heatmaps >= threshold] = 0.0
    else:
        raise ValueError(f"Unknown masking mode: {mode!r}")
    return masked


# ── Per-subject interpretation ─────────────────────────────────────────────────

def run_subject(
    subject: str,
    task:    str,
    device:  torch.device,
    cfg_override: Dict[str, Any] | None = None,
) -> None:
    dcfg = {**DATA_CFG,  **(cfg_override or {}).get("data",  {})}
    mcfg = {**MODEL_CFG, **(cfg_override or {}).get("model", {})}
    icfg = {**INT_CFG,   **(cfg_override or {}).get("int",   {})}

    set_all_seeds(icfg["seed"])

    # ── Load data ──────────────────────────────────────────────────────────
    raw_think, targets, raw_rest = load_karaone(
        subject   = subject,
        data_path = DATA_PATH,
        td        = dcfg["td"],
        load_rest = True,
    )
    data_clean, targets_clean = preprocess_trials(raw_think, targets,
                                                  td=dcfg["td"])
    rest_clean, _             = preprocess_trials(raw_rest,
                                                  np.zeros(raw_rest.shape[0],
                                                           dtype=np.int64),
                                                  td=dcfg["td"])
    data_task, targets_task   = _filter_task(data_clean, targets_clean, task)
    windows, labels           = make_channelwise(data_task, targets_task)

    n_classes   = len(np.unique(labels))
    data_length = windows.shape[2]

    _, _, test_loader = make_loaders(
        windows, labels,
        batch_size = icfg["batch_size"],
        seed       = icfg["seed"],
    )

    # ── Load trained model ─────────────────────────────────────────────────
    arch       = MODEL_REGISTRY[mcfg["arch"]]
    model      = arch(num_classes=n_classes, data_length=data_length,
                      config=mcfg).to(device)
    model_path = os.path.join(WEIGHTS_ROOT, task, subject, "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ── Signal analysis ────────────────────────────────────────────────────
    band_durs  = analyse_subject(data_task, rest_clean, sfreq=icfg["sfreq"])
    all_durs   = [d for durs in band_durs.values() for d in durs]
    t_stat, p_val = ttest_iws(all_durs, trial_duration=float(dcfg["td"]))
    iws_min    = min(all_durs) if all_durs else 0.0
    iws_max    = max(all_durs) if all_durs else float(dcfg["td"])
    iws_min_s  = int(iws_min * icfg["sfreq"])
    iws_max_s  = int(iws_max * icfg["sfreq"])

    print(f"  [{subject}] IWS range: {iws_min:.3f}–{iws_max:.3f} s  "
          f"(t={t_stat:.2f}, p={p_val:.2e})")

    # ── Grad-CAM heatmaps ──────────────────────────────────────────────────
    heatmaps, signals, gt_labels = batch_heatmaps(model, test_loader, device)

    # ── Threshold sweep (VBTM) ─────────────────────────────────────────────
    thresholds   = np.linspace(0.0, 1.0, icfg["n_thresh_steps"])
    best_thresh  = 0.0
    best_acc     = 0.0
    best_n_samps = data_length

    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()

    for thresh in thresholds:
        masked = _apply_mask(signals, heatmaps, thresh, mode="vbtm")
        n_samps = int((heatmaps >= thresh).sum(axis=1).mean())
        if not (iws_min_s <= n_samps <= iws_max_s):
            continue

        x_masked = torch.from_numpy(masked[:, np.newaxis, :]).float()
        y_masked = torch.from_numpy(gt_labels).long()
        from torch.utils.data import DataLoader, TensorDataset
        masked_loader = DataLoader(
            TensorDataset(x_masked, y_masked),
            batch_size = icfg["batch_size"],
        )
        acc, _ = evaluate(model, masked_loader, criterion, device)
        if acc > best_acc or (acc == best_acc and n_samps < best_n_samps):
            best_acc    = acc
            best_thresh = thresh
            best_n_samps = n_samps

    print(f"  [{subject}] Optimal threshold: {best_thresh:.3f}  "
          f"acc={best_acc * 100:.2f}%  samples={best_n_samps}")

    # ── Output directory ───────────────────────────────────────────────────
    out_dir = os.path.join(RESULTS_ROOT, task, subject)
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "heatmaps.npy"), heatmaps)
    np.save(os.path.join(out_dir, "signals.npy"),  signals)
    np.save(os.path.join(out_dir, "labels.npy"),   gt_labels)

    # Save threshold metadata.
    with open(os.path.join(out_dir, "threshold.txt"), "w") as f:
        f.write(f"optimal_threshold = {best_thresh:.6f}\n")
        f.write(f"vbtm_accuracy     = {best_acc * 100:.4f}\n")
        f.write(f"retained_samples  = {best_n_samps}\n")
        f.write(f"iws_min_samples   = {iws_min_s}\n")
        f.write(f"iws_max_samples   = {iws_max_s}\n")
        f.write(f"t_statistic       = {t_stat:.4f}\n")
        f.write(f"p_value           = {p_val:.6e}\n")

    # ── Cross-component masking evaluation ────────────────────────────────
    results_rows = []
    for variant in icfg["arch_variants"]:
        v_arch = MODEL_REGISTRY[variant]
        v_model = v_arch(num_classes=n_classes, data_length=data_length,
                         config=mcfg).to(device)
        v_path = os.path.join(WEIGHTS_ROOT, task, subject, f"{variant}_best.pth")
        if not os.path.exists(v_path):
            continue
        v_model.load_state_dict(torch.load(v_path, map_location=device))
        v_model.eval()
        for mode in ["vbtm", "vatm"]:
            masked   = _apply_mask(signals, heatmaps, best_thresh, mode)
            x_masked = torch.from_numpy(masked[:, np.newaxis, :]).float()
            y_masked = torch.from_numpy(gt_labels).long()
            from torch.utils.data import DataLoader, TensorDataset
            ml = DataLoader(TensorDataset(x_masked, y_masked),
                            batch_size=icfg["batch_size"])
            acc, _ = evaluate(v_model, ml, criterion, device)
            results_rows.append(f"{variant},{mode},{acc * 100:.4f}")

    if results_rows:
        with open(os.path.join(out_dir, "masking_results.csv"), "w") as f:
            f.write("arch,mode,accuracy\n")
            f.write("\n".join(results_rows) + "\n")


def run(cfg_override: Dict[str, Any] | None = None) -> None:
    dcfg   = {**DATA_CFG, **(cfg_override or {}).get("data", {})}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task   = dcfg["task"]
    for subject in dcfg["subjects"]:
        run_subject(subject, task, device, cfg_override)
