"""MS-EEGNet training pipeline — KARAOne dataset (Phase 1).

Configure DATA_PATH, MODEL_ROOT, and RESULTS_ROOT below, then run via
main_driver.py.  A separate model is trained for each (subject, task)
pair in subject-dependent mode.  Tasks B1--B5 run binary classification;
MC runs 11-class classification.

The subject-model index mapping is persisted to MODEL_ROOT so that
int_karaone.py can reload the correct weights for each subject.
"""
from __future__ import annotations

import gc
import os
from typing import Any, Dict, List

import torch
import torch.optim as optim

from core_dataset import make_channelwise, make_loaders, preprocess_trials
from core_loaders  import load_karaone
from core_loss     import SubCentreArcFaceLoss
from core_model    import MSEEGNet
from core_train    import train_model
from core_utils    import Timer, save_metadata, save_model_summary, set_all_seeds

# ── Paths — edit before running ───────────────────────────────────────────────

DATA_PATH    = "/path/to/KARAOne"           # root with Data/, Rest/, Targets/
MODEL_ROOT   = "/path/to/weights/karaone"
RESULTS_ROOT = "/path/to/results/karaone"

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_CFG: Dict[str, Any] = {
    "subjects": [
        "MM05", "MM08", "MM09", "MM10", "MM11",
        "MM12", "MM14", "MM15", "MM16", "MM18",
        "MM19", "MM20", "MM21",
    ],
    "td":   5,      # trial duration (seconds)
    "sfreq": 256,   # sampling frequency (Hz)
    # task is set by main_driver.py RUN_CFG
}

MODEL_CFG: Dict[str, Any] = {
    "wd_levels":      5,
    "F1":             8,
    "D":              2,
    "F2":             None,
    "kern_length":    64,
    "dropout":        0.5,
    "projection_dim": 128,
}

LOSS_CFG: Dict[str, Any] = {
    "num_sub_centres": 2,
    "scale":           30.0,
    "margin":          0.3,
}

TRAIN_CFG: Dict[str, Any] = {
    "batch_size":   32,
    "lr":           1e-4,
    "num_epochs":   50,
    "patience":     5,
    "min_delta":    1e-3,
    "target_loss":  0.3,
    "seed":         37,
}

# ── Binary task label encoding ────────────────────────────────────────────────

_BINARY_MAP: Dict[int, List[int]] = {
    0:  [1, 0, 0, 1, 0], 1:  [0, 0, 0, 1, 0],
    2:  [1, 1, 1, 0, 0], 3:  [1, 1, 0, 0, 0],
    4:  [1, 0, 1, 1, 0], 5:  [1, 0, 0, 1, 0],
    6:  [0, 0, 0, 0, 1], 7:  [1, 1, 0, 0, 0],
    8:  [1, 1, 0, 0, 0], 9:  [1, 0, 1, 0, 0],
    10: [1, 0, 1, 0, 0],
}
_TASK_IDX: Dict[str, int] = {"B1": 0, "B2": 1, "B3": 2, "B4": 3, "B5": 4}


def _encode_task(targets, task: str):
    import numpy as np
    col = _TASK_IDX[task]
    return np.array([_BINARY_MAP[int(t)][col] for t in targets])


# ── Subject-model mapping persistence ─────────────────────────────────────────

def _load_mapping(path: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if ":" in line:
                    s, idx = line.strip().split(":", 1)
                    mapping[s.strip()] = int(idx.strip())
    return mapping


def _save_mapping(mapping: Dict[str, int], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for s, idx in mapping.items():
            f.write(f"{s}: {idx}\n")


# ── Main training function ────────────────────────────────────────────────────

def run(task: str = "MC") -> None:
    """Train MS-EEGNet on KARAOne for the specified task.

    Args:
        task: One of 'B1'–'B5' or 'MC'.
    """
    import numpy as np

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed     = TRAIN_CFG["seed"]
    set_all_seeds(seed)

    is_binary = task in _TASK_IDX
    n_classes = 2 if is_binary else 11
    data_len  = DATA_CFG["td"] * DATA_CFG["sfreq"]  # 1280

    task_dir  = f"SubjDep_{task}" if is_binary else "SubjDep"
    model_dir = os.path.join(MODEL_ROOT,   task_dir, "MSEEGNet")
    res_dir   = os.path.join(RESULTS_ROOT, task_dir, "MSEEGNet")
    map_path  = os.path.join(model_dir, "subject_model_mapping.txt")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(res_dir, "Metrics"), exist_ok=True)

    mapping     = _load_mapping(map_path)
    model_count = max(mapping.values(), default=0)

    subjects = DATA_CFG["subjects"]
    np.random.shuffle(subjects)

    for subj in subjects:
        print(f"\n── {task} | {subj} ──")
        with Timer() as t:
            data_flat, targets = load_karaone(subj, DATA_PATH, DATA_CFG["td"])
            data, tgt = preprocess_trials(data_flat, targets)

            if is_binary:
                tgt = _encode_task(tgt, task)

            windows, labels = make_channelwise(data, tgt)
            tr_l, vl_l, ts_l = make_loaders(
                windows, labels,
                batch_size = TRAIN_CFG["batch_size"],
                seed       = seed,
            )

            model     = MSEEGNet(n_classes, data_len, config=MODEL_CFG)
            criterion = SubCentreArcFaceLoss(
                n_classes, MODEL_CFG["projection_dim"], **LOSS_CFG
            ).to(device)
            optimizer = optim.Adam(
                list(model.parameters()) + list(criterion.parameters()),
                lr=TRAIN_CFG["lr"],
            )

            model_count += 1
            model_path  = os.path.join(model_dir, f"model_{model_count}.pth")
            metrics_dir = os.path.join(res_dir, "Metrics")

            train_model(
                model         = model,
                train_loader  = tr_l,
                val_loader    = vl_l,
                test_loader   = ts_l,
                optimizer     = optimizer,
                criterion     = criterion,
                num_epochs    = TRAIN_CFG["num_epochs"],
                pass_features = True,
                device        = device,
                subject       = subj,
                dataset       = f"k1_{task}",
                patience      = TRAIN_CFG["patience"],
                min_delta     = TRAIN_CFG["min_delta"],
                target_loss   = TRAIN_CFG["target_loss"],
                model_path    = model_path,
                metrics_path  = metrics_dir,
            )

            mapping[subj] = model_count
            _save_mapping(mapping, map_path)

            save_metadata({
                "subject": subj, "task": task, "n_classes": n_classes,
                "model_idx": model_count, "elapsed": t.elapsed_str,
                **{f"model_{k}": v for k, v in MODEL_CFG.items()},
                **{f"train_{k}": v for k, v in TRAIN_CFG.items()},
            }, os.path.join(model_dir, f"meta_{subj}.txt"))

        print(f"Done in {t.elapsed_str}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()