"""Training loop for the MS-EEGNet framework (Chapter 5).

Differences from the Chapter 4 version:
- criterion is passed externally (SubCentreArcFaceLoss).
- optimizer is passed externally (Adam, lr=1e-4, configured in pipeline).
- pass_features=True routes model output through criterion for predictions.
- Early stopping requires val_loss < target_loss before patience counting.
- label_smoothing=0.1 is applied when falling back to CrossEntropyLoss.

Public API
----------
train_model     — full training loop with early stopping and metric logging.
evaluate        — accuracy + loss on any DataLoader (no gradient).
compute_metrics — weighted precision, recall, F1.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ── Metric helpers ────────────────────────────────────────────────────────────

def _weighted_prf(
    targets: np.ndarray,
    preds:   np.ndarray,
) -> Tuple[float, float, float]:
    classes = np.unique(np.concatenate([targets, preds]))
    total   = len(targets)
    wp = wr = wf = 0.0
    for cls in classes:
        tp = int(np.sum((preds == cls) & (targets == cls)))
        fp = int(np.sum((preds == cls) & (targets != cls)))
        fn = int(np.sum((preds != cls) & (targets == cls)))
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        w  = np.sum(targets == cls) / total
        wp += p * w; wr += r * w; wf += f1 * w
    return wp, wr, wf


def _save_epoch_metrics(metrics: Dict[str, List], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = list(metrics.keys())
    write_header = not os.path.exists(path)
    with open(path, "a") as f:
        if write_header:
            f.write(",".join(header) + "\n")
        for row in zip(*metrics.values()):
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")


def _save_test_summary(
    subject: str, loss: float, acc: float,
    p: float, r: float, f1: float, path: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a") as f:
        if write_header:
            f.write("subject,loss,accuracy,precision,recall,f1\n")
        f.write(f"{subject},{loss:.6f},{acc * 100:.4f},{p:.6f},{r:.6f},{f1:.6f}\n")


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate(
    model:         nn.Module,
    loader:        DataLoader,
    criterion:     nn.Module,
    device:        torch.device,
    pass_features: bool = False,
) -> Tuple[float, float]:
    """Accuracy and average loss over a DataLoader without gradient.

    Args:
        model:         Model in eval mode after this call.
        loader:        DataLoader to evaluate.
        criterion:     Loss function accepting (features/logits, targets).
        device:        Compute device.
        pass_features: If True, model output is features fed to criterion
                       for loss and predictions (ArcFace mode).

    Returns:
        (accuracy_fraction, mean_loss_per_batch)
    """
    model.eval()
    correct = total = 0
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y   = x.to(device), y.to(device)
            out    = model(x)
            if pass_features:
                loss, preds = criterion(out, y)
            else:
                logits = out[0] if isinstance(out, tuple) else out
                loss   = criterion(logits, y)
                preds  = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
            total_loss += loss.item()
    return correct / total, total_loss / len(loader)


def compute_metrics(
    model:         nn.Module,
    loader:        DataLoader,
    device:        torch.device,
    criterion:     Optional[nn.Module] = None,
    pass_features: bool = False,
) -> Tuple[float, float, float]:
    """Weighted precision, recall, F1 over a DataLoader.

    Args:
        model:         Trained model.
        loader:        DataLoader to evaluate.
        device:        Compute device.
        criterion:     Required when pass_features=True.
        pass_features: Route model output through criterion for predictions.

    Returns:
        (weighted_precision, weighted_recall, weighted_f1)
    """
    model.eval()
    all_preds:   List[int] = []
    all_targets: List[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            if pass_features:
                _, preds = criterion(out, y.to(device))
            else:
                logits = out[0] if isinstance(out, tuple) else out
                preds  = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y.numpy().tolist())
    return _weighted_prf(np.array(all_targets), np.array(all_preds))


def train_model(
    model:         nn.Module,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    test_loader:   DataLoader,
    optimizer:     torch.optim.Optimizer,
    criterion:     Optional[nn.Module]   = None,
    num_epochs:    int                   = 50,
    pass_features: bool                  = False,
    device:        Optional[torch.device] = None,
    subject:       str                   = "subject",
    dataset:       str                   = "dataset",
    patience:      int                   = 5,
    min_delta:     float                 = 1e-3,
    target_loss:   float                 = 0.3,
    model_path:    Optional[str]         = None,
    metrics_path:  Optional[str]         = None,
) -> dict:
    """Train one subject model with ArcFace loss and early stopping.

    Early stopping counts patience epochs only after val_loss first
    drops below *target_loss*, preventing premature termination.

    Args:
        model:         Model to train (moved to device internally).
        train_loader:  Training DataLoader.
        val_loader:    Validation DataLoader.
        test_loader:   Test DataLoader.
        optimizer:     Pre-constructed optimiser (e.g. Adam, lr=1e-4).
        criterion:     Loss module.  If None, CrossEntropyLoss with
                       label_smoothing=0.1 is used (non-ArcFace mode).
        num_epochs:    Maximum training epochs (default 50).
        pass_features: True for ArcFace mode — model returns embeddings
                       and criterion returns (loss, predictions).
        device:        Compute device (auto-detected if None).
        subject:       Subject identifier for logging and file names.
        dataset:       Dataset name for logging and file names.
        patience:      Early stopping patience in epochs (default 5).
        min_delta:     Minimum val-loss improvement to reset patience.
        target_loss:   Patience counting starts only once val_loss < this.
        model_path:    Path to save best weights as .pth (optional).
        metrics_path:  Directory for per-epoch CSV + test summary (optional).

    Returns:
        Best model state_dict.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    if criterion is None:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        pass_features = False

    if hasattr(criterion, "sub_centres"):
        criterion = criterion.to(device)

    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0
    target_reached = False

    epoch_metrics: Dict[str, List[float]] = {
        k: [] for k in [
            "train_loss", "val_loss",  "test_loss",
            "train_acc",  "val_acc",   "test_acc",
            "train_prec", "val_prec",  "test_prec",
            "train_rec",  "val_rec",   "test_rec",
            "train_f1",   "val_f1",    "test_f1",
        ]
    }

    for epoch in range(num_epochs):
        # ── Training ─────────────────────────────────────────────────────
        model.train()
        correct = total = 0
        running_loss = 0.0

        for x, y in train_loader:
            x, y  = x.to(device), y.to(device)
            out   = model(x)
            if pass_features:
                loss, preds = criterion(out, y)
            else:
                logits = out[0] if isinstance(out, tuple) else out
                loss   = criterion(logits, y)
                preds  = logits.argmax(dim=1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (preds == y).sum().item()
            total   += y.size(0)
            running_loss += loss.item()

        tr_acc  = correct / total
        tr_loss = running_loss / len(train_loader)
        tr_p, tr_r, tr_f1 = compute_metrics(
            model, train_loader, device, criterion, pass_features)

        # ── Validation / test ─────────────────────────────────────────────
        vl_acc, vl_loss = evaluate(model, val_loader,  criterion, device, pass_features)
        ts_acc, ts_loss = evaluate(model, test_loader, criterion, device, pass_features)
        vl_p, vl_r, vl_f1 = compute_metrics(
            model, val_loader,  device, criterion, pass_features)
        ts_p, ts_r, ts_f1 = compute_metrics(
            model, test_loader, device, criterion, pass_features)

        print(
            f"[{dataset}|{subject}] Epoch {epoch:03d}  "
            f"Train {tr_acc * 100:.2f}%/{tr_loss:.4f}  "
            f"Val {vl_acc * 100:.2f}%/{vl_loss:.4f}  "
            f"Test {ts_acc * 100:.2f}%/{ts_loss:.4f}"
        )

        for key, val in zip(epoch_metrics, [
            tr_loss, vl_loss, ts_loss,
            tr_acc,  vl_acc,  ts_acc,
            tr_p, vl_p, ts_p,
            tr_r, vl_r, ts_r,
            tr_f1, vl_f1, ts_f1,
        ]):
            epoch_metrics[key].append(val)

        # ── Early stopping ────────────────────────────────────────────────
        if vl_loss < target_loss:
            target_reached = True

        if vl_loss < best_val_loss - min_delta:
            best_val_loss  = vl_loss
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        elif target_reached:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if model_path is not None:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(best_state, model_path)

    if metrics_path is not None:
        _save_epoch_metrics(
            epoch_metrics,
            os.path.join(metrics_path, f"{dataset}_{subject}_train.csv"),
        )
        final_acc, final_loss = evaluate(
            model, test_loader, criterion, device, pass_features)
        final_p, final_r, final_f1 = compute_metrics(
            model, test_loader, device, criterion, pass_features)
        _save_test_summary(
            subject, final_loss, final_acc,
            final_p, final_r, final_f1,
            os.path.join(metrics_path, "test_summary.csv"),
        )

    return best_state