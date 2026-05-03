"""EEG preprocessing and DataLoader construction for the LSTM-CNN framework.

Two operations are provided:

preprocess_trials
    Strips auxiliary non-EEG channels from raw trial matrices, drops the
    known bad channel (index 28 in the KARAOne montage), and returns a
    channel-wise expanded dataset ready for windowing.

make_channelwise
    Expands a multi-channel trial matrix into individual single-channel
    windows.  Each window is a (1, samples) tensor representing one EEG
    channel from one trial, paired with the original trial label.

make_loaders
    Splits the channel-wise dataset into train / val / test DataLoaders
    using a stratified 70:15:15 partition.

Input convention
----------------
All downstream models (core_model.py) expect tensors of shape
(batch, 1, samples), i.e. a single-channel time series.  The channel
dimension encodes spatial information implicitly through separate
per-channel windows rather than a multi-channel tensor.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

# ── Constants ─────────────────────────────────────────────────────────────────

_K1_AUX_CHANNELS: List[int] = [64, 65, 66, 67]   # VEO, HEO, EKG, EMG indices
_K1_BAD_CHANNEL:  int       = 28                  # M1 (mastoid reference)
_DEFAULT_SFREQ:   int       = 256                 # target sampling frequency (Hz)
_DEFAULT_TD:      int       = 5                   # trial duration (seconds)

# ── Channel stripping ─────────────────────────────────────────────────────────


def preprocess_trials(
    data:        np.ndarray,
    targets:     np.ndarray,
    n_channels:  int = 68,
    aux_indices: List[int] = _K1_AUX_CHANNELS,
    bad_index:   Optional[int] = _K1_BAD_CHANNEL,
    sfreq:       int = _DEFAULT_SFREQ,
    td:          int = _DEFAULT_TD,
) -> Tuple[np.ndarray, np.ndarray]:
    """Strip auxiliary channels and a known bad channel from a flat trial matrix.

    The KARAOne raw matrix has 68 channels including four non-EEG channels
    (VEO, HEO, EKG, EMG at indices 64–67) and one mastoid reference channel
    (M1 at index 28) that is dropped before training.  The ASU dataset has
    no auxiliary channels to strip; pass ``aux_indices=[]`` and
    ``bad_index=None``.

    Args:
        data:        Flat trial matrix of shape
                     ``(n_trials, n_channels × samples_per_trial)``.
        targets:     Integer label array of shape ``(n_trials,)``.
        n_channels:  Total channel count before stripping (default 68).
        aux_indices: Channel indices to remove (non-EEG; default KARAOne).
        bad_index:   Single additional channel to remove after aux stripping,
                     or None.
        sfreq:       Sampling frequency in Hz (default 256).
        td:          Trial duration in seconds (default 5).

    Returns:
        Tuple ``(data_clean, targets)`` where *data_clean* has shape
        ``(n_trials, n_eeg_channels, samples_per_trial)`` with dtype
        float32 and *targets* has shape ``(n_trials,)`` with dtype int64.
    """
    n_trials    = data.shape[0]
    samples_per = int(sfreq * td)
    data_3d     = data.reshape(n_trials, n_channels, samples_per)

    # Build keep-mask excluding auxiliary channels.
    keep = [i for i in range(n_channels) if i not in aux_indices]
    data_3d = data_3d[:, keep, :]

    # Drop the bad channel by its post-strip index.
    if bad_index is not None:
        # Recompute index after aux channels have been removed.
        offset    = sum(1 for a in aux_indices if a < bad_index)
        adj_index = bad_index - offset
        all_ch    = list(range(data_3d.shape[1]))
        all_ch.pop(adj_index)
        data_3d = data_3d[:, all_ch, :]

    return data_3d.astype(np.float32), targets.astype(np.int64)


# ── Channel-wise expansion ────────────────────────────────────────────────────


def make_channelwise(
    data:    np.ndarray,
    targets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand multi-channel trials into individual single-channel windows.

    Each EEG channel from each trial becomes an independent training
    sample labelled with the original trial label.  This channel-wise
    strategy allows the model to learn discriminative features from each
    channel independently, without requiring co-activation across channels.

    Args:
        data:    Array of shape ``(n_trials, n_channels, samples)``.
        targets: Integer label array of shape ``(n_trials,)``.

    Returns:
        Tuple ``(windows, labels)`` where *windows* has shape
        ``(n_trials × n_channels, 1, samples)`` and *labels* has shape
        ``(n_trials × n_channels,)``.
    """
    n_trials, n_channels, samples = data.shape
    windows = data.reshape(n_trials * n_channels, 1, samples)
    labels  = np.repeat(targets, n_channels)
    return windows, labels


# ── DataLoader construction ───────────────────────────────────────────────────


def make_loaders(
    windows:    np.ndarray,
    labels:     np.ndarray,
    batch_size: int   = 64,
    val_frac:   float = 0.15,
    test_frac:  float = 0.15,
    seed:       int   = 37,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Stratified train / val / test split returning PyTorch DataLoaders.

    The split is performed at the channel-window level.  Because windows
    from the same trial appear in all three splits, there is no trial-level
    data leakage by design (subject-dependent paradigm).

    Args:
        windows:    Channel-wise window array of shape ``(N, 1, samples)``.
        labels:     Integer label array of shape ``(N,)``.
        batch_size: Mini-batch size for all three loaders (default 64).
        val_frac:   Fraction of data for validation (default 0.15).
        test_frac:  Fraction of data for test (default 0.15).
        seed:       Random seed for reproducible splits (default 37).

    Returns:
        Tuple ``(train_loader, val_loader, test_loader)``.
    """
    # First split off the test set.
    X_tv, X_test, y_tv, y_test = train_test_split(
        windows, labels,
        test_size    = test_frac,
        stratify     = labels,
        random_state = seed,
    )
    # Then split train/val from the remaining data.
    val_frac_adj = val_frac / (1.0 - test_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size    = val_frac_adj,
        stratify     = y_tv,
        random_state = seed,
    )

    def _make(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).long(),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return (
        _make(X_train, y_train, shuffle=True),
        _make(X_val,   y_val,   shuffle=False),
        _make(X_test,  y_test,  shuffle=False),
    )
