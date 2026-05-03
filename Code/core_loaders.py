"""Dataset loaders for the LSTM-CNN imagined speech framework.

KARAOne pipeline
----------------
Raw CSV files are at 1 kHz with 68 channels (64 EEG + VEO, HEO, EKG, EMG).
For each subject the loader:
  1. Loads the flat trial matrix from disk.
  2. Reshapes to (n_trials, 68, samples_at_1kHz).
  3. Runs trial-wise ICA on all 68 channels, using the four non-EEG
     channels (VEO, HEO, EKG, EMG) as reference signals to identify
     ocular, cardiac, and muscular artefact components.
  4. Downsamples from 1 kHz → 256 Hz with scipy.signal.resample.
  5. Returns a flat matrix of shape (n_trials, 68 × samples_at_256Hz)
     ready for core_dataset.preprocess_trials, which strips auxiliary
     channels and drops channel 28 to yield (n_trials, 63, samples).

ASU pipeline
------------
No preprocessing is applied.  Raw CSVs are expected to already be at
256 Hz.  Resting-state files follow the same layout as thinking-state
files and are loaded identically.

Both loaders optionally return resting-state data for the signal
analysis stage (int_signal.py).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
from mne.preprocessing import ICA
from scipy.signal import resample

mne.set_log_level("ERROR")

# ── KARAOne constants ─────────────────────────────────────────────────────────

_K1_SFREQ_RAW: int = 1_000   # recording frequency (Hz)
_K1_SFREQ_DS:  int = 256     # target frequency after downsampling (Hz)
_K1_N_CHAN:    int = 68       # total channels including non-EEG

# Full 68-channel montage as stored in the raw KARAOne CSV files.
_K1_CH_NAMES: list[str] = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7",
    "F5",  "F3",  "F1",  "FZ",  "F2",  "F4",
    "F6",  "F8",  "FT7", "FC5", "FC3", "FC1",
    "FCZ", "FC2", "FC4", "FC6", "FT8", "T7",
    "C5",  "C3",  "C1",  "CZ",  "C2",  "C4",  "C6",
    "T8",  "M1",  "TP7", "CP5", "CP3", "CP1", "CPZ",
    "CP2", "CP4", "CP6", "TP8", "M2",  "P7",
    "P5",  "P3",  "P1",  "PZ",  "P2",  "P4",  "P6",
    "P8",  "PO7", "PO5", "PO3", "POZ", "PO4",
    "PO6", "PO8", "CB1", "O1",  "OZ",  "O2",
    "CB2", "VEO", "HEO", "EKG", "EMG",
]

# Non-EEG reference channels used to label ICA components.
_K1_EOG_CHANNELS: list[str] = ["HEO", "VEO", "EKG", "EMG"]

# ICA parameters (fixed across all subjects for reproducibility).
_ICA_N_COMPONENTS: int = 15
_ICA_RANDOM_STATE: int = 123
_ICA_METHOD:       str = "fastica"


# ── KARAOne preprocessing helpers ────────────────────────────────────────────

def _ica_clean_trial(trial: np.ndarray) -> np.ndarray:
    """Remove ocular, cardiac, and muscular artefacts from a single trial.

    Operates on all 68 channels so that the four non-EEG channels
    (VEO, HEO, EKG, EMG) are available as reference signals for
    component labelling.  The cleaned data for all 68 channels is
    returned and the non-EEG channels are stripped later by
    core_dataset.preprocess_trials.

    Args:
        trial: Array of shape (68, n_samples_at_1kHz).

    Returns:
        Cleaned array of shape (68, n_samples_at_1kHz).
    """
    info = mne.create_info(
        ch_names = _K1_CH_NAMES,
        sfreq    = _K1_SFREQ_RAW,
        ch_types = ["eeg"] * _K1_N_CHAN,
    )
    raw = mne.io.RawArray(trial, info)
    # High-pass at 1 Hz to remove slow drifts before ICA.
    raw.filter(l_freq=1.0, h_freq=None, filter_length=int(4.9 * _K1_SFREQ_RAW))

    ica = ICA(
        n_components = _ICA_N_COMPONENTS,
        random_state = _ICA_RANDOM_STATE,
        method       = _ICA_METHOD,
        max_iter     = "auto",
    )
    ica.fit(raw, picks="all")

    # Iteratively label and exclude components tied to each reference channel.
    for ref_ch in _K1_EOG_CHANNELS:
        bad_idx, _ = ica.find_bads_eog(raw, ch_name=ref_ch, h_freq=499)
        ica.exclude = bad_idx
        raw = ica.apply(raw.copy())

    return raw.get_data()   # (68, n_samples_at_1kHz)


def _downsample_trials(
    data:         np.ndarray,
    original_freq: int = _K1_SFREQ_RAW,
    target_freq:   int = _K1_SFREQ_DS,
) -> np.ndarray:
    """Downsample a 3-D trial tensor along the time axis.

    Args:
        data:          Array of shape (n_trials, n_channels, n_samples).
        original_freq: Source sampling frequency in Hz.
        target_freq:   Target sampling frequency in Hz.

    Returns:
        Downsampled array of shape (n_trials, n_channels, n_samples_ds).
    """
    if target_freq >= original_freq:
        raise ValueError(
            f"target_freq ({target_freq}) must be less than "
            f"original_freq ({original_freq})."
        )
    n_samples_ds = int(data.shape[2] * target_freq / original_freq)
    return resample(data, n_samples_ds, axis=2)


def _preprocess_karaone(raw_flat: np.ndarray, td: int) -> np.ndarray:
    """Full KARAOne preprocessing pipeline for one subject.

    Steps: reshape → trial-wise ICA → downsample → re-flatten.

    Args:
        raw_flat: Flat trial matrix of shape
                  (n_trials, 68 × (td × _K1_SFREQ_RAW)).
        td:       Trial duration in seconds.

    Returns:
        Flat matrix of shape (n_trials, 68 × (td × _K1_SFREQ_DS)),
        ready for core_dataset.preprocess_trials.
    """
    n_trials    = raw_flat.shape[0]
    raw_samples = int(td * _K1_SFREQ_RAW)
    data        = raw_flat.reshape(n_trials, _K1_N_CHAN, raw_samples)

    cleaned = np.stack(
        [_ica_clean_trial(data[i]) for i in range(n_trials)],
        axis=0,
    )  # (n_trials, 68, raw_samples)

    downsampled = _downsample_trials(cleaned)  # (n_trials, 68, ds_samples)

    ds_samples = downsampled.shape[2]
    return downsampled.reshape(n_trials, _K1_N_CHAN * ds_samples)


# ── KARAOne loader ────────────────────────────────────────────────────────────

def load_karaone(
    subject:   str,
    data_path: str,
    td:        int  = 5,
    load_rest: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load, ICA-clean, and downsample KARAOne data for one subject.

    Directory layout expected under *data_path*::

        Data/    <subject>_Act.csv   — thinking-state trials (flat, 1 kHz)
        Rest/    <subject>_Rst.csv   — resting-state trials (flat, 1 kHz)
        Targets/ <subject>_EVE_n.csv — integer class labels

    Args:
        subject:   Subject identifier string (e.g. 'MM05').
        data_path: Root directory for this dataset split.
        td:        Trial duration in seconds (default 5).
        load_rest: If True, also load and return resting-state data.

    Returns:
        ``(data_think, targets)`` when *load_rest* is False, or
        ``(data_think, targets, data_rest)`` when True.
        *data_think* and *data_rest* have shape
        ``(n_trials, 68 × (td × 256))``.
        *targets* has shape ``(n_trials,)`` with raw integer labels
        (task encoding and channel-wise expansion are handled upstream).
    """
    think_path = os.path.join(data_path, "Data",    f"{subject}_Act.csv")
    tgt_path   = os.path.join(data_path, "Targets", f"{subject}_EVE_n.csv")

    raw_think = np.loadtxt(think_path, delimiter=",")
    targets   = np.loadtxt(tgt_path,   delimiter=",").astype(int)

    data_think = _preprocess_karaone(raw_think, td)

    if not load_rest:
        return data_think, targets

    rest_path = os.path.join(data_path, "Rest", f"{subject}_Rst.csv")
    raw_rest  = np.loadtxt(rest_path, delimiter=",")
    data_rest = _preprocess_karaone(raw_rest, td)

    return data_think, targets, data_rest
