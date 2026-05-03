# MS-EEGNet

**A multi-scale frequency-aware deep learning framework for phonetic similarity analysis in imagined speech EEG.**

MS-EEGNet adapts the EEGNet backbone to operate on wavelet-decomposed EEG representations via depthwise convolution along the frequency axis. Signals are first decomposed into six frequency bands using frozen db4 wavelet filters, then processed by MS-EEGNet with sub-centre ArcFace loss to handle intra-class variability. Smooth Grad-CAM is applied to models trained on the B4 and MC tasks, and cross-task Jaccard similarity analysis tests whether phonologically similar words share temporally overlapping neural activation patterns.

## Repository Structure

```bash
MS-EEGNet/
├── Code/
│   ├── core_dataset.py      # channel stripping, channel-wise expansion, train/val/test split
│   ├── core_loaders.py      # KARAOne loader (ICA + downsample)
│   ├── core_model.py        # WaveletDecomp + MSEEGNet architecture
│   ├── core_loss.py         # Sub-centre ArcFace loss (K=2, m=0.3, s=30)
│   ├── core_train.py        # Training loop with external optimizer/criterion
│   ├── core_gradcam.py      # GradCAM + Smooth Grad-CAM + GradCAMPlotter
│   ├── core_utils.py        # Reproducibility, timing, metadata
│   │
│   ├── main_karaone.py      # Phase-1: MS-EEGNet training — KARAOne
│   ├── main_driver.py       # Entry point for Phase-1
│   │
│   ├── int_analysis.py      # Jaccard overlap, group_stats, permutation tests
│   ├── int_karaone.py       # Phase-2: cross-task Grad-CAM + Jaccard
│   ├── int_driver.py        # Entry point for Phase-2/3
│   │
│   └── int_posthoc.py       # Phase-3 summary stats
│
├── CITATION.cff
├── requirements.txt
└── README.md
```

---

## Datasets

**KARAOne** — five binary tasks (B1–B5) and an 11-class multi-class (MC) task. This chapter focuses on B4 (all /iy/-phoneme words grouped) and MC (individual word labels) for cross-task saliency analysis. Preprocessing: ICA artefact removal, downsample to 256 Hz. [KARAOne Dataset](https://doi.org/10.3389/fnins.2015.00090)

---

## Installation

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

---

## Usage

All driver scripts are run from inside the `Code/` directory.

### Phase 1 — Training

Set `DATA_PATH`, `MODEL_ROOT`, and `RESULTS_ROOT` in `main_karaone.py`, then:

```bash
cd Code
python main_driver.py
```

`RUN_CFG["tasks"]` in `main_driver.py` controls which tasks are trained. Default: `["B4", "MC"]`.

### Phase 2 / 3 — Saliency analysis

Set `DATA_PATH`, `MODEL_ROOT`, and `RESULTS_ROOT` in `int_karaone.py` and `int_driver.py`, then:

```bash
python int_driver.py
```

Phase 2 generates per-subject `.npy` Jaccard files under `RESULTS_ROOT/Jaccard_Sims/`. Phase 3 runs the Wilcoxon and permutation tests and writes `phase3_report.txt`.

---

## Configuration

Each pipeline script contains self-contained `DATA_CFG`, `MODEL_CFG`, `LOSS_CFG`, and `TRAIN_CFG` dicts. No external config files are required.

Key hyperparameters (defaults match the published study):

| Parameter | Value |
| --- | --- |
| Optimiser | Adam |
| Learning rate | 1 × 10⁻⁴ |
| Batch size | 32 |
| Max epochs | 50 |
| Early stopping patience | 5 |
| Target loss gate | 0.3 |
| Wavelet decomposition levels | 5 (F = 6 bands) |
| Sub-centre ArcFace K | 2 |
| ArcFace margin m | 0.3 rad |
| ArcFace scale s | 30 |
| Smooth Grad-CAM samples V | 25 |
| Noise scale η | 0.1 |
| Jaccard threshold | top 15% |

---

## Citation

```bibtex
@inproceedings{mseegnet_premi2025,
  author    = {Arun Balasubramanian and Santhoshkumar Peddi and Debasis Samanta},
  title     = {Understanding the Phonetic Similarity in Imagined Speech using Cross-task Saliency Mapping},
  booktitle = {International Conference on Pattern Recognition and Machine Intelligence (PReMI'25)},
  series    = {Lecture Notes in Computer Science},
  volume    = {16357},
  pages     = {582--590},
  publisher = {Springer},
  year      = {2026},
}
```
