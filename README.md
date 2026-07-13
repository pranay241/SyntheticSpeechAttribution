# Synthetic Speech Attribution

Identify which text-to-speech (TTS) system generated a given synthetic speech
sample, using acoustic feature engineering and classical machine learning
classifiers.

## Problem

Given a `.wav` file of synthetic (TTS-generated) speech, predict which of
**25 TTS systems / generators** produced it (e.g. `ljspeech_vits`,
`ljspeech_fastspeech2`, `ljspeech_glow-tts`, `cv4_fastspeech2`, and others).
This is a multi-class audio classification / model-attribution problem —
useful for detecting and tracing the origin of AI-generated speech.

## Approach

Two parallel approaches were built and compared for the same attribution task:

```
                          Raw audio (.wav)
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                     ▼
  Acoustic feature extraction              Log-mel spectrogram (.npy)
       (MFCC, deltas, spectral stats)                │
              │                                     ▼
              ▼                          CNN (2× conv/pool → FC → FC)
  Feature table (.parquet / .csv)                    │
              │                                     │
              ▼                                     │
  Random Forest / XGBoost                            │
              │                                     │
              └─────────────────┬─────────────────┘
                                ▼
                Evaluation (accuracy, classification
                  report, confusion matrix)
```

### 1. Data extraction (`notebooks/` — data extraction notebook)

- Loads `metadata.csv`, which maps each audio sample to its `exp_name`
  (TTS system), `speaker_id`, and file path.
- Explores dataset composition: number of samples, unique TTS systems,
  unique speakers, and samples per system.
- Visualizes waveforms, STFT spectrograms, and log-mel spectrograms for
  sample clips from different TTS systems.
- Extracts a fixed-length acoustic feature vector per audio file:
  - **MFCC** (13 coefficients) — mean & std
  - **Delta** and **delta-delta** (first/second derivatives of MFCC) — mean & std
  - **Zero-crossing rate** — mean & std
  - **Spectral centroid** — mean & std
  - **Spectral bandwidth** — mean & std
  - **Spectral rolloff** — mean & std
  - **RMS energy** — mean & std

  → 88-dimensional feature vector per sample.
- Runs extraction across the full dataset with a resumable loop (`tqdm`
  progress bar, skips/logs files that fail to load).
- Saves the resulting feature table to `tts_features.parquet` (fast,
  compact) and `tts_features.csv` (human-readable) for downstream use.

  <img width="907" height="390" alt="image" src="https://github.com/user-attachments/assets/d4510889-0b22-48b0-b1c4-adcc6dc0c956" />


### 2. Classification (`notebooks/` — classical ML notebook)

- Loads the extracted feature table (`tts_features.parquet`).
- Splits into train/test sets (80/20, stratified by TTS system label).
- Trains and evaluates two classifiers:
  - **Random Forest** (200 trees) — also used to rank feature importance.
  - **XGBoost** (`multi:softmax`, 200 estimators, max depth 8, learning
    rate 0.1) over label-encoded classes.
- Reports accuracy, per-class precision/recall/F1 (classification report),
  and confusion matrix for both models.

### 3. Deep learning classifier (`notebooks/` — CNN notebook)

A second approach to the same attribution task, using log-mel spectrograms
directly as input to a CNN instead of hand-engineered acoustic features.

- **Input**: precomputed log-mel spectrograms (`.npy`, 128×128) per audio
  sample, indexed by `metadata.csv` + `label_map.json`.
- **Data handling**: dataset is copied from Google Drive to local Colab
  SSD once (multi-threaded copy, 32 workers) to remove Drive/FUSE I/O
  latency from every epoch; known-corrupted files are filtered out first.
- **Model**: a compact CNN —
  - Conv(1→32, 3×3) → ReLU → MaxPool
  - Conv(32→64, 3×3) → ReLU → MaxPool
  - Flatten → FC(→256) → ReLU → FC(→`num_classes`)
- **Training**: Adam optimizer, cross-entropy loss, batch size 64, up to
  20 epochs, with:
  - `ReduceLROnPlateau` LR scheduling on validation loss
  - Early stopping (patience 5, min delta 1e-4)
  - Best checkpoint saved by validation accuracy
- **Evaluation**: classification report (precision/recall/F1 per class)
  and confusion matrix on the held-out validation split; training/
  validation loss, accuracy, and LR curves plotted per epoch.
- **Artifacts**: best/final model weights, label encoder, training
  history, training curves, classification report, and confusion matrix
  are all saved and backed up to Drive (`results/logmel_dataset_outputs/`).
- **Inference**: a standalone `inference.py` script loads the trained
  model + label encoder and predicts the source TTS system for a new
  log-mel spectrogram file.

## Repository structure

```
data/        Raw dataset / metadata
notebooks/   Data extraction + classical ML notebooks
report/      Write-up of methodology and results
results/     Saved outputs (e.g. log-mel dataset outputs)
```

## Tech stack

Python, librosa, pandas, NumPy, scikit-learn, XGBoost, Matplotlib

## Possible extensions

- Deeper/pretrained CNN backbones or attention-based architectures over
  the spectrograms.
- Cross-dataset generalization: train on one set of TTS systems, test on
  unseen ones.
- Feature ablation to isolate which acoustic cues (classical) or spectral
  regions (CNN) are most attributable to each vocoder/TTS architecture.
- Ensembling the classical (Random Forest/XGBoost) and CNN predictions.

<img width="1500" height="1500" alt="confusion_matrix" src="https://github.com/user-attachments/assets/9249cd5a-3855-4c17-b638-526252fda533" />

<img width="2700" height="750" alt="training_curves" src="https://github.com/user-attachments/assets/9e68c533-2dd5-4892-a042-9feb2fa64952" />


