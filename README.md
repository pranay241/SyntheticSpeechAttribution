# Synthetic Speech Attribution

**Identifying the source generator of AI-synthesized speech using acoustic feature engineering and deep learning.**

A multi-class audio classification system that attributes a synthetic speech sample to 1 of 25 text-to-speech (TTS) systems — a model-attribution problem relevant to detecting and tracing the origin of AI-generated audio.

---

## Key Features

- 🎯 **25-way TTS system classification** — attributes speech to specific generators (e.g. `ljspeech_vits`, `ljspeech_fastspeech2`, `ljspeech_glow-tts`, `cv4_fastspeech2`).
- 🧪 **Two complementary approaches, both fully implemented** — hand-engineered acoustic features with classical ML, and a CNN trained directly on spectrograms.
- 📊 **Rigorous evaluation** — per-class precision/recall/F1, confusion matrices, and training curves for every model.
- ⚙️ **Production-conscious data handling** — resumable feature extraction, corrupted-file filtering, and I/O-optimized data loading for large-scale training.

---

## Approach

Two parallel pipelines were built and benchmarked against each other for the same attribution task:

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

---

## Technical Implementation

### 1. Feature Engineering & Data Pipeline
- Loads and profiles the dataset via `metadata.csv` — sample counts, TTS system distribution, speaker distribution.
- Extracts an 88-dimensional acoustic feature vector per sample: **MFCC** (13 coefficients), **delta** and **delta-delta** derivatives, **zero-crossing rate**, **spectral centroid**, **bandwidth**, **rolloff**, and **RMS energy** — mean and std for each.
- Resumable, fault-tolerant extraction loop across the full dataset, with results saved to both `.parquet` (fast) and `.csv` (inspectable) formats.

<img width="907" height="390" alt="Feature extraction visualization" src="https://github.com/user-attachments/assets/d4510889-0b22-48b0-b1c4-adcc6dc0c956" />

### 2. Classical Machine Learning
- Stratified 80/20 train/test split over the extracted feature table.
- **Random Forest** (200 trees), doubling as a feature-importance ranking tool.
- **XGBoost** (`multi:softmax`, 200 estimators, max depth 8) over label-encoded classes.
- Full evaluation suite: accuracy, per-class precision/recall/F1, and confusion matrices for both models.

### 3. Deep Learning Classifier
A CNN trained directly on log-mel spectrograms, as a higher-capacity alternative to hand-engineered features.

- **Input**: 128×128 log-mel spectrograms, indexed via `metadata.csv` + `label_map.json`.
- **Architecture**: Conv(1→32) → ReLU → MaxPool → Conv(32→64) → ReLU → MaxPool → FC(256) → ReLU → FC(num_classes).
- **Training**: Adam optimizer, cross-entropy loss, batch size 64, up to 20 epochs, with `ReduceLROnPlateau` scheduling and early stopping (patience 5).
- **Data engineering**: dataset staged from Google Drive to local SSD via multi-threaded copy (32 workers) to eliminate I/O bottlenecks during training; corrupted files filtered before training begins.
- **Deployment-ready inference**: a standalone `inference.py` loads the trained model and label encoder to predict the source TTS system for any new spectrogram.
- All artifacts — model weights, label encoder, training history, curves, and evaluation reports — are versioned and backed up automatically.

---

## Results

<img width="1500" height="1500" alt="Confusion matrix" src="https://github.com/user-attachments/assets/9249cd5a-3855-4c17-b638-526252fda533" />

<img width="2700" height="750" alt="Training curves" src="https://github.com/user-attachments/assets/9e68c533-2dd5-4892-a042-9feb2fa64952" />

Both the classical and deep learning pipelines were evaluated with full per-class breakdowns, giving direct insight into which TTS systems are most and least distinguishable from acoustic features alone.

---

## Repository Structure

```
data/        Raw dataset / metadata
notebooks/   Feature extraction, classical ML, and CNN notebooks
report/      Methodology and results write-up
results/     Saved outputs (models, logs, spectrogram datasets)
```

---

## Tech Stack

Python, PyTorch, librosa, pandas, NumPy, scikit-learn, XGBoost, Matplotlib

---

## Roadmap

- [ ] Deeper/pretrained CNN backbones or attention-based architectures
- [ ] Cross-dataset generalization — train on one set of TTS systems, evaluate on unseen ones
- [ ] Feature ablation to isolate the most attribution-relevant acoustic cues
- [ ] Ensemble the classical and CNN predictions for a stronger combined classifier
