# Tactile Terrain Recognition

Classifying terrain from tactile sensor signals using SVM, 1D-CNN and Random Forest,
based on a tactile terrain recognition paper (terrain classes: **HF, LF, D, G**).

> **Note:** the original dataset is not publicly available, so this project uses a
> **synthetically generated dataset** (4000 samples, 1000 per class) that mimics the
> paper's setup: a 6 x 69 tactile signal per sample plus motor RPM / current / control data.
> Results therefore show the behaviour of the methods on synthetic data and are not
> directly comparable to the paper's real-world numbers.

## Problem statement
Given a tactile reading (6 channels x 69 time steps) from a robot foot/sensor, predict the
terrain it is walking on: HF, LF, D or G.

## Project structure
```
data/raw/          generated tactile, motor and label data
data/processed/    features.csv (33 handcrafted features + label)
src/               all scripts
results/           confusion matrices, curves, comparison tables
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to run (from the project root)
```bash
python src/generate_dataset.py       # 1. generate synthetic data
python src/inspect_dataset.py        #    (optional) plots to check the data
python src/extract_features.py       # 2. extract the 33 handcrafted features
python src/train_svm.py              # 3. SVM baseline
python src/train_cnn.py              # 4. CNN baseline
python src/train_rf.py               # 5. Random Forest
python src/train_cnn_augmented.py    # 6. CNN with data augmentation (+ robustness test)
python src/compare_models.py         # 7. final comparison tables and plots
```
All models use the same stratified 80/20 split (`random_state=42`).

## Methods
- **SVM** (RBF kernel) on standardised handcrafted features.
- **1D CNN** on raw 6 x 69 tactile data, per-channel normalisation from training data.
- **Random Forest** (300 trees) on the same handcrafted features.
- **Data augmentation** (training set only): noise jitter, per-channel amplitude scaling,
  and time shift; training set is tripled (original + 2 augmented copies).

## Results
Final table is generated in `results/comparison/results_table.md`
(accuracy, macro precision / recall / F1), with per-class metrics in
`results/comparison/per_class_metrics.csv`.

<!-- Paste results_table.md here after running compare_models.py -->

## Limitations / future work
- Synthetic data: real sensor noise, wear and terrain variability are not captured.
- Test set comes from the same generator as training; real-world validation is needed.
- Future: real tactile data, deeper/temporal models, deployment on a robot.
