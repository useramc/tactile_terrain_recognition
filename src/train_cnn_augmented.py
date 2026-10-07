"""Retrain the 1D CNN on augmented training data and compare with the original CNN.

Run AFTER train_cnn.py (needs results/cnn/terrain_cnn.pth for the comparison).
Also runs a robustness test: both CNNs are evaluated on a clean test set and on a
perturbed (noisy / shifted / rescaled) test set to show what augmentation buys.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from train_cnn import (TerrainCNN, load_data, encode_labels, train_model,
                       evaluate_model, CLASS_NAMES as CNN_CLASSES,
                       BATCH_SIZE, LEARNING_RATE, EPOCHS, TEST_SIZE, RANDOM_STATE)
from augment import augment_dataset, jitter, scaling, time_shift
from utils import save_predictions, compute_metrics, plot_confusion

RESULTS_DIR = "results/cnn_augmented"
ORIGINAL_MODEL = "results/cnn/terrain_cnn.pth"
N_COPIES = 2          # training set becomes 3x (original + 2 augmented copies)


def make_loader(X, y, shuffle):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def predict(model, X, y, device):
    y_true, y_pred = evaluate_model(model, make_loader(X, y, False), device)
    return y_true, y_pred


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, labels = load_data()
    y = encode_labels(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    # Augment training data only (test set stays untouched)
    X_aug, y_aug = augment_dataset(X_train, y_train, n_copies=N_COPIES, seed=RANDOM_STATE)
    print(f"Training set: {len(X_train)} -> {len(X_aug)} after augmentation")

    # Normalise with statistics of the ORIGINAL training data
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std == 0, 1, std)
    norm = lambda a: (a - mean) / std

    # Perturbed test set (harder, unseen conditions) for the robustness check
    rng = np.random.default_rng(123)
    X_test_noisy = time_shift(scaling(jitter(X_test, rng, sigma=0.15), rng, 0.8, 1.2), rng, max_shift=6)

    model = TerrainCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loader = make_loader(norm(X_aug), y_aug, True)

    losses, accs = [], []
    for epoch in range(EPOCHS):
        loss, acc = train_model(model, loader, criterion, optimizer, device)
        losses.append(loss); accs.append(acc)
        print(f"Epoch [{epoch + 1:02d}/{EPOCHS}] Loss: {loss:.4f} Accuracy: {acc * 100:.2f}%")

    # Clean-test evaluation of the augmented CNN
    yt, yp = predict(model, norm(X_test), y_test, device)
    names_t, names_p = np.array(CNN_CLASSES)[yt], np.array(CNN_CLASSES)[yp]
    print("\nAUGMENTED CNN - clean test accuracy: %.2f%%" % (accuracy_score(yt, yp) * 100))
    print(classification_report(yt, yp, target_names=CNN_CLASSES, digits=4))
    save_predictions(f"{RESULTS_DIR}/predictions.csv", names_t, names_p)
    plot_confusion(names_t, names_p, "Augmented CNN Confusion Matrix", f"{RESULTS_DIR}/confusion_matrix.png")

    for name, vals, ylabel in [("training_loss", losses, "Training Loss"),
                               ("training_accuracy", accs, "Training Accuracy")]:
        plt.figure(figsize=(7, 5)); plt.plot(range(1, EPOCHS + 1), vals)
        plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title("Augmented CNN " + ylabel)
        plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/{name}.png", dpi=300); plt.close()
    torch.save(model.state_dict(), f"{RESULTS_DIR}/terrain_cnn_augmented.pth")

    # Robustness comparison: original CNN vs augmented CNN
    rows = {}
    original = TerrainCNN().to(device)
    if os.path.exists(ORIGINAL_MODEL):
        original.load_state_dict(torch.load(ORIGINAL_MODEL, map_location=device))
    else:
        original = None
        print("Original CNN weights not found - run train_cnn.py first for the comparison.")
    for tag, m in [("Original CNN", original), ("Augmented CNN", model)]:
        if m is None:
            continue
        _, p_clean = predict(m, norm(X_test), y_test, device)
        _, p_noisy = predict(m, norm(X_test_noisy), y_test, device)
        rows[tag] = (accuracy_score(y_test, p_clean), accuracy_score(y_test, p_noisy))

    lines = ["model,clean_test_accuracy,perturbed_test_accuracy"]
    print("\n" + "=" * 60 + "\nROBUSTNESS (clean vs perturbed test set)\n" + "=" * 60)
    for tag, (c, n) in rows.items():
        print(f"{tag:15s} clean: {c * 100:6.2f}%   perturbed: {n * 100:6.2f}%")
        lines.append(f"{tag},{c:.4f},{n:.4f}")
    open(f"{RESULTS_DIR}/robustness.csv", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
