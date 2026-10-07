"""Shared helpers: class names, saving predictions, computing metrics."""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

CLASS_NAMES = ["HF", "LF", "D", "G"]   # order used in all result tables/plots


def save_predictions(path, y_true, y_pred):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(path, index=False)


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_NAMES, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f}


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return cm
