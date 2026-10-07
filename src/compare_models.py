"""Final evaluation: compares SVM, CNN, Augmented CNN and Random Forest.

Reads results/<model>/predictions.csv written by each training script, so run
train_svm.py, train_cnn.py, train_rf.py and train_cnn_augmented.py first.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

from utils import CLASS_NAMES, compute_metrics, plot_confusion

MODELS = {
    "SVM": "results/svm",
    "CNN": "results/cnn",
    "CNN + Augmentation": "results/cnn_augmented",
    "Random Forest": "results/rf",
}
OUT = "results/comparison"


def main():
    os.makedirs(OUT, exist_ok=True)
    overall, per_class = [], []
    for name, folder in MODELS.items():
        path = f"{folder}/predictions.csv"
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} not found")
            continue
        df = pd.read_csv(path)
        m = compute_metrics(df.y_true, df.y_pred)
        overall.append({"Model": name, **{k.capitalize(): v for k, v in m.items()}})
        p, r, f, s = precision_recall_fscore_support(
            df.y_true, df.y_pred, labels=CLASS_NAMES, zero_division=0)
        for i, c in enumerate(CLASS_NAMES):
            per_class.append({"Model": name, "Class": c, "Precision": p[i], "Recall": r[i], "F1": f[i]})
        plot_confusion(df.y_true, df.y_pred, f"{name} Confusion Matrix",
                       f"{OUT}/cm_{name.replace(' ', '_').replace('+', 'plus')}.png")

    overall = pd.DataFrame(overall).set_index("Model")
    per_class = pd.DataFrame(per_class)
    overall.round(4).to_csv(f"{OUT}/overall_metrics.csv")
    per_class.round(4).to_csv(f"{OUT}/per_class_metrics.csv", index=False)

    print("\nOVERALL (macro-averaged precision/recall/F1)")
    print((overall * 100).round(2).to_string())

    f1 = per_class.pivot(index="Model", columns="Class", values="F1")[CLASS_NAMES]
    print("\nPER-CLASS F1 (%)")
    print((f1 * 100).round(2).to_string())
    hardest = f1.mean(axis=0).sort_values()
    print("\nMean F1 per class across models (lowest = hardest):")
    print((hardest * 100).round(2).to_string())
    print(f"\n>>> Hardest terrain to classify: {hardest.index[0]}")

    # Grouped bar chart of the four metrics
    ax = (overall * 100).plot.bar(figsize=(9, 5), rot=15)
    ax.set_ylim(90, 100.5); ax.set_ylabel("%"); ax.set_title("Model comparison")
    plt.tight_layout(); plt.savefig(f"{OUT}/model_comparison.png", dpi=300); plt.close()

    # Per-class F1 chart
    ax = (f1 * 100).T.plot.bar(figsize=(8, 5), rot=0)
    ax.set_ylim(90, 100.5); ax.set_ylabel("F1 (%)"); ax.set_title("Per-class F1 by model")
    plt.tight_layout(); plt.savefig(f"{OUT}/per_class_f1.png", dpi=300); plt.close()

    # Markdown results table, ready to paste in README / report
    with open(f"{OUT}/results_table.md", "w") as fh:
        fh.write("| Model | Accuracy | Precision | Recall | F1 |\n|---|---|---|---|---|\n")
        for name, r in overall.iterrows():
            fh.write(f"| {name} | {r.Accuracy*100:.2f}% | {r.Precision*100:.2f}% | "
                     f"{r.Recall*100:.2f}% | {r.F1*100:.2f}% |\n")
    print("\nSaved everything to", OUT)


if __name__ == "__main__":
    main()
