"""Third model: Random Forest on the 33 handcrafted features (same split as SVM/CNN)."""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from utils import CLASS_NAMES, save_predictions, compute_metrics, plot_confusion

FEATURE_PATH = "data/processed/features.csv"
RESULTS_DIR = "results/rf"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def train_rf():
    df = pd.read_csv(FEATURE_PATH)
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print("Train:", X_train.shape, "Test:", X_test.shape)

    # Trees are scale-invariant, so no StandardScaler is needed here.
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    m = compute_metrics(y_test, y_pred)
    print("\n" + "=" * 50 + "\nRANDOM FOREST RESULTS\n" + "=" * 50)
    print(f"Accuracy: {m['accuracy'] * 100:.2f}%")
    print(classification_report(y_test, y_pred, digits=4))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_predictions(f"{RESULTS_DIR}/predictions.csv", y_test.values, y_pred)
    plot_confusion(y_test, y_pred, "Random Forest Confusion Matrix",
                   f"{RESULTS_DIR}/confusion_matrix.png")

    # Feature importance (top 15) - useful for viva: which features matter most
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
    imp.to_csv(f"{RESULTS_DIR}/feature_importance.csv", header=["importance"])
    plt.figure(figsize=(8, 6))
    imp.tail(15).plot.barh()
    plt.xlabel("Importance")
    plt.title("Random Forest - Top 15 Features")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/feature_importance.png", dpi=300)
    plt.close()
    print("Saved results to", RESULTS_DIR)


if __name__ == "__main__":
    train_rf()
