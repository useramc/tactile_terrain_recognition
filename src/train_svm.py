import os

import pandas as pd
from utils import save_predictions
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

FEATURE_PATH = "data/processed/features.csv"
RESULTS_DIR = "results/svm"


# --------------------------------------------------
# Settings
# --------------------------------------------------

TEST_SIZE = 0.20
RANDOM_STATE = 42


# --------------------------------------------------
# Main
# --------------------------------------------------

def train_svm():

    # --------------------------------------------------
    # Load feature dataset
    # --------------------------------------------------

    df = pd.read_csv(FEATURE_PATH)

    X = df.drop(columns=["label"])
    y = df["label"]

    print("Dataset loaded")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # --------------------------------------------------
    # Train-test split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTrain-test split:")
    print("Training samples:", len(X_train))
    print("Testing samples :", len(X_test))

    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting class distribution:")
    print(y_test.value_counts().sort_index())

    # --------------------------------------------------
    # Feature scaling
    # --------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nFeature scaling complete.")

    # --------------------------------------------------
    # SVM model
    # --------------------------------------------------

    svm_model = SVC(
        kernel="rbf",
        random_state=RANDOM_STATE
    )

    print("\nTraining SVM...")
    svm_model.fit(X_train_scaled, y_train)

    print("SVM training complete.")

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    y_pred = svm_model.predict(X_test_scaled)

    # --------------------------------------------------
    # Accuracy
    # --------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print("SVM RESULTS")
    print("=" * 50)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # --------------------------------------------------
    # Classification report
    # --------------------------------------------------

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            digits=4
        )
    )

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["HF", "LF", "D", "G"]
    )

    print("Confusion Matrix:")
    print(cm)

    # --------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------

    os.makedirs(RESULTS_DIR, exist_ok=True)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["HF", "LF", "D", "G"]
    )

    display.plot(cmap="Blues")

    plt.title("SVM Confusion Matrix")
    plt.tight_layout()

    confusion_path = os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(confusion_path, dpi=300)
    plt.close()

    save_predictions(os.path.join(RESULTS_DIR, "predictions.csv"), y_test.values, y_pred)

    print("\nConfusion matrix saved to:")
    print(confusion_path)


if __name__ == "__main__":
    train_svm()