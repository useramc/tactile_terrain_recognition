import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

TACTILE_PATH = "data/raw/tactile_data.npy"
LABEL_PATH = "data/raw/labels.csv"

RESULTS_DIR = "results/cnn"

TEST_SIZE = 0.20
RANDOM_STATE = 42

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 30

NUM_CLASSES = 4

CLASS_NAMES = ["D", "G", "HF", "LF"]


class TerrainCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # Layer 1
            nn.Conv1d(
                in_channels=6,
                out_channels=24,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Layer 2
            nn.Conv1d(
                in_channels=24,
                out_channels=24,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Layer 3
            nn.Conv1d(
                in_channels=24,
                out_channels=24,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Layer 4
            nn.Conv1d(
                in_channels=24,
                out_channels=24,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Dropout(0.5)
        )

      
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(24 * 4, 128),
            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(128, NUM_CLASSES)
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x

def load_data():

    tactile_data = np.load(TACTILE_PATH)

    labels_df = pd.read_csv(LABEL_PATH)

    if "label" in labels_df.columns:
        labels = labels_df["label"].values
    else:
        labels = labels_df.iloc[:, -1].values

    print("Dataset loaded")
    print("Tactile data shape:", tactile_data.shape)
    print("Labels shape:", labels.shape)

    return tactile_data, labels


def encode_labels(labels):

    label_to_id = {
        "D": 0,
        "G": 1,
        "HF": 2,
        "LF": 3
    }

    encoded = np.array([
        label_to_id[label]
        for label in labels
    ])

    return encoded


def normalize_data(X_train, X_test):

    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)

    std = np.where(std == 0, 1, std)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test


def train_model(model, train_loader, criterion, optimizer, device):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == y_batch).sum().item()

        total += y_batch.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def evaluate_model(model, loader, device):

    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)

            outputs = model(X_batch)

            preds = torch.argmax(outputs, dim=1)

            predictions.extend(
                preds.cpu().numpy()
            )

            true_labels.extend(
                y_batch.numpy()
            )

    return np.array(true_labels), np.array(predictions)


def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    X, labels = load_data()

    y = encode_labels(labels)

    print("\nClass distribution:")

    unique, counts = np.unique(
        y,
        return_counts=True
    )

    for class_id, count in zip(unique, counts):

        print(
            f"{CLASS_NAMES[class_id]}: {count}"
        )


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


    X_train, X_test = normalize_data(
        X_train,
        X_test
    )

    print("\nNormalization complete.")


    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.long
    )

    y_test_tensor = torch.tensor(
        y_test,
        dtype=torch.long
    )


    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    test_dataset = TensorDataset(
        X_test_tensor,
        y_test_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nUsing device:", device)


    model = TerrainCNN().to(device)

    print("\nCNN architecture:")
    print(model)


    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    print("\nTraining CNN...")

    training_losses = []
    training_accuracies = []

    for epoch in range(EPOCHS):

        loss, accuracy = train_model(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        training_losses.append(loss)
        training_accuracies.append(accuracy)

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
            f"Loss: {loss:.4f} "
            f"Accuracy: {accuracy * 100:.2f}%"
        )


    y_true, y_pred = evaluate_model(
        model,
        test_loader,
        device
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    print("\n")
    print("=" * 50)
    print("CNN RESULTS")
    print("=" * 50)

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES
        )
    )


    print("Confusion Matrix:")

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print(cm)

    plt.figure(figsize=(7, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    display.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title(
        "CNN Confusion Matrix"
    )

    plt.tight_layout()

    confusion_path = os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=300
    )

    plt.close()


    plt.figure(figsize=(7, 5))

    plt.plot(
        range(1, EPOCHS + 1),
        training_losses
    )

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("CNN Training Loss")

    plt.tight_layout()

    loss_path = os.path.join(
        RESULTS_DIR,
        "training_loss.png"
    )

    plt.savefig(
        loss_path,
        dpi=300
    )

    plt.close()


    plt.figure(figsize=(7, 5))

    plt.plot(
        range(1, EPOCHS + 1),
        training_accuracies
    )

    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.title("CNN Training Accuracy")

    plt.tight_layout()

    accuracy_path = os.path.join(
        RESULTS_DIR,
        "training_accuracy.png"
    )

    plt.savefig(
        accuracy_path,
        dpi=300
    )

    plt.close()


    model_path = os.path.join(
        RESULTS_DIR,
        "terrain_cnn.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )


    print("\nResults saved to:")
    print(confusion_path)
    print(loss_path)
    print(accuracy_path)
    print(model_path)


if __name__ == "__main__":
    main()