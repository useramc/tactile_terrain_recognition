import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

DATA_DIR = os.path.join("data", "raw")
RESULTS_DIR = os.path.join("results", "dataset_inspection")

CLASS_NAMES = {
    0: "HF",
    1: "LF",
    2: "D",
    3: "G"
}

N_CHANNELS = 6
N_TIME_STEPS = 69


# ============================================================
# Load dataset
# ============================================================

def load_dataset():

    tactile_data = np.load(
        os.path.join(DATA_DIR, "tactile_data.npy")
    )

    labels = pd.read_csv(
        os.path.join(DATA_DIR, "labels.csv")
    )

    control_data = pd.read_csv(
        os.path.join(DATA_DIR, "control_data.csv")
    )

    motor_rpm = np.load(
        os.path.join(DATA_DIR, "motor_rpm.npy")
    )

    input_current = np.load(
        os.path.join(DATA_DIR, "input_current.npy")
    )

    return (
        tactile_data,
        labels,
        control_data,
        motor_rpm,
        input_current
    )


# ============================================================
# Basic dataset information
# ============================================================

def inspect_basic_information(
    tactile_data,
    labels,
    control_data,
    motor_rpm,
    input_current
):

    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print(f"\nTactile data shape : {tactile_data.shape}")
    print(f"Labels shape       : {labels.shape}")
    print(f"Control data shape : {control_data.shape}")
    print(f"Motor RPM shape    : {motor_rpm.shape}")
    print(f"Input current shape: {input_current.shape}")

    print("\nExpected tactile shape:")
    print("(N, 6, 69)")

    print("\nActual:")
    print(tactile_data.shape)

    print("\nData type:")
    print(tactile_data.dtype)

    print("\nMissing / invalid values:")
    print(f"NaN values       : {np.isnan(tactile_data).sum()}")
    print(f"Infinite values  : {np.isinf(tactile_data).sum()}")

    print("\nControl data columns:")
    print(list(control_data.columns))


# ============================================================
# Class distribution
# ============================================================

def inspect_class_distribution(labels):

    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)

    counts = labels["terrain"].value_counts()

    print("\nSamples per class:")

    for class_id, class_name in CLASS_NAMES.items():

        count = counts.get(class_name, 0)

        print(
            f"{class_name}: {count}"
        )

    # Plot class distribution

    os.makedirs(RESULTS_DIR, exist_ok=True)

    plt.figure(figsize=(7, 5))

    plt.bar(
        counts.index,
        counts.values
    )

    plt.xlabel("Terrain Class")
    plt.ylabel("Number of Samples")
    plt.title("Synthetic Dataset Class Distribution")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "class_distribution.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# Plot one sample from each class
# ============================================================

def plot_sample_signals(tactile_data, labels):

    print("\nGenerating sample signal plots...")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    time = np.arange(N_TIME_STEPS)

    for class_id, class_name in CLASS_NAMES.items():

        indices = np.where(
            labels["label_id"].values == class_id
        )[0]

        sample_index = indices[0]

        sample = tactile_data[sample_index]

        plt.figure(figsize=(10, 6))

        for channel in range(N_CHANNELS):

            plt.plot(
                time,
                sample[channel],
                label=f"Channel {channel + 1}"
            )

        plt.xlabel("Time Sample")
        plt.ylabel("Force")
        plt.title(
            f"Example Tactile Signal - {class_name}"
        )

        plt.legend()
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                RESULTS_DIR,
                f"sample_{class_name}.png"
            ),
            dpi=300
        )

        plt.close()


# ============================================================
# Plot normal taxels separately
# ============================================================

def plot_normal_taxels(tactile_data, labels):

    print("Generating normal-force plots...")

    time = np.arange(N_TIME_STEPS)

    for class_id, class_name in CLASS_NAMES.items():

        indices = np.where(
            labels["label_id"].values == class_id
        )[0]

        sample_index = indices[0]

        sample = tactile_data[sample_index]

        plt.figure(figsize=(10, 6))

        for channel in range(5):

            plt.plot(
                time,
                sample[channel],
                label=f"Normal Taxel {channel + 1}"
            )

        plt.xlabel("Time Sample")
        plt.ylabel("Normal Force")

        plt.title(
            f"Normal Taxel Signals - {class_name}"
        )

        plt.legend()
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                RESULTS_DIR,
                f"normal_taxels_{class_name}.png"
            ),
            dpi=300
        )

        plt.close()


# ============================================================
# Plot shear signal
# ============================================================

def plot_shear_signals(tactile_data, labels):

    print("Generating shear-force plots...")

    time = np.arange(N_TIME_STEPS)

    plt.figure(figsize=(10, 6))

    for class_id, class_name in CLASS_NAMES.items():

        indices = np.where(
            labels["label_id"].values == class_id
        )[0]

        sample_index = indices[0]

        shear_signal = tactile_data[
            sample_index,
            5
        ]

        plt.plot(
            time,
            shear_signal,
            label=class_name
        )

    plt.xlabel("Time Sample")
    plt.ylabel("Shear Force")

    plt.title(
        "Shear Force Comparison Across Terrain Classes"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "shear_comparison.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# Plot total normal force comparison
# ============================================================

def plot_total_force_comparison(
    tactile_data,
    labels
):

    print("Generating total-force comparison...")

    time = np.arange(N_TIME_STEPS)

    plt.figure(figsize=(10, 6))

    for class_id, class_name in CLASS_NAMES.items():

        indices = np.where(
            labels["label_id"].values == class_id
        )[0]

        sample_index = indices[0]

        normal_force = np.sum(
            tactile_data[
                sample_index,
                :5,
                :
            ],
            axis=0
        )

        plt.plot(
            time,
            normal_force,
            label=class_name
        )

    plt.xlabel("Time Sample")
    plt.ylabel("Total Normal Force")

    plt.title(
        "Total Normal Force Comparison"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "total_force_comparison.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# Print value ranges
# ============================================================

def inspect_value_ranges(
    tactile_data,
    control_data
):

    print("\n" + "=" * 60)
    print("VALUE RANGES")
    print("=" * 60)

    print("\nTactile data:")

    for channel in range(N_CHANNELS):

        channel_data = tactile_data[:, channel, :]

        print(
            f"Channel {channel + 1}: "
            f"min={channel_data.min():.2f}, "
            f"max={channel_data.max():.2f}, "
            f"mean={channel_data.mean():.2f}, "
            f"std={channel_data.std():.2f}"
        )

    print("\nControl variables:")

    for column in control_data.columns:

        values = control_data[column]

        print(
            f"{column}: "
            f"min={values.min():.2f}, "
            f"max={values.max():.2f}, "
            f"mean={values.mean():.2f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("SYNTHETIC TACTILE DATASET INSPECTION")
    print("=" * 60)

    (
        tactile_data,
        labels,
        control_data,
        motor_rpm,
        input_current
    ) = load_dataset()

    inspect_basic_information(
        tactile_data,
        labels,
        control_data,
        motor_rpm,
        input_current
    )

    inspect_class_distribution(
        labels
    )

    inspect_value_ranges(
        tactile_data,
        control_data
    )

    plot_sample_signals(
        tactile_data,
        labels
    )

    plot_normal_taxels(
        tactile_data,
        labels
    )

    plot_shear_signals(
        tactile_data,
        labels
    )

    plot_total_force_comparison(
        tactile_data,
        labels
    )

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)

    print("\nPlots saved to:")
    print("results/dataset_inspection/")


if __name__ == "__main__":
    main()