import pandas as pd
import numpy as np


FEATURE_PATH = "data/processed/features.csv"


def diagnose_features():

    df = pd.read_csv(FEATURE_PATH)

    feature_columns = [
        col for col in df.columns
        if col != "label"
    ]

    X = df[feature_columns]
    y = df["label"]

    print("=" * 70)
    print("FEATURE SEPARABILITY DIAGNOSTIC")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Class means
    # --------------------------------------------------

    print("\n1. CLASS MEANS")
    print("-" * 70)

    class_means = df.groupby("label")[feature_columns].mean().T

    print(class_means.to_string())

    # --------------------------------------------------
    # 2. Separation score
    # --------------------------------------------------

    print("\n\n2. FEATURE SEPARATION SCORES")
    print("-" * 70)

    separation_scores = {}

    for feature in feature_columns:

        overall_mean = X[feature].mean()

        between_variance = 0
        within_variance = 0

        for label in y.unique():

            class_values = X.loc[y == label, feature]

            class_mean = class_values.mean()
            class_size = len(class_values)

            between_variance += (
                class_size *
                (class_mean - overall_mean) ** 2
            )

            within_variance += np.sum(
                (class_values - class_mean) ** 2
            )

        score = (
            between_variance /
            (within_variance + 1e-8)
        )

        separation_scores[feature] = score

    ranked_features = sorted(
        separation_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nMost separating features:\n")

    for rank, (feature, score) in enumerate(
        ranked_features,
        start=1
    ):
        print(
            f"{rank:2d}. "
            f"{feature:<35} "
            f"{score:.4f}"
        )

    # --------------------------------------------------
    # 3. Feature ranges by class
    # --------------------------------------------------

    print("\n\n3. CLASS RANGES FOR TOP 10 FEATURES")
    print("-" * 70)

    top_features = [
        feature
        for feature, _ in ranked_features[:10]
    ]

    for feature in top_features:

        print(f"\n{feature}")

        for label in ["HF", "LF", "D", "G"]:

            values = X.loc[
                y == label,
                feature
            ]

            print(
                f"  {label}: "
                f"{values.min():.3f} - "
                f"{values.max():.3f}"
            )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print("\n\n4. SUMMARY")
    print("-" * 70)

    print(
        "The top features above are the ones we need "
        "to investigate first."
    )

    print(
        "\nIf their class ranges barely overlap, "
        "they may be making our synthetic dataset "
        "too easy to classify."
    )


if __name__ == "__main__":
    diagnose_features()