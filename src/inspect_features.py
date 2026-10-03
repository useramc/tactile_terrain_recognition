import pandas as pd
import numpy as np


FEATURE_PATH = "data/processed/features.csv"


def inspect_features():

    # --------------------------------------------------
    # Load feature dataset
    # --------------------------------------------------

    df = pd.read_csv(FEATURE_PATH)

    feature_columns = [col for col in df.columns if col != "label"]

    print("Feature dataset inspection")
    print("=" * 50)

    # --------------------------------------------------
    # Basic information
    # --------------------------------------------------

    print("\nDataset shape:")
    print(df.shape)

    print("\nNumber of features:")
    print(len(feature_columns))

    print("\nFeature columns:")
    for i, feature in enumerate(feature_columns, start=1):
        print(f"{i:2d}. {feature}")

    # --------------------------------------------------
    # Class distribution
    # --------------------------------------------------

    print("\nClass distribution:")
    print(df["label"].value_counts().sort_index())

    # --------------------------------------------------
    # Missing values
    # --------------------------------------------------

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    # --------------------------------------------------
    # Infinite values
    # --------------------------------------------------

    numeric_features = df[feature_columns]

    inf_count = np.isinf(numeric_features.to_numpy()).sum()

    print("\nInfinite values:")
    print(inf_count)

    # --------------------------------------------------
    # Constant features
    # --------------------------------------------------

    print("\nConstant features:")

    constant_features = []

    for feature in feature_columns:
        if df[feature].nunique() <= 1:
            constant_features.append(feature)

    if constant_features:
        for feature in constant_features:
            print(feature)
    else:
        print("None")

    # --------------------------------------------------
    # Feature statistics
    # --------------------------------------------------

    print("\nFeature statistics:")
    print("-" * 80)

    statistics = pd.DataFrame({
        "min": numeric_features.min(),
        "max": numeric_features.max(),
        "mean": numeric_features.mean(),
        "std": numeric_features.std()
    })

    print(statistics.to_string())

    # --------------------------------------------------
    # First five rows
    # --------------------------------------------------

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    # --------------------------------------------------
    # Final validation
    # --------------------------------------------------

    print("\nValidation summary:")
    print("-" * 50)

    if df.shape == (4000, 34):
        print("✓ Dataset shape is correct")
    else:
        print("✗ Unexpected dataset shape")

    if len(feature_columns) == 33:
        print("✓ 33 features found")
    else:
        print("✗ Unexpected number of features")

    if df.isnull().sum().sum() == 0:
        print("✓ No missing values")
    else:
        print("✗ Missing values found")

    if inf_count == 0:
        print("✓ No infinite values")
    else:
        print("✗ Infinite values found")

    if not constant_features:
        print("✓ No constant features")
    else:
        print("✗ Constant features found")


if __name__ == "__main__":
    inspect_features()
    