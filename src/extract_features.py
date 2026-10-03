import os
import numpy as np
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

TACTILE_PATH = "data/raw/tactile_data.npy"
MOTOR_RPM_PATH = "data/raw/motor_rpm.npy"
INPUT_CURRENT_PATH = "data/raw/input_current.npy"
CONTROL_PATH = "data/raw/control_data.csv"
LABEL_PATH = "data/raw/labels.csv"

OUTPUT_DIR = "data/processed"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "features.csv")


# --------------------------------------------------
# Feature extraction
# --------------------------------------------------

def extract_features():
    # Load data
    tactile_data = np.load(TACTILE_PATH)
    motor_rpm = np.load(MOTOR_RPM_PATH)
    input_current = np.load(INPUT_CURRENT_PATH)

    control_data = pd.read_csv(CONTROL_PATH)
    labels = pd.read_csv(LABEL_PATH)

    # Use the label column from labels.csv
    if "label" in labels.columns:
        label_column = "label"
    else:
        label_column = labels.columns[-1]

    labels_array = labels[label_column].values

    print("Loaded data:")
    print("Tactile data :", tactile_data.shape)
    print("Motor RPM    :", motor_rpm.shape)
    print("Input current:", input_current.shape)
    print("Control data :", control_data.shape)
    print("Labels       :", labels.shape)

    features = []

    for i in range(len(tactile_data)):

        # ------------------------------------------
        # Basic signals
        # ------------------------------------------

        normal_force = tactile_data[i, :5, :]   # 5 normal taxels
        shear_force = tactile_data[i, 5, :]     # 1 shear taxel

        rpm = motor_rpm[i]
        current = input_current[i]

        force_sum = np.sum(normal_force, axis=0)

        # Index of maximum total normal force
        peak_sum_idx = np.argmax(force_sum)

        # ------------------------------------------
        # Control parameters
        # ------------------------------------------

        motor_rpm_mean = control_data.loc[i, "motor_rpm_mean"]
        Ts = control_data.loc[i, "Ts"]
        Tc = control_data.loc[i, "Tc"]
        omega_slow = control_data.loc[i, "omega_slow"]

        # ------------------------------------------
        # 1. Peak amplitude of sum
        # ------------------------------------------

        peak_sum_force = np.max(force_sum)

        # ------------------------------------------
        # 2. Area under curve of sum
        # ------------------------------------------

        area_sum_force = np.trapezoid(force_sum)

        # ------------------------------------------
        # 3. Motor RPM
        # ------------------------------------------

        motor_rpm_feature = motor_rpm_mean

        # ------------------------------------------
        # 4. Ratio Ts to Tc
        # ------------------------------------------

        ts_tc_ratio = Ts / Tc

        # ------------------------------------------
        # 5. Average amplitude of sum
        # ------------------------------------------

        mean_sum_force = np.mean(force_sum)

        # ------------------------------------------
        # 6. Gait parameter - omega_slow
        # ------------------------------------------

        omega_slow_feature = omega_slow

        # ------------------------------------------
        # 7-11. Individual taxel peak amplitudes
        # ------------------------------------------

        taxel_peaks = np.max(normal_force, axis=1)

        # ------------------------------------------
        # 12-16. Taxel peak / peak sum ratios
        # ------------------------------------------

        taxel_peak_ratios = taxel_peaks / (peak_sum_force + 1e-8)

        # ------------------------------------------
        # 17-19. Shear statistics
        # ------------------------------------------

        shear_min = np.min(shear_force)
        shear_max = np.max(shear_force)
        shear_mean = np.mean(shear_force)

        # ------------------------------------------
        # 20. Motor RPM at peak sum
        # ------------------------------------------

        motor_rpm_at_peak_sum = rpm[peak_sum_idx]

        # ------------------------------------------
        # 21. Input current at peak sum
        # ------------------------------------------

        input_current_at_peak_sum = current[peak_sum_idx]

        # ------------------------------------------
        # 22. Input current average
        # ------------------------------------------

        input_current_mean = np.mean(current)

        # ------------------------------------------
        # 23. Input current range
        # ------------------------------------------

        input_current_range = np.max(current) - np.min(current)

        # ------------------------------------------
        # 24-28. Average force of individual taxels
        # ------------------------------------------

        taxel_means = np.mean(normal_force, axis=1)

        # ------------------------------------------
        # 29-33. Motor RPM at peak force
        # for each individual taxel
        # ------------------------------------------

        motor_rpm_at_taxel_peaks = []

        for taxel in range(5):
            peak_idx = np.argmax(normal_force[taxel])
            motor_rpm_at_taxel_peaks.append(rpm[peak_idx])

        # ------------------------------------------
        # Combine all 33 features
        # ------------------------------------------

        sample_features = [
            # 1-6
            peak_sum_force,
            area_sum_force,
            motor_rpm_feature,
            ts_tc_ratio,
            mean_sum_force,
            omega_slow_feature,

            # 7-11
            *taxel_peaks,

            # 12-16
            *taxel_peak_ratios,

            # 17-19
            shear_min,
            shear_max,
            shear_mean,

            # 20-23
            motor_rpm_at_peak_sum,
            input_current_at_peak_sum,
            input_current_mean,
            input_current_range,

            # 24-28
            *taxel_means,

            # 29-33
            *motor_rpm_at_taxel_peaks,
        ]

        features.append(sample_features)

    # --------------------------------------------------
    # Feature names
    # --------------------------------------------------

    feature_names = [
        # 1-6
        "peak_sum_force",
        "area_sum_force",
        "motor_rpm",
        "ts_tc_ratio",
        "mean_sum_force",
        "omega_slow",

        # 7-11
        "taxel1_peak",
        "taxel2_peak",
        "taxel3_peak",
        "taxel4_peak",
        "taxel5_peak",

        # 12-16
        "taxel1_peak_ratio",
        "taxel2_peak_ratio",
        "taxel3_peak_ratio",
        "taxel4_peak_ratio",
        "taxel5_peak_ratio",

        # 17-19
        "shear_min",
        "shear_max",
        "shear_mean",

        # 20-23
        "motor_rpm_at_peak_sum",
        "input_current_at_peak_sum",
        "input_current_mean",
        "input_current_range",

        # 24-28
        "taxel1_mean",
        "taxel2_mean",
        "taxel3_mean",
        "taxel4_mean",
        "taxel5_mean",

        # 29-33
        "motor_rpm_at_taxel1_peak",
        "motor_rpm_at_taxel2_peak",
        "motor_rpm_at_taxel3_peak",
        "motor_rpm_at_taxel4_peak",
        "motor_rpm_at_taxel5_peak",
    ]

    # Safety check
    assert len(feature_names) == 33
    assert len(features[0]) == 33

    # --------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------

    features_df = pd.DataFrame(
        features,
        columns=feature_names
    )

    features_df["label"] = labels_array

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    if features_df.isnull().sum().sum() > 0:
        raise ValueError("NaN values found in extracted features.")

    if np.isinf(features_df.select_dtypes(include=np.number)).sum().sum() > 0:
        raise ValueError("Infinite values found in extracted features.")

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    features_df.to_csv(OUTPUT_PATH, index=False)

    print("\nFeature extraction complete!")
    print("Feature dataset shape:", features_df.shape)
    print("Number of features   :", len(feature_names))
    print("Output file           :", OUTPUT_PATH)

    print("\nFeature columns:")
    for number, name in enumerate(feature_names, start=1):
        print(f"{number:2d}. {name}")


if __name__ == "__main__":
    extract_features()