import os
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42
N_SAMPLES_PER_CLASS = 1000

N_CLASSES = 4
N_CHANNELS = 6
N_TIME_STEPS = 69

CLASS_NAMES = ["HF", "LF", "D", "G"]

CLASS_IDS = {
    "HF": 0,
    "LF": 1,
    "D": 2,
    "G": 3
}

OUTPUT_DIR = os.path.join("data", "raw")


# ============================================================
# Utility functions
# ============================================================

def gaussian_curve(t, center, width):
    """Generate a Gaussian-shaped curve."""
    return np.exp(
        -((t - center) ** 2) / (2 * width ** 2)
    )


def smooth_noise(rng, size, scale=1.0):
    """
    Generate correlated noise instead of independent
    sample-to-sample noise.
    """

    noise = rng.normal(0, scale, size)

    kernel = np.array([
        0.15,
        0.25,
        0.20,
        0.25,
        0.15
    ])

    padded = np.pad(
        noise,
        (2, 2),
        mode="edge"
    )

    smoothed = np.convolve(
        padded,
        kernel,
        mode="valid"
    )

    return smoothed


# ============================================================
# Terrain-specific parameters
# ============================================================

TERRAIN_PARAMETERS = {

    # --------------------------------------------------------
    # High friction + high stiffness
    # --------------------------------------------------------
    "HF": {
        "force_amplitude": (85, 110),
        "contact_width": (5.5, 8.0),
        "force_noise": (1.5, 3.0),

        "shear_amplitude": (3, 7),
        "shear_noise": (0.5, 1.0),

        "rpm": (135, 165),
        "current": (1.8, 2.5),
    },

    # --------------------------------------------------------
    # Low friction + low stiffness
    # --------------------------------------------------------
    "LF": {
        "force_amplitude": (78, 105),
        "contact_width": (6.5, 9.5),
        "force_noise": (2.5, 5.0),

        "shear_amplitude": (10, 20),
        "shear_noise": (2.0, 4.0),

        "rpm": (140, 175),
        "current": (1.5, 2.3),
    },

    # --------------------------------------------------------
    # Deformable
    # --------------------------------------------------------
    "D": {
        "force_amplitude": (55, 80),
        "contact_width": (11, 16),
        "force_noise": (1.5, 3.5),

        "shear_amplitude": (4, 9),
        "shear_noise": (0.7, 1.5),

        "rpm": (105, 140),
        "current": (2.0, 3.0),
    },

    # --------------------------------------------------------
    # Granular
    # --------------------------------------------------------
    "G": {
        "force_amplitude": (60, 90),
        "contact_width": (9, 14),
        "force_noise": (4.0, 7.0),

        "shear_amplitude": (7, 15),
        "shear_noise": (2.5, 5.0),

        "rpm": (115, 150),
        "current": (1.8, 2.8),
    },
}


# ============================================================
# Generate terrain-specific normal-force profile
# ============================================================

def generate_normal_force(
    terrain,
    t,
    amplitude,
    width,
    center,
    rng
):
    """
    Generate the underlying total normal-force profile.

    The four terrains have different general contact
    characteristics, but random variation is retained.
    """

    width_normalized = width / N_TIME_STEPS

    # --------------------------------------------------------
    # HF: concentrated, relatively sharp contact
    # --------------------------------------------------------

    if terrain == "HF":

        force = amplitude * gaussian_curve(
            t,
            center,
            width_normalized
        )

    # --------------------------------------------------------
    # LF: similar basic contact profile to HF.
    # The main terrain distinction is shear/slip.
    # --------------------------------------------------------

    elif terrain == "LF":

        force = amplitude * gaussian_curve(
            t,
            center,
            width_normalized
        )

        # Small asymmetry caused by changing contact
        # conditions.
        asymmetry = rng.uniform(-0.08, 0.08)

        force *= (
            1
            + asymmetry * (t - center)
        )

    # --------------------------------------------------------
    # D: broad, flatter contact profile
    # --------------------------------------------------------

    elif terrain == "D":

        # Combine two nearby broad Gaussian components
        # to create a flatter contact region.
        component_1 = gaussian_curve(
            t,
            center - 0.04,
            width_normalized * 1.15
        )

        component_2 = gaussian_curve(
            t,
            center + 0.04,
            width_normalized * 1.15
        )

        force = amplitude * (
            0.55 * component_1
            + 0.55 * component_2
        )

    # --------------------------------------------------------
    # G: irregular / multi-peak contact
    # --------------------------------------------------------

    elif terrain == "G":

        main_force = gaussian_curve(
            t,
            center,
            width_normalized
        )

        # Secondary contact peaks.
        peak_1 = gaussian_curve(
            t,
            center - rng.uniform(0.08, 0.13),
            width_normalized * rng.uniform(0.7, 1.0)
        )

        peak_2 = gaussian_curve(
            t,
            center + rng.uniform(0.07, 0.12),
            width_normalized * rng.uniform(0.7, 1.0)
        )

        force = amplitude * (
            0.65 * main_force
            + rng.uniform(0.15, 0.30) * peak_1
            + rng.uniform(0.15, 0.30) * peak_2
        )

    else:
        raise ValueError(
            f"Unknown terrain: {terrain}"
        )

    return force


# ============================================================
# Generate one robot step
# ============================================================

def generate_step(terrain, rng):
    """
    Generate one synthetic robot step.

    Returns
    -------
    tactile_data : ndarray, shape (6, 69)
        Channels 1-5: normal-force taxels
        Channel 6: shear-force taxel

    control_data : dict
        Synthetic gait/control variables.
    """

    params = TERRAIN_PARAMETERS[terrain]

    t = np.linspace(
        0,
        1,
        N_TIME_STEPS
    )

    # --------------------------------------------------------
    # Random step-specific parameters
    # --------------------------------------------------------

    amplitude = rng.uniform(
        *params["force_amplitude"]
    )

    width = rng.uniform(
        *params["contact_width"]
    )

    center = rng.uniform(
        0.42,
        0.58
    )

    baseline = rng.uniform(
        0.5,
        2.0
    )

    # --------------------------------------------------------
    # Generate shared terrain-specific force profile
    # --------------------------------------------------------

    base_force = generate_normal_force(
        terrain,
        t,
        amplitude,
        width,
        center,
        rng
    )

    base_force += baseline

    # --------------------------------------------------------
    # Shared variation across all five taxels
    #
    # This represents the fact that all taxels experience
    # the same underlying contact event.
    # --------------------------------------------------------

    shared_variation = smooth_noise(
        rng,
        N_TIME_STEPS,
        scale=rng.uniform(0.8, 1.5)
    )

    # Granular terrain gets stronger shared irregularity.
    if terrain == "G":

        shared_variation += smooth_noise(
            rng,
            N_TIME_STEPS,
            scale=rng.uniform(2.0, 4.0)
        )

    # Deformable terrain remains smoother.
    elif terrain == "D":

        shared_variation *= 0.5

    # --------------------------------------------------------
    # Generate five normal-force taxels
    # --------------------------------------------------------

    normal_channels = []

    for taxel in range(5):

        # ----------------------------------------------------
        # Each taxel receives a slightly different proportion
        # of the same underlying force.
        # ----------------------------------------------------

        sensor_scale = rng.uniform(
            0.90,
            1.08
        )

        # Only a very small timing difference.
        shift = rng.integers(
            -1,
            2
        )

        shifted_force = np.roll(
            base_force,
            shift
        )

        # ----------------------------------------------------
        # Shared terrain variation
        # ----------------------------------------------------

        shifted_variation = np.roll(
            shared_variation,
            shift
        )

        # ----------------------------------------------------
        # Small taxel-specific noise
        #
        # Much smaller than before so the five curves remain
        # correlated.
        # ----------------------------------------------------

        individual_noise_scale = (
            rng.uniform(
                0.15,
                0.35
            )
            * rng.uniform(
                *params["force_noise"]
            )
        )

        individual_noise = smooth_noise(
            rng,
            N_TIME_STEPS,
            scale=individual_noise_scale
        )

        # ----------------------------------------------------
        # Combine components
        # ----------------------------------------------------

        signal = (
            shifted_force * sensor_scale
            + shifted_variation
            + individual_noise
        )

        # ----------------------------------------------------
        # Additional granular irregularity
        #
        # Same underlying irregularity is shared across
        # taxels, but each taxel responds slightly differently.
        # ----------------------------------------------------

        if terrain == "G":

            granular_component = smooth_noise(
                rng,
                N_TIME_STEPS,
                scale=rng.uniform(
                    1.0,
                    2.0
                )
            )

            signal += granular_component

        # ----------------------------------------------------
        # Deformable terrain remains smooth
        # ----------------------------------------------------

        elif terrain == "D":

            # Mild smoothing toward the local mean.
            kernel = np.array([
                0.2,
                0.3,
                0.3,
                0.2
            ])

            padded = np.pad(
                signal,
                (1, 2),
                mode="edge"
            )

            signal = np.convolve(
                padded,
                kernel,
                mode="valid"
            )

        # ----------------------------------------------------
        # Force cannot be negative
        # ----------------------------------------------------

        signal = np.maximum(
            signal,
            0
        )

        normal_channels.append(
            signal
        )

    normal_channels = np.array(
        normal_channels
    )

    # ========================================================
    # Generate shear-force channel
    # ========================================================

    shear_amplitude = rng.uniform(
        *params["shear_amplitude"]
    )

    shear_profile = (
        shear_amplitude
        * gaussian_curve(
            t,
            center + rng.uniform(
                -0.03,
                0.03
            ),
            (width * 1.2) / N_TIME_STEPS
        )
    )

    shear_noise_scale = rng.uniform(
        *params["shear_noise"]
    )

    shear_noise = smooth_noise(
        rng,
        N_TIME_STEPS,
        scale=shear_noise_scale
    )

    shear_signal = (
        shear_profile
        + shear_noise
    )

    # --------------------------------------------------------
    # LF: strong oscillations caused by slip
    # --------------------------------------------------------

    if terrain == "LF":

        frequency = rng.uniform(
            7,
            12
        )

        oscillation = (
            rng.uniform(2, 5)
            * np.sin(
                2 * np.pi * frequency * t
                + rng.uniform(
                    0,
                    2 * np.pi
                )
            )
        )

        shear_signal += oscillation

    # --------------------------------------------------------
    # G: irregular shear fluctuations
    # --------------------------------------------------------

    elif terrain == "G":

        irregularity = smooth_noise(
            rng,
            N_TIME_STEPS,
            scale=rng.uniform(
                2,
                5
            )
        )

        shear_signal += irregularity

    # --------------------------------------------------------
    # D: smoother / lower shear
    # --------------------------------------------------------

    elif terrain == "D":

        shear_signal *= rng.uniform(
            0.75,
            0.95
        )

    # --------------------------------------------------------
    # Allow positive and negative shear values
    # --------------------------------------------------------

    shear_signal -= (
        np.mean(shear_signal) * 0.15
    )

    # ========================================================
    # Combine all six tactile channels
    # ========================================================

    tactile_data = np.vstack([
        normal_channels,
        shear_signal
    ])

    # ========================================================
    # Generate control / gait variables
    # ========================================================

    motor_rpm_mean = rng.uniform(
        *params["rpm"]
    )

    rpm_variation = smooth_noise(
        rng,
        N_TIME_STEPS,
        scale=rng.uniform(
            1.0,
            3.0
        )
    )

    motor_rpm_signal = (
        motor_rpm_mean
        + rpm_variation
    )

    # --------------------------------------------------------
    # Gait timing
    # --------------------------------------------------------

    Ts = rng.uniform(
        0.25,
        0.55
    )

    Tc = rng.uniform(
        0.55,
        0.95
    )

    # --------------------------------------------------------
    # Slow angular velocity
    # --------------------------------------------------------

    omega_slow = rng.uniform(
        0.5,
        1.5
    )

    # --------------------------------------------------------
    # Input current
    # --------------------------------------------------------

    mean_current = rng.uniform(
        *params["current"]
    )

    force_sum = np.sum(
        normal_channels,
        axis=0
    )

    normalized_force = (
        force_sum
        / (np.max(force_sum) + 1e-8)
    )

    input_current_signal = (
        mean_current
        + 0.25 * normalized_force
        + smooth_noise(
            rng,
            N_TIME_STEPS,
            scale=0.05
        )
    )

    input_current_signal = np.maximum(
        input_current_signal,
        0
    )

    # --------------------------------------------------------
    # Store control variables
    # --------------------------------------------------------

    control_data = {
        "motor_rpm": motor_rpm_signal,
        "Ts": Ts,
        "Tc": Tc,
        "omega_slow": omega_slow,
        "input_current": input_current_signal
    }

    return (
        tactile_data,
        control_data
    )


# ============================================================
# Generate complete dataset
# ============================================================

def generate_dataset():

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    total_samples = (
        N_CLASSES
        * N_SAMPLES_PER_CLASS
    )

    # --------------------------------------------------------
    # Allocate arrays
    # --------------------------------------------------------

    tactile_dataset = np.zeros(
        (
            total_samples,
            N_CHANNELS,
            N_TIME_STEPS
        ),
        dtype=np.float32
    )

    labels = np.zeros(
        total_samples,
        dtype=np.int64
    )

    motor_rpm = np.zeros(
        (
            total_samples,
            N_TIME_STEPS
        ),
        dtype=np.float32
    )

    input_current = np.zeros(
        (
            total_samples,
            N_TIME_STEPS
        ),
        dtype=np.float32
    )

    Ts_values = np.zeros(
        total_samples
    )

    Tc_values = np.zeros(
        total_samples
    )

    omega_slow_values = np.zeros(
        total_samples
    )

    sample_index = 0

    # ========================================================
    # Generate samples class by class
    # ========================================================

    for terrain in CLASS_NAMES:

        print(
            f"Generating {terrain} samples..."
        )

        for _ in range(
            N_SAMPLES_PER_CLASS
        ):

            tactile_data, control_data = (
                generate_step(
                    terrain,
                    rng
                )
            )

            tactile_dataset[
                sample_index
            ] = tactile_data

            labels[
                sample_index
            ] = CLASS_IDS[terrain]

            motor_rpm[
                sample_index
            ] = control_data["motor_rpm"]

            input_current[
                sample_index
            ] = control_data["input_current"]

            Ts_values[
                sample_index
            ] = control_data["Ts"]

            Tc_values[
                sample_index
            ] = control_data["Tc"]

            omega_slow_values[
                sample_index
            ] = control_data["omega_slow"]

            sample_index += 1

    # ========================================================
    # Shuffle dataset
    # ========================================================

    indices = rng.permutation(
        total_samples
    )

    tactile_dataset = (
        tactile_dataset[indices]
    )

    labels = labels[indices]

    motor_rpm = motor_rpm[indices]

    input_current = (
        input_current[indices]
    )

    Ts_values = Ts_values[indices]

    Tc_values = Tc_values[indices]

    omega_slow_values = (
        omega_slow_values[indices]
    )

    # ========================================================
    # Create output directory
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ========================================================
    # Save tactile data
    # ========================================================

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "tactile_data.npy"
        ),
        tactile_dataset
    )

    # ========================================================
    # Save labels
    # ========================================================

    labels_df = pd.DataFrame({
        "label_id": labels,
        "terrain": [
            CLASS_NAMES[label]
            for label in labels
        ]
    })

    labels_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "labels.csv"
        ),
        index=False
    )

    # ========================================================
    # Save control variables
    # ========================================================

    control_df = pd.DataFrame({
        "Ts": Ts_values,
        "Tc": Tc_values,
        "omega_slow": omega_slow_values,

        "motor_rpm_mean": np.mean(
            motor_rpm,
            axis=1
        ),

        "input_current_mean": np.mean(
            input_current,
            axis=1
        )
    })

    control_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "control_data.csv"
        ),
        index=False
    )

    # ========================================================
    # Save time-dependent control signals
    # ========================================================

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "motor_rpm.npy"
        ),
        motor_rpm
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "input_current.npy"
        ),
        input_current
    )

    # ========================================================
    # Print summary
    # ========================================================

    print(
        "\nDataset generation complete!"
    )

    print(
        f"\nTactile data shape: "
        f"{tactile_dataset.shape}"
    )

    print(
        f"Labels shape:       "
        f"{labels.shape}"
    )

    print("\nClass distribution:")

    for class_name, class_id in CLASS_IDS.items():

        count = np.sum(
            labels == class_id
        )

        print(
            f"{class_name}: {count}"
        )

    print("\nFiles created:")

    print(
        "data/raw/tactile_data.npy"
    )

    print(
        "data/raw/labels.csv"
    )

    print(
        "data/raw/control_data.csv"
    )

    print(
        "data/raw/motor_rpm.npy"
    )

    print(
        "data/raw/input_current.npy"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    generate_dataset()