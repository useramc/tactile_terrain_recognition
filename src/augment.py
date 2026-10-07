"""Data augmentation for tactile data of shape (N, 6, 69).

Each augmentation mimics a realistic variation of the sensor/robot:
  jitter     - sensor noise
  scaling    - different contact force / sensor gain (per channel)
  time_shift - the contact event occurs slightly earlier/later in the window
Augmentation is applied ONLY to the training set (never to test data).
"""
import numpy as np


def jitter(x, rng, sigma=0.03):
    # noise relative to each channel's own spread
    scale = x.std(axis=2, keepdims=True) + 1e-8
    return x + rng.normal(0, sigma, x.shape) * scale


def scaling(x, rng, low=0.9, high=1.1):
    factors = rng.uniform(low, high, size=(x.shape[0], x.shape[1], 1))
    return x * factors


def time_shift(x, rng, max_shift=4):
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        s = rng.integers(-max_shift, max_shift + 1)
        out[i] = np.roll(x[i], s, axis=1)
    return out


def augment_once(x, rng):
    """One randomly augmented copy of every sample."""
    return time_shift(scaling(jitter(x, rng), rng), rng)


def augment_dataset(X, y, n_copies=2, seed=42):
    """Return original + n_copies augmented versions of (X, y)."""
    rng = np.random.default_rng(seed)
    Xs, ys = [X], [y]
    for _ in range(n_copies):
        Xs.append(augment_once(X, rng).astype(X.dtype))
        ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


if __name__ == "__main__":
    X = np.load("data/raw/tactile_data.npy")[:10]
    y = np.arange(10)
    Xa, ya = augment_dataset(X, y, n_copies=2)
    print("Original:", X.shape, "-> Augmented:", Xa.shape, ya.shape)
