"""
Data Augmentation Module
Applies temporal and spatial augmentations to landmark sequences.

Based on AI4Bharat INCLUDE & OpenHands best practices:
  - Temporal speed jitter (±15%)
  - Spatial Gaussian noise
  - Subtle spatial scaling/rotation

NOTE: Do NOT horizontally flip signs that are handedness-dependent.
"""

import numpy as np


def time_warp(seq, sigma=0.2, num_knots=4):
    """
    Apply temporal warping via smooth random interpolation.

    Args:
        seq: np.ndarray of shape (T, D) — landmark sequence.
        sigma: Controls the magnitude of warping.
        num_knots: Number of control points for the warp spline.

    Returns:
        np.ndarray of shape (T, D) — warped sequence.
    """
    orig_len = seq.shape[0]
    # Generate random warp path
    knot_x = np.linspace(0, orig_len - 1, num=num_knots + 2)
    knot_y = knot_x + np.random.normal(0, sigma * orig_len, size=num_knots + 2)
    warp_path = np.interp(np.arange(orig_len), knot_x, knot_y)
    warp_path = np.clip(warp_path, 0, orig_len - 1)

    # Interpolate each dimension
    warped = np.zeros_like(seq, dtype=np.float32)
    for d in range(seq.shape[1]):
        warped[:, d] = np.interp(warp_path, np.arange(orig_len), seq[:, d])
    return warped


def spatial_jitter(seq, noise_std=0.01):
    """
    Add subtle Gaussian noise to all landmark coordinates.

    Args:
        seq: np.ndarray of shape (T, D).
        noise_std: Standard deviation of Gaussian noise.

    Returns:
        np.ndarray of shape (T, D) — jittered sequence.
    """
    noise = np.random.normal(0, noise_std, seq.shape).astype(np.float32)
    return (seq + noise).astype(np.float32)


def temporal_crop(seq, crop_ratio=0.1):
    """
    Randomly crop a contiguous subsequence and resample back to original length.

    Args:
        seq: np.ndarray of shape (T, D).
        crop_ratio: Fraction of sequence to remove (0 to 0.3).

    Returns:
        np.ndarray of shape (T, D) — cropped and resampled sequence.
    """
    orig_len = seq.shape[0]
    crop_len = int(orig_len * (1 - crop_ratio))
    start = np.random.randint(0, orig_len - crop_len + 1)
    cropped = seq[start:start + crop_len]

    # Resample back to original length
    indices = np.linspace(0, crop_len - 1, orig_len)
    resampled = np.zeros_like(seq, dtype=np.float32)
    for d in range(seq.shape[1]):
        resampled[:, d] = np.interp(indices, np.arange(crop_len), cropped[:, d])
    return resampled


def augment_sequence(seq, p_time_warp=0.5, p_jitter=0.5, p_crop=0.3):
    """
    Randomly apply a combination of augmentations to a sequence.

    Args:
        seq: np.ndarray of shape (T, D).
        p_time_warp: Probability of applying time warping.
        p_jitter: Probability of applying spatial jitter.
        p_crop: Probability of applying temporal cropping.

    Returns:
        np.ndarray of shape (T, D) — augmented sequence.
    """
    aug_seq = seq.copy()

    if np.random.random() < p_time_warp:
        aug_seq = time_warp(aug_seq)
    if np.random.random() < p_jitter:
        aug_seq = spatial_jitter(aug_seq)
    if np.random.random() < p_crop:
        aug_seq = temporal_crop(aug_seq, crop_ratio=np.random.uniform(0.05, 0.15))

    return aug_seq
