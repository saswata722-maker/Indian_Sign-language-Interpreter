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


def random_rotation(seq, max_degrees=10):
    """
    Rotate the entire sequence in the XY plane around its centroid.

    Simulates camera/body orientation changes without distorting the
    internal geometry of the hand pose (rigid transform).

    Args:
        seq: np.ndarray of shape (T, D) with XY pairs interleaved as
             (x0, y0, x1, y1, ...) starting at column 0.
        max_degrees: Maximum rotation magnitude in degrees.

    Returns:
        np.ndarray of shape (T, D) — rotated sequence.
    """
    theta = np.random.uniform(-max_degrees, max_degrees) * np.pi / 180.0
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    out = seq.copy()
    # Assumes pairs (x, y) at even/odd column indices starting at 0
    xs = seq[:, 0::2]
    ys = seq[:, 1::2]

    cx, cy = xs.mean(), ys.mean()
    xs_c, ys_c = xs - cx, ys - cy

    out[:, 0::2] = cx + cos_t * xs_c - sin_t * ys_c
    out[:, 1::2] = cy + sin_t * xs_c + cos_t * ys_c
    return out.astype(np.float32)


def random_scaling(seq, scale_range=(0.9, 1.1)):
    """
    Scale the sequence spatially around its centroid.

    Simulates the signer being closer/farther from the camera.

    Args:
        seq: np.ndarray of shape (T, D) with XY pairs interleaved.
        scale_range: Min/max uniform scale factor.

    Returns:
        np.ndarray of shape (T, D) — scaled sequence.
    """
    s = np.random.uniform(scale_range[0], scale_range[1])
    out = seq.copy()
    xs, ys = seq[:, 0::2], seq[:, 1::2]
    cx, cy = xs.mean(), ys.mean()
    out[:, 0::2] = cx + (xs - cx) * s
    out[:, 1::2] = cy + (ys - cy) * s
    return out.astype(np.float32)


def random_translation(seq, max_shift=0.05):
    """
    Translate the whole sequence by a small random XY offset.

    Simulates signer positioning off-center in the frame.

    Args:
        seq: np.ndarray of shape (T, D) with XY pairs interleaved.
        max_shift: Maximum absolute shift per axis (in normalized coords).

    Returns:
        np.ndarray of shape (T, D) — translated sequence.
    """
    dx = np.random.uniform(-max_shift, max_shift)
    dy = np.random.uniform(-max_shift, max_shift)
    out = seq.copy()
    out[:, 0::2] += dx
    out[:, 1::2] += dy
    return out.astype(np.float32)


def augment_sequence(seq, p_time_warp=0.5, p_jitter=0.5, p_crop=0.3,
                     p_rotation=0.3, p_scaling=0.3, p_translation=0.3):
    """
    Randomly apply a combination of augmentations to a sequence.

    Args:
        seq: np.ndarray of shape (T, D).
        p_time_warp: Probability of applying time warping.
        p_jitter: Probability of applying spatial jitter.
        p_crop: Probability of applying temporal cropping.
        p_rotation: Probability of applying spatial rotation.
        p_scaling: Probability of applying spatial scaling.
        p_translation: Probability of applying spatial translation.

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
    if np.random.random() < p_rotation:
        aug_seq = random_rotation(aug_seq)
    if np.random.random() < p_scaling:
        aug_seq = random_scaling(aug_seq)
    if np.random.random() < p_translation:
        aug_seq = random_translation(aug_seq)

    return aug_seq
