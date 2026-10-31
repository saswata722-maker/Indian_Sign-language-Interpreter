"""Geometry-preserving augmentation of XYZ landmarks with detection flags."""
import numpy as np
from preprocessing.features import sample_at


def time_warp(seq, sigma=.2, num_knots=4, feature_version=1):
    if len(seq) < 2:
        return seq.copy()
    increments = np.exp(np.random.normal(0, sigma, num_knots+1))
    knots = np.r_[0, np.cumsum(increments)]
    knots *= (len(seq)-1)/knots[-1]
    positions = np.interp(np.arange(len(seq)), np.linspace(0, len(seq)-1, len(knots)), knots)
    return sample_at(seq, positions, feature_version)


def _spatial(seq, transform):
    out = seq.astype(np.float32).copy()
    points = out[:, :-3].reshape(len(out), -1, 3)
    transformed = transform(points.copy())
    valid = np.ones(points.shape[:2], dtype=bool)
    valid[:, :33] = seq[:, -3, None] > .5
    valid[:, 33:54] = seq[:, -2, None] > .5
    valid[:, 54:75] = seq[:, -1, None] > .5
    points[valid] = transformed[valid]
    return out


def spatial_jitter(seq, noise_std=.01):
    return _spatial(seq, lambda p: p + np.random.normal(0, noise_std, p.shape))


def random_rotation(seq, max_degrees=10):
    angle = np.deg2rad(np.random.uniform(-max_degrees, max_degrees))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    def rotate(p):
        p[..., :2] = p[..., :2] @ rotation.T
        return p
    return _spatial(seq, rotate)


def random_scaling(seq, scale_range=(.9, 1.1)):
    scale = np.random.uniform(*scale_range)
    return _spatial(seq, lambda p: p*scale)


def random_translation(seq, max_shift=.05):
    shift = np.r_[np.random.uniform(-max_shift, max_shift, 2), 0]
    return _spatial(seq, lambda p: p+shift)


def temporal_crop(seq, crop_ratio=.1, feature_version=1):
    length = max(1, int(len(seq)*(1-crop_ratio)))
    start = np.random.randint(0, len(seq)-length+1)
    return sample_at(seq[start:start+length], np.linspace(0, length-1, len(seq)), feature_version)


def augment_sequence(seq, p_time_warp=.5, p_jitter=.5, p_crop=.3,
                     p_rotation=.3, p_scaling=.3, p_translation=.3, feature_version=1):
    out = seq.copy()
    for probability, operation in [
        (p_time_warp, lambda s: time_warp(s, feature_version=feature_version)),
        (p_jitter, spatial_jitter),
        (p_crop, lambda s: temporal_crop(s, np.random.uniform(.05, .15), feature_version)),
        (p_rotation, random_rotation), (p_scaling, random_scaling),
        (p_translation, random_translation),
    ]:
        if np.random.random() < probability:
            out = operation(out)
    return out
