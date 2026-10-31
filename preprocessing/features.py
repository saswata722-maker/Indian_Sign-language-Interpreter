"""Versioned preprocessing. V1 preserves the shipped checkpoint; V2 masks missing data."""
import numpy as np


def mask_missing(seq):
    out = np.asarray(seq, dtype=np.float32).copy()
    if out.ndim != 2 or out.shape[1] not in (228, 1632):
        raise ValueError('Expected (T, 228) or (T, 1632) landmarks with flags')
    out[:, -3:] = out[:, -3:] > .5
    for flag, start, stop in [(-3, 0, 99), (-2, 99, 162), (-1, 162, 225)]:
        out[out[:, flag] == 0, start:stop] = 0
    return out


def sample_at(seq, positions, feature_version=1):
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2 or len(seq) == 0 or not np.isfinite(seq).all():
        raise ValueError('Expected a nonempty finite sequence')
    if feature_version not in (1, 2):
        raise ValueError('Unsupported feature_version')
    if feature_version == 2:
        seq = mask_missing(seq)
    positions = np.clip(positions, 0, len(seq)-1)
    lo, hi = np.floor(positions).astype(int), np.ceil(positions).astype(int)
    frac = (positions-lo).astype(np.float32)[:, None]
    out = seq[lo]*(1-frac) + seq[hi]*frac
    if feature_version == 2:
        out[:, -3:] = np.minimum(seq[lo, -3:], seq[hi, -3:])
        out = mask_missing(out)
    return out.astype(np.float32)


def resample_sequence(seq, target_len=64, feature_version=1):
    if target_len < 1:
        raise ValueError('target_len must be positive')
    return sample_at(seq, np.linspace(0, len(seq)-1, target_len), feature_version)


def add_velocity(seq, feature_version=1):
    delta = np.diff(seq, axis=0, prepend=seq[:1])
    if feature_version == 2:
        delta[:, -3:] = 0
        for flag, start, stop in [(-3, 0, 99), (-2, 99, 162), (-1, 162, 225)]:
            valid = (seq[:, flag] > .5) & (np.roll(seq[:, flag], 1) > .5)
            valid[0] = False
            delta[~valid, start:stop] = 0
    return np.concatenate([seq, delta], axis=1).astype(np.float32)


def prepare_sequence(raw, config, timestamps=None):
    prep = config['preprocessing']
    version = prep.get('feature_version', 1)
    if timestamps is None:
        seq = resample_sequence(raw, prep['target_seq_len'], version)
    else:
        times = np.asarray(timestamps)
        if len(times) != len(raw) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('Capture timestamps must be finite and strictly increasing')
        positions = np.interp(np.linspace(times[0], times[-1], prep['target_seq_len']), times, np.arange(len(raw)))
        seq = sample_at(raw, positions, version)
    if prep.get('use_velocity', False):
        seq = add_velocity(seq, version)
    if seq.shape[1] != config['model']['input_dim']:
        raise ValueError('Checkpoint input_dim does not match preprocessing')
    return seq


def window_quality(raw, timestamps=None):
    """Hand XY motion per reference 25-FPS step, excluding tracking transitions."""
    raw = np.asarray(raw)
    pose = float((raw[:, -3] > .5).mean())
    hands = float((raw[:, -2:].max(axis=1) > .5).mean())
    motions = []
    dt = np.ones(max(0, len(raw)-1), dtype=np.float32)
    if timestamps is not None:
        dt = np.maximum(np.diff(timestamps)*25, 1e-3)
    for flag, start in [(-2, 99), (-1, 162)]:
        valid = (raw[:-1, flag] > .5) & (raw[1:, flag] > .5)
        coords = raw[:, start:start+63].reshape(-1, 21, 3)[..., :2]
        if valid.any():
            motions.extend((np.abs(np.diff(coords, axis=0)).mean(axis=(1, 2))/dt)[valid])
    return pose, hands, float(np.mean(motions)) if motions else 0.0
