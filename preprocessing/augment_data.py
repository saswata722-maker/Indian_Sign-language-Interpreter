"""Landmark-sequence augmentation utilities.

Used ON-THE-FLY by `training/dataset.py` and optionally offline via CLI to
write augmented copies of .npy sequences.

All functions assume the feature layout of `models/landmark_extractor.py`:
    [pose(99) | left_hand(63) | right_hand(63) | face(1404)? | flags(3)]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.landmark_extractor import (  # noqa: E402
    FACE_DIM, HAND_DIM, HANDS_DIM, POSE_DIM, segment_offsets,
)


def horizontal_flip(seq: np.ndarray, include_face: bool = True) -> np.ndarray:
    """Mirror a sequence in x: swap hands, negate x coordinates."""
    off = segment_offsets(include_face)
    out = seq.copy()
    d = seq.shape[-1]
    face_end = off["face"] + FACE_DIM if include_face else off["flags"]
    # x coordinates are at local offset 0 within each landmark triplet.
    out[:, 0:POSE_DIM:3] *= -1
    if include_face:
        out[:, off["face"]:face_end:3] *= -1
    # swap left/right hand blocks
    out[:, off["left_hand"]:off["left_hand"] + HAND_DIM] = seq[:, off["right_hand"]:off["right_hand"] + HAND_DIM]
    out[:, off["right_hand"]:off["right_hand"] + HAND_DIM] = seq[:, off["left_hand"]:off["left_hand"] + HAND_DIM]
    # swap hand-detection flags (positions 1 and 2 of the flag triple)
    f = off["flags"]
    out[:, f + 1], out[:, f + 2] = seq[:, f + 2].copy(), seq[:, f + 1].copy()
    return out


def time_warp(seq: np.ndarray, factor: float) -> np.ndarray:
    """Resample time axis by `factor` (>1 = slower/longer, <1 = faster)."""
    t = seq.shape[0]
    new_t = max(4, int(round(t * factor)))
    idx = np.linspace(0, t - 1, new_t)
    lo = np.floor(idx).astype(int)
    hi = np.clip(lo + 1, 0, t - 1)
    w = (idx - lo)[:, None, None].astype(seq.dtype)
    return (1 - w) * seq[lo] + w * seq[hi]


def jitter(seq: np.ndarray, sigma: float = 0.005, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    noise = rng.normal(0.0, sigma, seq.shape).astype(seq.dtype)
    noise[:, -3:] = 0.0  # do not corrupt detection flags
    return seq + noise


def random_temporal_crop(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Trim up to 20% of frames (10% each side) to vary start/end."""
    t = seq.shape[0]
    max_trim = max(1, int(t * 0.1))
    a = rng.integers(0, max_trim + 1)
    b = rng.integers(0, max_trim + 1)
    return seq[a: t - b] if t - b > a else seq


def augment(seq: np.ndarray, include_face: bool = True,
            rng: np.random.Generator | None = None) -> np.ndarray:
    """Random composition of flip / time-warp / jitter / crop."""
    rng = rng or np.random.default_rng()
    out = seq
    if rng.random() < 0.5:
        out = horizontal_flip(out, include_face)
    if rng.random() < 0.5:
        out = time_warp(out, float(rng.uniform(0.8, 1.2)))
    out = random_temporal_crop(out, rng)
    out = jitter(out, sigma=float(rng.uniform(0.002, 0.008)), rng=rng)
    return np.ascontiguousarray(out, dtype=np.float32)


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Offline augmentation of .npy sequences")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--copies", type=int, default=1)
    ap.add_argument("--no_face", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    files = sorted(args.src.rglob("*.npy"))
    for f in files:
        seq = np.load(f)
        for k in range(args.copies):
            out = args.dst / f.relative_to(args.src)
            out = out.with_name(out.stem + f"_aug{k}.npy")
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, augment(seq, include_face=not args.no_face, rng=rng))
    print(f"Wrote {len(files) * args.copies} augmented sequences to {args.dst}")


if __name__ == "__main__":
    _cli()
