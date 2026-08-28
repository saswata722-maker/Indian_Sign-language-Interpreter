"""PyTorch Dataset over landmark .npy sequences + split CSVs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.augment_data import augment  # noqa: E402


def pad_or_truncate(seq: np.ndarray, seq_len: int, train: bool,
                    rng: np.random.Generator | None = None):
    """Return (seq_len, D) float32 sequence + (seq_len,) bool validity mask.

    Training: random temporal window; eval: centered window.
    Sequences shorter than seq_len are zero-padded at the END (mask=False
    on padded frames).
    """
    t = seq.shape[0]
    if t > seq_len:
        if train:
            start = int(rng.integers(0, t - seq_len + 1)) if rng is not None else 0
        else:
            start = (t - seq_len) // 2
        seq = seq[start:start + seq_len]
        t = seq_len
    mask = np.zeros(seq_len, dtype=bool)
    mask[:t] = True
    if t < seq_len:
        seq = np.concatenate(
            [seq, np.zeros((seq_len - t, seq.shape[1]), dtype=seq.dtype)], axis=0
        )
    return seq.astype(np.float32), mask


class LandmarkDataset(Dataset):
    """Loads .npy landmark sequences listed in a split CSV.

    CSV columns: filepath,label,label_idx  (see utils/make_splits.py)
    """

    def __init__(self, csv_path: str | Path, seq_len: int = 96,
                 include_face: bool = True, train: bool = False, seed: int = 42):
        df = pd.read_csv(csv_path)
        self.paths = df["filepath"].tolist()
        self.labels = df["label_idx"].astype(int).tolist()
        self.classes = sorted(df["label"].unique())
        self.seq_len = seq_len
        self.include_face = include_face
        self.train = train
        self.rng = np.random.default_rng(seed if train else 0)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        seq = np.load(self.paths[i]).astype(np.float32)
        if self.train:
            seq = augment(seq, include_face=self.include_face, rng=self.rng)
        seq, mask = pad_or_truncate(seq, self.seq_len, train=self.train, rng=self.rng)
        return (
            torch.from_numpy(seq),
            torch.from_numpy(mask),
            torch.tensor(self.labels[i], dtype=torch.long),
        )
