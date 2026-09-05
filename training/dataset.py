"""
Dataset & Preprocessing Module
Loads .npy files from landmarks/, resamples them to a fixed sequence length,
and prepares batches for PyTorch training.

Based on AI4Bharat INCLUDE best practices:
  - Linear interpolation resampling to target_seq_len
  - Optional data augmentation (time warp, jitter, crop)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

from preprocessing.augment_data import augment_sequence


def resample_sequence(seq, target_len=40):
    """
    Resample a sequence to a fixed length using linear interpolation.
    Vectorized across all feature dimensions for speed.

    Args:
        seq: np.ndarray of shape (T, D).
        target_len: Desired output sequence length.

    Returns:
        np.ndarray of shape (target_len, D).
    """
    curr_len = seq.shape[0]
    if curr_len == target_len:
        return seq
    if curr_len == 1:
        return np.repeat(seq, target_len, axis=0)
    indices = np.linspace(0, curr_len - 1, target_len)
    lo = np.floor(indices).astype(np.int64)
    hi = np.minimum(lo + 1, curr_len - 1)
    frac = (indices - lo).astype(np.float32)[:, None]
    resampled = seq[lo] * (1.0 - frac) + seq[hi] * frac
    return resampled.astype(np.float32)


class ISLLandmarkDataset(Dataset):
    """
    PyTorch Dataset for ISL landmark sequences.

    Args:
        npy_files: List of paths to .npy landmark files.
        labels: List of integer labels corresponding to npy_files.
        target_len: Fixed sequence length to resample to (default 40).
        augment: Whether to apply data augmentation.
    """

    def __init__(self, npy_files, labels, target_len=40, augment=False,
                 use_velocity=False):
        self.npy_files = npy_files
        self.labels = labels
        self.target_len = target_len
        self.augment = augment
        self.use_velocity = use_velocity

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        """
        Load and preprocess a single landmark sequence.

        Returns:
            tuple: (sequence_tensor, label_tensor)
                   sequence_tensor shape: (target_len, feature_dim)
                   label: scalar int
        """
        data = np.load(self.npy_files[idx]).astype(np.float32)
        data = resample_sequence(data, self.target_len)

        if self.augment:
            data = augment_sequence(data)

        # Append velocity (frame-to-frame delta) features: signs are defined
        # by motion, so deltas give the model a direct motion signal.
        if self.use_velocity:
            delta = np.diff(data, axis=0, prepend=data[:1])
            data = np.concatenate([data, delta], axis=1)

        return (
            torch.tensor(data, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def discover_classes(landmarks_dir="landmarks", exclude=("Extra", "extra")):
    """
    Discover sign classes from the directory structure.

    Args:
        landmarks_dir: Root directory containing class subdirectories with .npy files.
        exclude: Class folder names to exclude (junk/non-sign folders).

    Returns:
        tuple: (class_folders, class_to_idx)
            class_folders: Sorted list of Path objects for class directories.
            class_to_idx: Dict mapping class name to integer index.
    """
    root = Path(landmarks_dir)
    class_folders = sorted([
        cf for cf in root.rglob("*")
        if cf.is_dir() and list(cf.glob("*.npy")) and cf.name not in exclude
    ])
    # Assign sequential indices to unique class names (handles duplicate
    # folder names by merging them into a single class)
    class_to_idx = {}
    for cf in class_folders:
        if cf.name not in class_to_idx:
            class_to_idx[cf.name] = len(class_to_idx)
    return class_folders, class_to_idx


def load_dataset(landmarks_dir="landmarks"):
    """
    Load all .npy files and their labels from the landmarks directory.

    Args:
        landmarks_dir: Root directory containing class subdirectories.

    Returns:
        tuple: (files, labels, class_to_idx)
            files: List of Path objects to .npy files.
            labels: List of integer labels.
            class_to_idx: Dict mapping class name to index.
    """
    class_folders, class_to_idx = discover_classes(landmarks_dir)

    files, labels = [], []
    for cf in class_folders:
        for f in cf.glob("*.npy"):
            files.append(f)
            labels.append(class_to_idx[cf.name])

    print(f"Found {len(files)} landmark files across {len(class_folders)} classes")
    print(f"Classes: {list(class_to_idx.keys())}")

    return files, labels, class_to_idx
