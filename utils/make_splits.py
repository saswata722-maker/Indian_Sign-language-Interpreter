"""
Data Splitting Module
Utility to create train/validation/test splits from the dataset.

Based on AI4Bharat INCLUDE's stratified splitting approach to maintain
class balance across all splits.
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json


def make_splits(landmarks_dir="landmarks", val_split=0.15, test_split=0.15,
                random_seed=42, output_file="splits.json"):
    """
    Create stratified train/validation/test splits from landmark files.

    Args:
        landmarks_dir: Directory containing extracted landmarks.
        val_split: Fraction of data for validation.
        test_split: Fraction of data for testing.
        random_seed: Random seed for reproducibility.
        output_file: Output JSON file path for the splits.

    Returns:
        Dict containing file paths and labels for each split.
    """
    root = Path(landmarks_dir)

    # Discover classes and files
    class_folders = sorted([
        cf for cf in root.rglob("*")
        if cf.is_dir() and list(cf.glob("*.npy"))
    ])
    class_to_idx = {cf.name: i for i, cf in enumerate(class_folders)}

    all_files = []
    all_labels = []
    for cf in class_folders:
        for f in cf.glob("*.npy"):
            all_files.append(str(f))
            all_labels.append(class_to_idx[cf.name])

    print(f"Found {len(all_files)} files across {len(class_folders)} classes")

    # First split: separate test set
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        all_files, all_labels,
        test_size=test_split,
        stratify=all_labels,
        random_state=random_seed
    )

    # Second split: separate validation from training
    adjusted_val_split = val_split / (1 - test_split)
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels,
        test_size=adjusted_val_split,
        stratify=train_val_labels,
        random_state=random_seed
    )

    splits = {
        'train': {'files': train_files, 'labels': train_labels},
        'val': {'files': val_files, 'labels': val_labels},
        'test': {'files': test_files, 'labels': test_labels},
        'class_to_idx': class_to_idx,
    }

    # Save splits
    with open(output_file, 'w') as f:
        json.dump(splits, f, indent=2)

    print(f"Split sizes — Train: {len(train_files)}, "
          f"Val: {len(val_files)}, Test: {len(test_files)}")
    print(f"Splits saved to {output_file}")

    return splits


if __name__ == "__main__":
    make_splits()
