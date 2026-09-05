"""
Full Dataset Evaluation Script
Analyzes the complete landmarks dataset and reports statistics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from collections import defaultdict


def evaluate_dataset(landmarks_dir="landmarks"):
    """Analyze the full dataset and report statistics."""
    root = Path(landmarks_dir)

    # Find all class folders
    class_folders = sorted([
        f for f in root.rglob("*")
        if f.is_dir() and list(f.glob("*.npy"))
    ])

    if not class_folders:
        print("No landmark files found!")
        return

    print("=" * 60)
    print("FULL DATASET EVALUATION")
    print("=" * 60)

    total_samples = 0
    class_counts = {}
    feature_dims = set()
    seq_lengths = []

    for cf in class_folders:
        npy_files = list(cf.glob("*.npy"))
        count = len(npy_files)
        class_counts[cf.name] = count
        total_samples += count

        # Sample a few files to check dimensions
        for f in npy_files[:3]:
            try:
                data = np.load(f, allow_pickle=True)
                feature_dims.add(data.shape[1])
                seq_lengths.append(data.shape[0])
            except Exception as e:
                print(f"  Warning: Could not load {f}: {e}")

    print(f"\nTotal classes: {len(class_folders)}")
    print(f"Total samples: {total_samples}")
    print(f"Feature dimensions: {feature_dims}")
    if seq_lengths:
        print(f"Sequence lengths: min={min(seq_lengths)}, max={max(seq_lengths)}, "
              f"mean={np.mean(seq_lengths):.1f}, median={np.median(seq_lengths):.0f}")

    print(f"\nClass distribution:")
    print(f"  Min samples/class: {min(class_counts.values())}")
    print(f"  Max samples/class: {max(class_counts.values())}")
    print(f"  Mean samples/class: {np.mean(list(class_counts.values())):.1f}")

    # Check for class imbalance
    counts = list(class_counts.values())
    imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
    print(f"  Imbalance ratio (max/min): {imbalance_ratio:.1f}")

    # Classes with very few samples
    min_threshold = 10
    small_classes = {k: v for k, v in class_counts.items() if v < min_threshold}
    if small_classes:
        print(f"\n⚠ Classes with < {min_threshold} samples (may need attention):")
        for name, count in sorted(small_classes.items(), key=lambda x: x[1])[:10]:
            print(f"  - {name}: {count} samples")
        if len(small_classes) > 10:
            print(f"  ... and {len(small_classes) - 10} more")

    # Sample a few files to verify data quality
    print(f"\nData quality check (sampling):")
    sample_issues = 0
    for cf in class_folders[:5]:
        for f in list(cf.glob("*.npy"))[:2]:
            try:
                data = np.load(f, allow_pickle=True)
                # Check for NaN/Inf
                if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                    print(f"  ⚠ NaN/Inf in {f.name}")
                    sample_issues += 1
                # Check for all zeros
                if np.all(data == 0):
                    print(f"  ⚠ All zeros in {f.name}")
                    sample_issues += 1
            except Exception as e:
                print(f"  ⚠ Error loading {f}: {e}")
                sample_issues += 1

    if sample_issues == 0:
        print("  ✓ No issues detected in sample")

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)
    if imbalance_ratio > 5:
        print("• High class imbalance detected — consider weighted sampling or oversampling")
    if feature_dims and min(feature_dims) != max(feature_dims):
        print("• Inconsistent feature dimensions — ensure all videos extracted with same settings")
    if min(counts) < 5:
        print("• Some classes have very few samples — consider removing or augmenting")
    print("• Use stratified train/val split to preserve class balance")
    print("• Monitor per-class accuracy during training")

    return class_counts


if __name__ == "__main__":
    evaluate_dataset()
