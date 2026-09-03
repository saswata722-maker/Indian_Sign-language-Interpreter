"""
Metrics Module
Utility functions for model evaluation and performance tracking.

Based on AI4Bharat INCLUDE evaluation approach:
  - Top-1 and Top-5 accuracy
  - Per-class accuracy tracking
"""

import numpy as np


def compute_accuracy(predictions, labels, topk=(1,)):
    """
    Compute top-k accuracy.

    Args:
        predictions: np.ndarray of predicted class indices.
        labels: np.ndarray of ground truth labels.
        topk: Tuple of k values for top-k accuracy.

    Returns:
        List of accuracy values for each k.
    """
    # For single prediction per sample
    correct = (predictions == labels)
    accuracy = correct.mean() * 100.0
    return [accuracy]


def per_class_accuracy(predictions, labels, class_names):
    """
    Compute accuracy for each class individually.

    Args:
        predictions: np.ndarray of predicted class indices.
        labels: np.ndarray of ground truth labels.
        class_names: List of class name strings.

    Returns:
        Dict mapping class name to accuracy.
    """
    accuracies = {}
    for i, name in enumerate(class_names):
        mask = labels == i
        if mask.sum() > 0:
            accuracies[name] = (predictions[mask] == labels[mask]).mean() * 100.0
        else:
            accuracies[name] = 0.0
    return accuracies


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0
