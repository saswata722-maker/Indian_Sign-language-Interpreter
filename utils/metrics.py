"""Metric helpers: accuracy, confusion matrix, per-class reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)


def confusion(y_true, y_pred, n_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


def report(y_true, y_pred, class_names: list[str]) -> str:
    return classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, zero_division=0,
    )


def topk_accuracy(y_true, probs: np.ndarray, k: int = 3) -> float:
    return float(
        top_k_accuracy_score(y_true, probs, k=k, labels=list(range(probs.shape[1])))
    )


def plot_confusion(cm: np.ndarray, class_names: list[str], out_path: str | Path,
                   normalize: bool = True) -> None:
    if normalize:
        cm = cm.astype(np.float64)
        cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1e-9, None)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix" + (" (row-normalized)" if normalize else ""))
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
