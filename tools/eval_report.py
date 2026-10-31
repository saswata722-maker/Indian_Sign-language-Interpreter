"""
Standalone evaluation report generator for ISL-Sense.

Produces a classification report and confusion matrix (saved as PNG + CSV)
from the trained checkpoint over the landmarks directory. Does NOT modify
any file in the main codebase — reads only from models/, landmarks/, config.yaml.

Usage:
    python tools/eval_report.py [--checkpoint models/best_model.pth] \
        [--landmarks landmarks] [--config config.yaml] \
        [--out eval_report]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from training.dataset import ISLLandmarkDataset, load_dataset
from training.model import HybridSignModel
from utils.make_splits import select_split


def load_model(checkpoint_path, config_path, device):
    """Load model + class names from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_to_idx = checkpoint.get("class_to_idx", {})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]

    # Try config from checkpoint, fall back to config file
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model_config = checkpoint.get("config", config)

    model = HybridSignModel(
        input_dim=model_config["model"]["input_dim"],
        num_classes=len(class_names),
        d_model=model_config["model"]["d_model"],
        nhead=model_config["model"]["nhead"],
        num_transformer_layers=model_config["model"]["num_transformer_layers"],
        dropout=model_config["model"]["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names, config, model_config


def evaluate(model, loader, device, class_names, threshold=0.35):
    """Run inference; return labels, preds, probs, rejected mask."""
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            best_prob, best_idx = probs.max(dim=1)
            all_preds.extend(best_idx.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(best_prob.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    rejected = all_probs < threshold
    return all_labels, all_preds, all_probs, rejected


def classification_report(labels, preds, class_names):
    """Per-class precision/recall/f1/support (sklearn-style, no sklearn dep)."""
    n_classes = len(class_names)
    report = {}
    for i, name in enumerate(class_names):
        tp = int(((preds == i) & (labels == i)).sum())
        fp = int(((preds == i) & (labels != i)).sum())
        fn = int(((labels == i) & (preds != i)).sum())
        support = int((labels == i).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        report[name] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": support,
        }
    # macro averages
    keys = ["precision", "recall", "f1"]
    macro = {k: float(np.mean([report[n][k] for n in class_names])) for k in keys}
    return report, macro


def confusion_matrix(labels, preds, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm, class_names, out_path, normalize=True):
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(12, n * 0.25), max(10, n * 0.25)))
    data = cm.astype(float) / cm.sum(axis=1, keepdims=True) if normalize else cm
    data = np.nan_to_num(data)
    im = ax.imshow(data, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=5)
    ax.set_yticklabels(class_names, fontsize=5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" +
                 (" (normalized by row)" if normalize else " (counts)"))
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/best_model.pth")
    ap.add_argument("--landmarks", default="landmarks")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="eval_report")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--threshold", type=float, default=0.35)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Landmarks:  {args.landmarks}")

    model, class_names, config, model_config = load_model(
        args.checkpoint, args.config, device)
    print(f"Classes: {len(class_names)}")

    use_velocity = model_config.get("preprocessing", {}).get("use_velocity", False)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    files, labels = select_split(checkpoint, args.landmarks, args.split)
    loader = DataLoader(
        ISLLandmarkDataset(files, labels,
                           target_len=model_config["preprocessing"]["target_seq_len"],
                           augment=False, use_velocity=use_velocity,
                           feature_version=model_config['preprocessing'].get('feature_version', 1)),
        batch_size=config["training"]["batch_size"], shuffle=False)

    all_labels, all_preds, all_probs, rejected = evaluate(
        model, loader, device, class_names, threshold=args.threshold)

    n_total = len(all_labels)
    n_accepted = int((~rejected).sum())
    acc_all = float((all_preds == all_labels).mean())
    acc_acc = float(((all_preds == all_labels) & (~rejected)).sum() / max(n_accepted, 1))
    eff_acc = float(((all_preds == all_labels) & (~rejected)).sum() / n_total)
    rej_rate = float(rejected.mean())

    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total clips:     {n_total}")
    print(f"Accepted:        {n_accepted} ({100*(1-rej_rate):.1f}%)")
    print(f"Rejected:        {int(rejected.sum())} ({100*rej_rate:.1f}%)")
    print(f"Accuracy (all):  {100*acc_all:.1f}%")
    print(f"Accuracy (acc):  {100*acc_acc:.1f}%")
    print(f"Effective acc:   {100*eff_acc:.1f}%")

    report, macro = classification_report(all_labels, all_preds, class_names)

    # Save metrics JSON
    metrics = {
        "split": args.split,
        "n_total": n_total,
        "n_accepted": n_accepted,
        "rejection_rate": rej_rate,
        "accuracy_all": acc_all,
        "accuracy_accepted": acc_acc,
        "effective_accuracy": eff_acc,
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "macro_f1": macro["f1"],
        "per_class": report,
        "threshold": args.threshold,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save per-class CSV
    with open(out_dir / "classification_report.csv", "w") as f:
        f.write("class,precision,recall,f1,support\n")
        for name in class_names:
            r = report[name]
            f.write(f"{name},{r['precision']:.4f},{r['recall']:.4f},"
                    f"{r['f1']:.4f},{r['support']}\n")
        f.write(f"macro,{macro['precision']:.4f},{macro['recall']:.4f},"
                f"{macro['f1']:.4f},{n_total}\n")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, len(class_names))
    np.save(out_dir / "confusion_matrix.npy", cm)
    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix_counts.png",
                          normalize=False)

    # Top errors: pairs (true -> pred) with highest counts
    errors = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                errors.append((cm[i, j], class_names[i], class_names[j]))
    errors.sort(reverse=True)
    with open(out_dir / "top_errors.txt", "w") as f:
        f.write("Top misclassifications (count, true -> pred):\n")
        for count, t, p in errors[:30]:
            f.write(f"  {count:4d}  {t} -> {p}\n")

    print(f"\nReports saved to {out_dir}/:")
    print("  metrics.json          - aggregate metrics + per-class precision/recall/f1")
    print("  classification_report.csv - per-class CSV")
    print("  confusion_matrix.png  - normalized heatmap")
    print("  confusion_matrix_counts.png - raw counts heatmap")
    print("  top_errors.txt        - most-confused class pairs")
    print("  confusion_matrix.npy  - raw matrix")


if __name__ == "__main__":
    main()
