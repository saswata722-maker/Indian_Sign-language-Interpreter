"""
Evaluation Module
Evaluates trained model on validation/test set with detailed metrics.

Based on AI4Bharat INCLUDE evaluation approach.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import yaml

from training.dataset import ISLLandmarkDataset, load_dataset
from training.model import HybridSignModel


def evaluate_model(model_path, landmarks_dir="landmarks", config_path="config.yaml"):
    """
    Evaluate a trained model on the dataset.

    Args:
        model_path: Path to the saved model checkpoint.
        landmarks_dir: Directory containing extracted landmarks.
        config_path: Path to training configuration YAML file.
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load model checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Initialize model
    model = HybridSignModel(
        input_dim=config['model']['input_dim'],
        num_classes=len(class_to_idx),
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_transformer_layers=config['model']['num_transformer_layers'],
        dropout=config['model']['dropout']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load dataset
    files, labels, _ = load_dataset(landmarks_dir)

    # Create dataloader
    use_velocity = config['preprocessing'].get('use_velocity', False)
    loader = DataLoader(
        ISLLandmarkDataset(files, labels,
                           target_len=config['preprocessing']['target_seq_len'],
                           augment=False,
                           use_velocity=use_velocity),
        batch_size=config['training']['batch_size'],
        shuffle=False
    )

    # Evaluation
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    accuracy = (all_preds == all_labels).mean()
    print(f"\n{'='*50}")
    print(f"Evaluation Results")
    print(f"{'='*50}")
    print(f"Overall Accuracy: {accuracy*100:.1f}%")
    print(f"Total samples: {len(all_labels)}")
    print(f"\nClassification Report:")

    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    return accuracy, all_preds, all_labels


def evaluate_live_style(model_path, landmarks_dir="landmarks", config_path="config.yaml"):
    """
    Evaluate mimicking live inference post-processing: the confidence
    threshold rejects low-confidence predictions (no label shown).

    Note: the live majority-vote (smoothing_window) operates over
    consecutive sliding windows of a continuous stream, which cannot be
    reproduced from isolated per-class test clips. The dominant effect of
    live post-processing is the confidence threshold, which is measured here.

    Reports:
      - accuracy (all, argmax): baseline, no threshold
      - accuracy (accepted only): correct / clips above threshold
      - effective accuracy: correct / total (rejected counted as incorrect)
      - rejection rate: fraction of clips below threshold
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    class_to_idx = checkpoint['class_to_idx']

    model = HybridSignModel(
        input_dim=config['model']['input_dim'],
        num_classes=len(class_to_idx),
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_transformer_layers=config['model']['num_transformer_layers'],
        dropout=config['model']['dropout']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    files, labels, _ = load_dataset(landmarks_dir)
    use_velocity = config['preprocessing'].get('use_velocity', False)
    loader = DataLoader(
        ISLLandmarkDataset(files, labels,
                           target_len=config['preprocessing']['target_seq_len'],
                           augment=False,
                           use_velocity=use_velocity),
        batch_size=config['training']['batch_size'],
        shuffle=False
    )

    threshold = config['inference']['confidence_threshold']
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1)
            best_prob, best_idx = probs.max(dim=1)
            all_preds.extend(best_idx.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(best_prob.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accepted = all_probs >= threshold
    n_total = len(all_labels)
    n_accepted = int(accepted.sum())
    n_rejected = n_total - n_accepted
    correct_all = int((all_preds == all_labels).sum())
    correct_acc = int(((all_preds == all_labels) & accepted).sum())

    acc_all = correct_all / n_total if n_total else 0.0
    acc_accepted = correct_acc / n_accepted if n_accepted else 0.0
    eff_acc = correct_acc / n_total if n_total else 0.0
    rej_rate = n_rejected / n_total if n_total else 0.0

    print(f"\n{'='*50}")
    print(f"Live-Style Evaluation (threshold={threshold})")
    print(f"{'='*50}")
    print(f"Total clips: {n_total}")
    print(f"Accepted: {n_accepted} ({n_accepted/n_total:.1%})  "
          f"Rejected: {n_rejected} ({rej_rate:.1%})")
    print(f"Accuracy (all, argmax):        {acc_all*100:.1f}%")
    print(f"Accuracy (accepted only):      {acc_accepted*100:.1f}%")
    print(f"Effective accuracy (rej=wrong): {eff_acc*100:.1f}%")
    print("Note: majority-vote smoothing not simulated (needs consecutive "
          "stream windows, not isolated clips).")
    return acc_accepted, eff_acc, rej_rate


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "live":
        evaluate_live_style("models/best_model.pth", "landmarks", "config.yaml")
    else:
        evaluate_model("models/best_model.pth", "landmarks", "config.yaml")
