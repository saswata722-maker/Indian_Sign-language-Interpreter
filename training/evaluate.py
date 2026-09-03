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
    loader = DataLoader(
        ISLLandmarkDataset(files, labels,
                           target_len=config['preprocessing']['target_seq_len'],
                           augment=False),
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


if __name__ == "__main__":
    evaluate_model("models/best_model.pth", "landmarks", "config.yaml")
