"""
Training Loop Module
Trains the HybridSignModel on extracted landmarks.

Features:
  - Stratified train/validation split
  - Label smoothing
  - AdamW optimizer with weight decay
  - Periodic validation evaluation
  - Model checkpoint saving
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml
import os

from training.dataset import ISLLandmarkDataset, load_dataset
from training.model import HybridSignModel


def train_isl(landmarks_dir="landmarks", config_path="config.yaml"):
    """
    Train the ISL recognition model.

    Args:
        landmarks_dir: Directory containing extracted landmarks.
        config_path: Path to training configuration YAML file.
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Discover and load dataset
    files, labels, class_to_idx = load_dataset(landmarks_dir)

    if len(files) == 0:
        print("No landmark files found! Run preprocessing/extract_landmarks.py first.")
        return

    # Stratified train/validation split
    train_f, val_f, train_l, val_l = train_test_split(
        files, labels,
        test_size=config['training']['val_split'],
        stratify=labels,
        random_state=config['training']['random_seed']
    )

    print(f"Train samples: {len(train_f)}, Validation samples: {len(val_f)}")

    # Data loaders
    train_loader = DataLoader(
        ISLLandmarkDataset(train_f, train_l,
                           target_len=config['preprocessing']['target_seq_len'],
                           augment=True),
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        ISLLandmarkDataset(val_f, val_l,
                           target_len=config['preprocessing']['target_seq_len'],
                           augment=False),
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    model = HybridSignModel(
        input_dim=config['model']['input_dim'],
        num_classes=len(class_folders := class_to_idx),
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_transformer_layers=config['model']['num_transformer_layers'],
        dropout=config['model']['dropout']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss, optimizer
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config['training']['label_smoothing']
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Training loop
    best_val_acc = 0.0
    model_save_dir = Path(config['model_save_dir'])
    model_save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config['training']['num_epochs'] + 1):
        # --- Training phase ---
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += len(y)

        train_acc = correct / total if total > 0 else 0.0

        # --- Validation phase ---
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                v_correct += (out.argmax(dim=1) == y).sum().item()
                v_total += len(y)

        val_acc = v_correct / v_total if v_total > 0 else 0.0

        # Logging
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{config['training']['num_epochs']} | "
                  f"Loss: {total_loss/total:.4f} | "
                  f"Train Acc: {train_acc*100:.1f}% | "
                  f"Val Acc: {val_acc*100:.1f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = model_save_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_to_idx': class_to_idx,
                'config': config,
            }, save_path)
            print(f"  → Saved best model (val_acc: {val_acc*100:.1f}%)")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc*100:.1f}%")
    return model, class_to_idx


if __name__ == "__main__":
    train_isl("landmarks", "config.yaml")
