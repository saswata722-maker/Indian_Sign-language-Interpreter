"""
Training Loop Module
Trains the HybridSignModel on extracted landmarks.

Features:
  - Stratified train/validation split
  - Label smoothing
  - AdamW optimizer with weight decay
  - Learning rate scheduling (CosineAnnealingWarmRestarts)
  - Early stopping
  - Model checkpoint saving (best + periodic)
  - Gradient clipping
  - Mixed precision training (fp16)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml
import os
import time
import sys

# Add project root to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset import ISLLandmarkDataset, load_dataset
from training.model import HybridSignModel

def train_isl(landmarks_dir="landmarks", config_path="config.yaml"):
    """
    Train the ISL recognition model.

    Args:
        landmarks_dir: Directory containing extracted landmarks.
        config_path: Path to training configuration YAML file.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

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
    print(f"Train: {len(train_f)}, Validation: {len(val_f)}")

    train_loader = DataLoader(
        ISLLandmarkDataset(train_f, train_l,
                           target_len=config['preprocessing']['target_seq_len'],
                           augment=True),
        batch_size=config['training']['batch_size'],
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        ISLLandmarkDataset(val_f, val_l,
                           target_len=config['preprocessing']['target_seq_len'],
                           augment=False),
        batch_size=config['training']['batch_size'],
        shuffle=False, num_workers=0, pin_memory=True
    )


    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Model
    model = HybridSignModel(
        input_dim=config['model']['input_dim'],
        num_classes=len(class_to_idx),
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_transformer_layers=config['model']['num_transformer_layers'],
        dropout=config['model']['dropout']
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss, optimizer
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config['training']['label_smoothing']
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # LR scheduler: Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # Early stopping
    patience = config['training'].get('early_stopping_patience', 15)
    min_delta = config['training'].get('early_stopping_min_delta', 0.001)
    epochs_no_improve = 0
    early_stop = False

    # Gradient clipping
    max_grad_norm = config['training'].get('max_grad_norm', 1.0)

    # Mixed precision training (fp16) for faster GPU training
    use_amp = device.type == "cuda" and config['training'].get('use_amp', True)
    torch.backends.cudnn.benchmark = True
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Model saving
    model_save_dir = Path(config['model_save_dir'])
    model_save_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, config['training']['num_epochs'] + 1):
        if early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch - 1}")
            break

        # --- Training phase ---
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device.type, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(y)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += len(y)

        train_acc = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # --- Validation phase ---
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast(device.type, enabled=use_amp):
                    out = model(x)
                v_correct += (out.argmax(dim=1) == y).sum().item()
                v_total += len(y)

        val_acc = v_correct / v_total if v_total > 0 else 0.0

        # Logging
        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:03d}/{config['training']['num_epochs']} | "
                  f"Loss: {avg_loss:.4f} | Train: {train_acc*100:.1f}% | "
                  f"Val: {val_acc*100:.1f}% | LR: {current_lr:.2e} | "
                  f"Time: {elapsed/60:.1f}m")

        # Save best model / early stopping check
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'class_to_idx': class_to_idx,
                'config': config,
            }, model_save_dir / "best_model.pth")
            print(f"  -> Saved best model (val_acc: {val_acc*100:.1f}%)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                early_stop = True
                print(f"  -> No improvement for {patience} epochs, stopping...")

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'class_to_idx': class_to_idx,
                'config': config,
            }, model_save_dir / f"checkpoint_epoch_{epoch}.pth")

    elapsed_total = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"  Best validation accuracy: {best_val_acc*100:.1f}%")
    print(f"  Total time: {elapsed_total/60:.1f} minutes")
    return model, class_to_idx


if __name__ == "__main__":
    train_isl("landmarks", "config.yaml")

