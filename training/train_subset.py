"""
Subset Training Script for Quick Verification
Trains on the Seasons_1of1 subset to verify the pipeline works end-to-end.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from training.dataset import ISLLandmarkDataset
from training.model import HybridSignModel


def train_subset(subset_path="landmarks/Seasons_1of1/Seasons", epochs=30):
    """Train on a small subset to verify the pipeline."""
    root = Path(subset_path)
    if not root.exists():
        print(f"Subset path not found: {root}")
        print("Make sure landmark extraction has been run.")
        return

    # Find all sign class directories
    class_folders = sorted([cf for cf in root.iterdir() if cf.is_dir() and list(cf.glob("*.npy"))])
    class_to_idx = {cf.name: i for i, cf in enumerate(class_folders)}

    if len(class_folders) == 0:
        print(f"No class folders with .npy files found in {root}")
        return

    files, labels = [], []
    for cf in class_folders:
        for f in cf.glob("*.npy"):
            files.append(f)
            labels.append(class_to_idx[cf.name])

    print(f"Found {len(files)} samples across {len(class_folders)} classes:")
    for cf in class_folders:
        count = len(list(cf.glob("*.npy")))
        print(f"  - {cf.name}: {count} samples")

    # Stratified train/validation split
    train_f, val_f, train_l, val_l = train_test_split(
        files, labels, test_size=0.25, stratify=labels, random_state=42
    )
    print(f"\nTrain: {len(train_f)} | Validation: {len(val_f)}")

    train_loader = DataLoader(
        ISLLandmarkDataset(train_f, train_l, target_len=40, augment=True),
        batch_size=8, shuffle=True
    )
    val_loader = DataLoader(
        ISLLandmarkDataset(val_f, val_l, target_len=40, augment=False),
        batch_size=8, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Verify input dimensions from first batch
    sample_x, sample_y = next(iter(train_loader))
    input_dim = sample_x.shape[2]
    print(f"Input dim: {input_dim} | Sequence len: {sample_x.shape[1]}")

    model = HybridSignModel(
        input_dim=input_dim,
        num_classes=len(class_folders),
        d_model=64,
        nhead=4,
        num_transformer_layers=1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        total_loss, correct, total = 0, 0, 0
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

        train_acc = correct / total * 100
        avg_loss = total_loss / total

        # Validation
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                v_correct += (out.argmax(dim=1) == y).sum().item()
                v_total += len(y)

        val_acc = v_correct / v_total * 100

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs} | Loss: {avg_loss:.4f} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | Best: {best_val_acc:.1f}%")

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.1f}%")

    if best_val_acc > 80:
        print("Pipeline verified successfully! Ready to scale up to full dataset.")
    elif best_val_acc > 50:
        print("Pipeline working. May need tuning for better accuracy.")
    else:
        print("Low accuracy — check data quality or increase epochs.")


if __name__ == "__main__":
    train_subset()
