"""Evaluate a trained checkpoint on the test split.

Usage:
    python training/evaluate.py --checkpoint models/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.model import HybridLSTMTransformer  # noqa: E402
from training.dataset import LandmarkDataset  # noqa: E402
from training.train import evaluate  # noqa: E402
from utils.metrics import (  # noqa: E402
    confusion, plot_confusion, report, topk_accuracy,
)


@torch.no_grad()
def collect_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for x, mask, y in loader:
        logits = model(x.to(device), mask.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="models/checkpoints/best.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    d = cfg["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = LandmarkDataset(Path(d["splits_dir"]) / f"{args.split}.csv",
                         d["seq_len"], d["include_face"], train=False)
    classes = ckpt.get("classes") or ds.classes
    print(f"{args.split}: {len(ds)} samples | {len(classes)} classes")

    model = HybridLSTMTransformer(
        input_dim=ckpt["input_dim"], num_classes=ckpt["num_classes"],
        d_model=cfg["model"]["d_model"], lstm_layers=cfg["model"]["lstm_layers"],
        transformer_layers=cfg["model"]["transformer_layers"],
        nhead=cfg["model"]["nhead"], ff_dim=cfg["model"]["ff_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"])

    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    criterion = nn.CrossEntropyLoss()
    loss, acc = evaluate(model, loader, criterion, device)
    probs, labels = collect_probs(model, loader, device)

    preds = probs.argmax(1)
    print(f"\nloss {loss:.4f} | top-1 acc {acc:.4f} | top-3 acc "
          f"{topk_accuracy(labels, probs, k=3):.4f}\n")
    print(report(labels, preds, classes))

    cm = confusion(labels, preds, len(classes))
    out = Path("runs") / f"confusion_{args.split}.png"
    plot_confusion(cm, classes, out)
    print(f"Confusion matrix -> {out}")


if __name__ == "__main__":
    main()
