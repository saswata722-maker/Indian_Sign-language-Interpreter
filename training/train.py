"""Train the hybrid LSTM+Transformer on landmark sequences.

Usage:
    python training/train.py --config training/config.yaml
    python training/train.py --config training/config.yaml --include 10   # sanity run
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.landmark_extractor import feature_dim  # noqa: E402
from models.model import HybridLSTMTransformer  # noqa: E402
from training.dataset import LandmarkDataset  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp: bool):
    model.train()
    tot_loss = tot_correct = tot_n = 0
    for x, mask, y in loader:
        x, mask, y = x.to(device, non_blocking=True), mask.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits = model(x, mask)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        tot_loss += loss.item() * y.size(0)
        tot_correct += (logits.argmax(1) == y).sum().item()
        tot_n += y.size(0)
    return tot_loss / max(tot_n, 1), tot_correct / max(tot_n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    tot_loss = tot_correct = tot_n = 0
    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        logits = model(x, mask)
        tot_loss += criterion(logits, y).item() * y.size(0)
        tot_correct += (logits.argmax(1) == y).sum().item()
        tot_n += y.size(0)
    return tot_loss / max(tot_n, 1), tot_correct / max(tot_n, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--include", type=int, default=None,
                    help="Sanity run: keep only classes with label_idx < N")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    set_seed(t["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    splits = Path(d["splits_dir"])
    train_ds = LandmarkDataset(splits / "train.csv", d["seq_len"], d["include_face"],
                               train=True, seed=t["seed"])
    val_ds = LandmarkDataset(splits / "val.csv", d["seq_len"], d["include_face"],
                             train=False)

    def filter_incl(ds, n):
        keep = [i for i, y in enumerate(ds.labels) if y < n]
        ds.paths = [ds.paths[i] for i in keep]
        ds.labels = [ds.labels[i] for i in keep]

    if args.include:
        filter_incl(train_ds, args.include)
        filter_incl(val_ds, args.include)
    print(f"Train {len(train_ds)} | Val {len(val_ds)} "
          f"| classes {len(set(train_ds.labels))}")

    def dl(ds, sh):
        return DataLoader(ds, batch_size=t["batch_size"], shuffle=sh,
                          num_workers=t["num_workers"], pin_memory=True,
                          drop_last=sh)

    train_loader, val_loader = dl(train_ds, True), dl(val_ds, False)

    input_dim = feature_dim(d["include_face"])
    num_classes = d["num_classes"] if not args.include else args.include
    model = HybridLSTMTransformer(
        input_dim=input_dim, num_classes=num_classes,
        d_model=m["d_model"], lstm_layers=m["lstm_layers"],
        transformer_layers=m["transformer_layers"], nhead=m["nhead"],
        ff_dim=m["ff_dim"], dropout=m["dropout"],
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,} "
          f"| input_dim={input_dim}")

    criterion = nn.CrossEntropyLoss(label_smoothing=t["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"],
                                  weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t["scheduler_t_max"])
    scaler = torch.amp.GradScaler("cuda", enabled=t["amp"] and device.type == "cuda")

    ckpt_dir = Path(t["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(t["runs_dir"]); runs_dir.mkdir(parents=True, exist_ok=True)

    history, best_val, bad_epochs = [], -1.0, 0
    for epoch in range(1, t["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                          optimizer, scaler, device, t["amp"])
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})
        print(f"epoch {epoch:3d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f} | {time.time()-t0:.1f}s")

        ckpt_common = {"model": model.state_dict(), "config": cfg, "epoch": epoch,
                       "input_dim": input_dim, "num_classes": num_classes}
        torch.save({**ckpt_common, "val_acc": va_acc}, ckpt_dir / "last.pt")
        if va_acc > best_val:
            best_val, bad_epochs = va_acc, 0
            torch.save({**ckpt_common, "val_acc": va_acc, "classes": train_ds.classes},
                       ckpt_dir / "best.pt")
            print(f"  -> new best ({va_acc:.3f}) saved")
        else:
            bad_epochs += 1
            if bad_epochs >= t["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    with (runs_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader(); w.writerows(history)
    print(f"Best val acc: {best_val:.3f} | checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
