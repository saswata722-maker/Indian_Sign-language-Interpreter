"""Build the INCLUDE-50 label map and stratified train/val/test splits.

Scans data/landmarks for .npy sequences (class = immediate parent folder,
e.g. "48. Hello"), selects the `top_n` most populated classes, builds
utils/label_map.json and writes data/splits/{train,val,test}.csv with
columns: filepath,label,label_idx.

NOTE: the official Zenodo Train_Test_Split.zip is not downloadable (404),
so we use our own stratified 80/10/10 split at the video level. When the
full INCLUDE dataset is later used, the same script supports top_n=263.

Usage:
    python utils/make_splits.py --landmarks_dir data/landmarks --top_n 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def scan_landmarks(landmarks_dir: Path) -> dict[str, list[Path]]:
    """class word -> list of .npy paths.

    The class is normally the immediate parent folder (e.g. "48. Hello"), but
    INCLUDE ships some word folders with an `Extra/` subfolder containing
    additional footage of the SAME word. Those sequences must join the parent
    word's class, so the class name is taken from the grandparent folder when
    the immediate parent is named "Extra".
    """
    per_class: dict[str, list[Path]] = {}
    for npy in sorted(landmarks_dir.rglob("*.npy")):
        cls = npy.parent.name
        if cls.lower() == "extra":
            cls = npy.parent.parent.name
        per_class.setdefault(cls, []).append(npy)
    return per_class


def build_label_map(per_class: dict[str, list[Path]], top_n: int) -> tuple[list[str], dict[str, int]]:
    counts = Counter({w: len(v) for w, v in per_class.items()})
    chosen = sorted(counts, key=lambda w: (-counts[w], w))[:top_n]
    label_map = {i: w for i, w in enumerate(sorted(chosen))}
    label_inv = {w: i for i, w in label_map.items()}
    return chosen, label_inv, label_map


def stratified_split(paths: list[Path], ratios=(0.8, 0.1, 0.1),
                     rng: random.Random | None = None) -> dict[str, list[Path]]:
    rng = rng or random.Random(42)
    paths = sorted(paths)
    rng.shuffle(paths)
    n = len(paths)
    n_train = max(1, int(n * ratios[0]))
    n_val = max(1, int(n * ratios[1])) if n >= 3 else (1 if n == 2 else 0)
    return {
        "train": paths[:n_train],
        "val": paths[n_train:n_train + n_val],
        "test": paths[n_train + n_val:],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--landmarks_dir", type=Path, default=Path("data/landmarks"))
    ap.add_argument("--out_dir", type=Path, default=Path("data/splits"))
    ap.add_argument("--label_map_path", type=Path, default=Path("utils/label_map.json"))
    ap.add_argument("--top_n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    per_class = scan_landmarks(args.landmarks_dir)
    if not per_class:
        raise SystemExit(f"No .npy landmark files found under {args.landmarks_dir}")

    chosen, label_inv, label_map = build_label_map(per_class, args.top_n)
    print(f"{len(per_class)} classes available; using top {len(chosen)}:")
    for w in chosen:
        print(f"  {label_inv[w]:3d}  {w:<35s} {len(per_class[w])} videos")

    rng = random.Random(args.seed)
    rows = {"train": [], "val": [], "test": []}
    for w in chosen:
        for split, paths in stratified_split(per_class[w], rng=rng).items():
            for p in paths:
                rows[split].append((str(p).replace("\\", "/"), w, label_inv[w]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, items in rows.items():
        csv_path = args.out_dir / f"{split}.csv"
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("filepath,label,label_idx\n")
            for fp, w, idx in items:
                f.write(f'"{fp}","{w}",{idx}\n')
        print(f"{split}: {len(items)} -> {csv_path}")

    args.label_map_path.parent.mkdir(parents=True, exist_ok=True)
    args.label_map_path.write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    inv_path = args.label_map_path.with_name("label_map_inv.json")
    inv_path.write_text(
        json.dumps(label_inv, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"label map ({len(label_map)} classes) -> {args.label_map_path}")


if __name__ == "__main__":
    main()
