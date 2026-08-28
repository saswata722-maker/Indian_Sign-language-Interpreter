"""Extract landmark sequences from all INCLUDE videos into .npy files.

Walks raw_data/<Category_Part>/<Category>/<NN. Word>/<video>.MOV and writes
data/landmarks/<Category>/<NN. Word>/<video>.npy with shape (T, D) float32.

Resumable: existing .npy files are skipped, so the job can be stopped and
restarted (run overnight).

Usage:
    python preprocessing/extract_landmarks.py --raw_dir raw_data --out_dir data/landmarks
    python preprocessing/extract_landmarks.py --limit_videos 5          # quick test
    python preprocessing/extract_landmarks.py --only_category Greetings --no_face
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.landmark_extractor import LandmarkExtractor, feature_dim  # noqa: E402

VIDEO_EXTS = {".MOV", ".MP4", ".AVI", ".MKV", ".MOV"}


def find_videos(raw_dir: Path, only_category: str | None):
    """Yield (video_path, word_dir) pairs for every video under raw_dir."""
    for video in sorted(raw_dir.rglob("*")):
        if not video.is_file() or video.suffix.upper() not in VIDEO_EXTS:
            continue
        word_dir = video.parent  # e.g. .../Greetings/48. Hello
        if only_category and only_category.lower() not in str(word_dir).lower():
            continue
        yield video, word_dir


def resolve_raw_dir(raw_dir: Path) -> Path:
    """Return the directory that actually holds the INCLUDE category folders.

    Handles both layouts:
      data/raw_data/<CategoryPart>/<Category>/<NN. Word>/<video>
      data/raw_data/raw_data/<CategoryPart>/<Category>/<NN. Word>/<video>
    (the nested `raw_data` arises when the INCLUDE zip is unpacked inside
    data/raw_data).
    """
    nested = raw_dir / "raw_data"
    if nested.is_dir() and not any(p.is_file() for p in raw_dir.iterdir()):
        return nested
    return raw_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw_dir", default="data/raw_data", type=Path)
    ap.add_argument("--out_dir", default="data/landmarks", type=Path)
    ap.add_argument("--include_face", dest="include_face", action="store_true", default=True)
    ap.add_argument("--no_face", dest="include_face", action="store_false",
                    help="Exclude 468 facial landmarks (D=234 instead of 1632)")
    ap.add_argument("--model_complexity", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--max_frames", type=int, default=200,
                    help="Cap frames per video (INCLUDE signs are short)")
    ap.add_argument("--only_category", type=str, default=None,
                    help="Substring filter on category, e.g. 'Greetings'")
    ap.add_argument("--limit_videos", type=int, default=None,
                    help="Process only the first N videos (smoke test)")
    args = ap.parse_args()
    args.raw_dir = resolve_raw_dir(args.raw_dir)

    jobs = list(find_videos(args.raw_dir, args.only_category))
    if args.limit_videos:
        jobs = jobs[: args.limit_videos]
    if not jobs:
        print(f"No videos found under {args.raw_dir}")
        return

    print(f"{len(jobs)} videos | feature dim D = {feature_dim(args.include_face)}")
    extractor = LandmarkExtractor(
        include_face=args.include_face, model_complexity=args.model_complexity
    )

    n_done = n_skip = n_fail = 0
    t0 = time.time()
    with extractor:
        for video, word_dir in tqdm(jobs, desc="extracting", unit="video"):
            out_path = args.out_dir / word_dir.relative_to(args.raw_dir) / (
                video.stem + ".npy"
            )
            if out_path.exists():  # resumable
                n_skip += 1
                continue
            try:
                seq = extractor.extract_video(str(video), max_frames=args.max_frames)
            except Exception as exc:  # noqa: BLE001
                print(f"\nFAILED {video}: {exc}")
                n_fail += 1
                continue
            if seq.shape[0] == 0:
                print(f"\nEMPTY {video}")
                n_fail += 1
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, seq.astype(np.float32))
            n_done += 1

    dt = time.time() - t0
    print(f"\nDone in {dt/60:.1f} min | extracted {n_done} | skipped {n_skip} "
          f"| failed {n_fail} | out: {args.out_dir}")


if __name__ == "__main__":
    main()
