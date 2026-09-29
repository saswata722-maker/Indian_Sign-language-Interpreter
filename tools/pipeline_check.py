"""Pipeline verification: run real INCLUDE videos through the SAME code path
used by live inference (extractor -> 64-frame window -> velocity -> model).

Purpose: separate "live code bug" from "domain/signer gap".
Also tests horizontal mirroring (handedness trap) and webcam-like
degradation (lower res, dimmer, blurrier).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
import yaml

from preprocessing.extract_landmarks import LandmarkExtractor
from training.dataset import resample_sequence
from training.model import HybridSignModel

CONFIG = yaml.safe_load(open("config.yaml"))
SEQ = CONFIG["preprocessing"]["target_seq_len"]
USE_VEL = CONFIG["preprocessing"].get("use_velocity", False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load("models/best_model.pth", map_location=device)
mcfg = ckpt["config"]
idx_to_class = {v: k for k, v in ckpt["class_to_idx"].items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

model = HybridSignModel(
    input_dim=mcfg["model"]["input_dim"], num_classes=len(class_names),
    d_model=mcfg["model"]["d_model"], nhead=mcfg["model"]["nhead"],
    num_transformer_layers=mcfg["model"]["num_transformer_layers"],
    dropout=mcfg["model"]["dropout"]).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

ex = LandmarkExtractor(include_face=False, model_complexity=1)


def features_from_video(path, variant="original"):
    cap = cv2.VideoCapture(str(path))
    feats, hand_ok = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if variant == "flipped":
            frame = cv2.flip(frame, 1)
        elif variant == "webcam_like":
            frame = cv2.resize(frame, (640, 480))
            frame = (frame * 0.6).astype(np.uint8)
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
        f = ex.process_frame(frame)
        feats.append(f)
        if f[-2] > 0.5 or f[-1] > 0.5:
            hand_ok += 1
    cap.release()
    if not feats:
        return None, 0.0
    return resample_sequence(np.stack(feats), SEQ), hand_ok / len(feats)


def predict(seq):
    x = seq
    if USE_VEL:
        x = np.concatenate([seq, np.diff(seq, axis=0, prepend=seq[:1])], axis=1)
    with torch.no_grad():
        p = torch.softmax(model(torch.tensor(x, dtype=torch.float32)
                                .unsqueeze(0).to(device)), dim=1).cpu().numpy()[0]
    return int(p.argmax()), float(p.max())


raw = Path("raw_data")
folders = sorted([p for p in raw.rglob("*") if p.is_dir() and list(p.glob("*.MOV"))])[:6]
clips = [(v, f.name) for f in folders for v in sorted(f.glob("*.MOV"))[:2]]

print("=" * 66)
print("PIPELINE VERIFICATION  (INCLUDE videos through the live code path)")
print("=" * 66)
for variant in ["original", "flipped", "webcam_like"]:
    correct, total, confs = 0, 0, []
    for v, label in clips:
        seq, _ = features_from_video(v, variant)
        if seq is None:
            continue
        idx, conf = predict(seq)
        correct += class_names[idx].strip().lower() == label.strip().lower()
        total += 1
        confs.append(conf)
    mean_conf = np.mean(confs) if confs else 0.0
    pct = 100 * correct / max(total, 1)
    print(f"{variant:12s}: {correct}/{total} correct ({pct:3.0f}%)  mean conf {mean_conf:.1%}")

print("")
print("Per-clip detail (original):")
for v, label in clips:
    seq, hand_frac = features_from_video(v, "original")
    if seq is None:
        continue
    idx, conf = predict(seq)
    pred = class_names[idx]
    ok = "OK  " if pred.strip().lower() == label.strip().lower() else "MISS"
    print(f"  {ok} {label[:22]:22s} -> {pred[:22]:22s} conf={conf:5.1%} hands={hand_frac:.0%}")
ex.close()
