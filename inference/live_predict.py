"""Real-time webcam ISL recognizer with subtitle overlay.

Opens the webcam, buffers a sliding window of landmark feature frames
(extracted with the SAME LandmarkExtractor used for training), runs the
trained hybrid LSTM+Transformer every `stride` frames, smooths predictions
with a moving-majority vote, and renders the predicted word as a subtitle
on the video feed.

Usage:
    python inference/live_predict.py --checkpoint models/checkpoints/best.pt
    python inference/live_predict.py --camera 1 --window 64 --stride 4
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.landmark_extractor import LandmarkExtractor  # noqa: E402
from models.model import HybridLSTMTransformer  # noqa: E402


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    m = cfg["model"]
    model = HybridLSTMTransformer(
        input_dim=ckpt["input_dim"], num_classes=ckpt["num_classes"],
        d_model=m["d_model"], lstm_layers=m["lstm_layers"],
        transformer_layers=m["transformer_layers"], nhead=m["nhead"],
        ff_dim=m["ff_dim"], dropout=m["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    classes = ckpt.get("classes")
    if classes is None:  # fallback: read label_map.json
        import json
        classes = [None] * ckpt["num_classes"]
        for k, v in json.loads(Path("utils/label_map.json").read_text(encoding="utf-8")).items():
            classes[int(k)] = v
    return model, classes, cfg


def draw_subtitle(frame, text: str, conf: float, top3: list[tuple[str, float]]):
    h, w = frame.shape[:2]
    # subtitle background bar
    bar_h = 64
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (30, 30, 30), -1)
    label = f"{text}  ({conf * 100:.0f}%)" if text else "..."
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
    cv2.putText(frame, label, ((w - tw) // 2, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    # top-3 list
    y = 28
    for word, p in top3:
        cv2.putText(frame, f"{word}: {p * 100:.0f}%", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 1, cv2.LINE_AA)
        y += 24
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="models/checkpoints/best.pt")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--window", type=int, default=None, help="override seq_len")
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--smooth_k", type=int, default=None)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, cfg = load_model(args.checkpoint, device)
    inf = cfg["inference"]
    window = args.window or inf["window"]
    stride = args.stride or inf["stride"]
    smooth_k = args.smooth_k or inf["smooth_k"]
    include_face = cfg["data"]["include_face"]
    print(f"Device: {device} | window={window} stride={stride} smooth_k={smooth_k} "
          f"| {len(classes)} classes")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)  # DSHOW: fast startup on Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("Cannot open webcam")

    buffer: deque[np.ndarray] = deque(maxlen=window)
    recent: deque[str] = deque(maxlen=smooth_k)
    extractor = LandmarkExtractor(include_face=include_face, static_image_mode=False)
    frame_idx = 0
    last_pred, last_conf, last_top3 = "", 0.0, []

    with extractor, torch.inference_mode():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            feats = extractor.process_frame(frame)
            buffer.append(feats)
            frame_idx += 1

            if len(buffer) == window and frame_idx % stride == 0:
                x = torch.from_numpy(np.stack(buffer)).unsqueeze(0).to(device)
                mask = torch.ones(1, window, dtype=torch.bool, device=device)
                logits = model(x, mask)
                prob = torch.softmax(logits, 1)[0]
                conf, idx = prob.max(0)
                recent.append(classes[int(idx)])
                word, count = Counter(recent).most_common(1)[0]
                last_pred, last_conf = word, conf.item()
                last_top3 = [(classes[i], float(p)) for p, i in
                             zip(*torch.topk(prob, k=min(3, len(classes))))]

            fps = 0.0  # rendered below
            frame = draw_subtitle(frame, last_pred, last_conf, last_top3)
            cv2.putText(frame, f"fps~{fps:.0f} frames buffered {len(buffer)}/{window}",
                        (10, frame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.imshow("ISL Translator (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
