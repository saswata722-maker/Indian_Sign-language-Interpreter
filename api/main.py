"""
ISL-Sense Web API — Render deployment backend.

Reuses the existing codebase unchanged:
  - preprocessing.extract_landmarks.LandmarkExtractor  (normalization + anchor)
  - training.model.HybridSignModel                      (sequence classifier)
  - training.dataset.resample_sequence                  (temporal resampling)
  - config.yaml                                         (gating thresholds, model hyperparams)

Pipeline per frame (matches live_predict.py exactly):
  1. Extract normalized landmarks via MediaPipe
  2. Buffer frames in a per-client sliding window
  3. Gate on anchor_valid + pose/hand coverage + motion
  4. Resample -> velocity (diff) -> concat  (identical to training)
  5. Model inference -> softmax -> top-3
  6. Confidence threshold + temporal majority-vote debounce

Endpoints:
  GET  /classes   -> list of class names
  GET  /health    -> service health
  POST /predict   -> {client_id, image_b64} -> {label, confidence, top3, status}
  POST /reset     -> {client_id}            -> reset anchor + buffer
"""

import base64
import io
import os
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to path so existing modules import cleanly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing.extract_landmarks import LandmarkExtractor
from training.dataset import resample_sequence
from training.model import HybridSignModel

# ---------------------------------------------------------------------------
# Config & model loading (once at startup)
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", str(ROOT / "config.yaml")))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(ROOT / "models" / "best_model.pth")))

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

SEQ_LEN = int(os.environ.get("SEQ_LEN", config["inference"]["seq_len"]))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", config["inference"]["confidence_threshold"]))
SMOOTHING_WINDOW = int(os.environ.get("SMOOTHING_WINDOW", config["inference"]["smoothing_window"]))

gate_cfg = config.get("inference", {}).get("gate", {})
GATE_ENABLED = os.environ.get("GATE_ENABLED", str(gate_cfg.get("enabled", True))).lower() == "true"
GATE_POSE = float(os.environ.get("GATE_MIN_POSE_COVERAGE", gate_cfg.get("min_pose_coverage", 0.6)))
GATE_HAND = float(os.environ.get("GATE_MIN_HAND_COVERAGE", gate_cfg.get("min_hand_coverage", 0.4)))
GATE_MOTION = float(os.environ.get("GATE_MIN_MOTION", gate_cfg.get("min_motion", 0.003)))

USE_VELOCITY = config["preprocessing"].get("use_velocity", False)

use_cuda = os.environ.get("USE_CUDA", "false").lower() == "true" and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=device)
model_config = checkpoint.get("config", config)

if "class_to_idx" in checkpoint:
    idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}
    CLASS_NAMES = [idx_to_class[i] for i in range(len(idx_to_class))]
else:
    CLASS_NAMES = ["loud", "quiet", "happy", "sad", "Beautiful", "Ugly", "Deaf", "Blind"]

model = HybridSignModel(
    input_dim=model_config["model"]["input_dim"],
    num_classes=len(CLASS_NAMES),
    d_model=model_config["model"]["d_model"],
    nhead=model_config["model"]["nhead"],
    num_transformer_layers=model_config["model"]["num_transformer_layers"],
    dropout=model_config["model"]["dropout"],
).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

extractor = LandmarkExtractor(
    include_face=config["extraction"]["include_face"],
    model_complexity=config["extraction"]["model_complexity"],
    min_detection_confidence=config["extraction"]["min_detection_confidence"],
    min_tracking_confidence=config["extraction"]["min_tracking_confidence"],
)

print(f"Model loaded: {len(CLASS_NAMES)} classes, input_dim={model_config['model']['input_dim']}, device={device}")


# ---------------------------------------------------------------------------
# Per-client state (HTTP is stateless; track buffers by client_id)
# ---------------------------------------------------------------------------
class ClientState:
    def __init__(self):
        self.buffer = []  # list of feature vectors
        self.recent_preds = deque(maxlen=SMOOTHING_WINDOW)
        self.current_label = "Waiting for gesture..."
        self.confidence = 0.0
        self.last_probs = []


clients: dict[str, ClientState] = {}


def get_client(client_id: str) -> ClientState:
    if client_id not in clients:
        clients[client_id] = ClientState()
    return clients[client_id]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    client_id: str
    image_b64: str  # base64-encoded JPEG/PNG frame


class PredictResponse(BaseModel):
    client_id: str
    label: str
    confidence: float
    status: str  # "calibrating" | "idle" | "predicting" | "buffering"
    top3: list[list[str, float]]
    pose_coverage: float
    hand_coverage: float
    motion: float
    calibrated: bool
    buffer_len: int


class ResetRequest(BaseModel):
    client_id: str


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="ISL-Sense Web API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
STATIC_DIR = ROOT / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "classes": len(CLASS_NAMES), "device": str(device)}


@app.get("/classes")
def classes():
    return {"classes": CLASS_NAMES}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    client = get_client(req.client_id)

    # Decode base64 image
    try:
        img_bytes = base64.b64decode(req.image_b64.split(",")[-1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Extract features (same as live_predict.py line 145)
    feat = extractor.process_frame(frame)
    client.buffer.append(feat)

    result = {
        "client_id": req.client_id,
        "label": client.current_label,
        "confidence": client.confidence,
        "status": "buffering",
        "top3": client.last_probs,
        "pose_coverage": 0.0,
        "hand_coverage": 0.0,
        "motion": 0.0,
        "calibrated": bool(extractor.anchor_valid),
        "buffer_len": len(client.buffer),
    }

    if len(client.buffer) < SEQ_LEN:
        return result

    raw = np.array(client.buffer, dtype=np.float32)

    # Gate on the RAW window (same as live_predict.py lines 182-190)
    pose_cov = float((raw[:, -3] > 0.5).mean())
    hand_cov = float(((raw[:, -2] > 0.5) | (raw[:, -1] > 0.5)).mean())
    motion = float(np.abs(np.diff(raw[:, :228], axis=0)).mean())

    calibrated = bool(extractor.anchor_valid)
    gate_ok = calibrated and (
        not GATE_ENABLED
        or (pose_cov >= GATE_POSE and hand_cov >= GATE_HAND and motion >= GATE_MOTION)
    )

    result["pose_coverage"] = pose_cov
    result["hand_coverage"] = hand_cov
    result["motion"] = motion
    result["calibrated"] = calibrated
    result["buffer_len"] = len(client.buffer)

    if not gate_ok:
        if not calibrated:
            client.current_label = "Calibrating..."
            client.confidence = 0.0
            result["status"] = "calibrating"
        else:
            client.current_label = (
                "Show hands & sign" if hand_cov < GATE_HAND else "Waiting for motion..."
            )
            client.confidence = 0.0
            result["status"] = "idle"
        client.last_probs = []
        client.recent_preds.clear()
        result["label"] = client.current_label
        result["confidence"] = 0.0
        result["top3"] = []
        return result

    # --- Inference pipeline: resample -> velocity -> concat (matches training) ---
    seq = resample_sequence(raw, target_len=SEQ_LEN)

    if USE_VELOCITY:
        delta = np.diff(seq, axis=0, prepend=seq[:1])
        seq = np.concatenate([seq, delta], axis=1)

    tensor_x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor_x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [(CLASS_NAMES[i], float(probs[i])) for i in top3_idx]
        best_idx = top3_idx[0]
        best_prob = float(probs[best_idx])

    client.last_probs = top3

    if best_prob >= CONFIDENCE_THRESHOLD:
        client.recent_preds.append(CLASS_NAMES[best_idx])
        majority = max(set(client.recent_preds), key=client.recent_preds.count)
        client.current_label = majority
        client.confidence = best_prob
        result["status"] = "predicting"
    else:
        result["status"] = "below_threshold"

    result["label"] = client.current_label
    result["confidence"] = client.confidence
    result["top3"] = top3

    # Keep window sliding: drop oldest frame to maintain rolling buffer
    client.buffer.pop(0)

    return result


@app.post("/reset")
def reset(req: ResetRequest):
    """Force recalibration of the normalization anchor (same as 'r' key in live_predict.py)."""
    extractor.reset_anchor()
    if req.client_id in clients:
        clients[req.client_id].buffer = []
        clients[req.client_id].recent_preds.clear()
        clients[req.client_id].current_label = "Calibrating..."
        clients[req.client_id].confidence = 0.0
        clients[req.client_id].last_probs = []
    return {"status": "anchor reset", "client_id": req.client_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
