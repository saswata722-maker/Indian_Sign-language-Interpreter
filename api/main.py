"""Per-client, ordered webcam inference using the shared live runtime."""
import base64
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import cv2
import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from inference.runtime import Predictor, StreamState

ROOT = Path(__file__).resolve().parent.parent
with open(os.environ.get('CONFIG_PATH', ROOT/'config.yaml')) as f:
    config = yaml.safe_load(f)
device = 'cuda' if os.environ.get('USE_CUDA', 'false').lower() == 'true' and torch.cuda.is_available() else 'cpu'
predictor = Predictor(os.environ.get('MODEL_PATH', ROOT/'models/best_model.pth'), config, device)
inference_config = dict(config['inference'])
for key, env, cast in [('confidence_threshold', 'CONFIDENCE_THRESHOLD', float),
                       ('smoothing_window', 'SMOOTHING_WINDOW', int)]:
    if env in os.environ:
        inference_config[key] = cast(os.environ[env])
gate = dict(inference_config.get('gate', {}))
for key, env in [('min_pose_coverage', 'GATE_MIN_POSE_COVERAGE'),
                 ('min_hand_coverage', 'GATE_MIN_HAND_COVERAGE'), ('min_motion', 'GATE_MIN_MOTION')]:
    if env in os.environ:
        gate[key] = float(os.environ[env])
if 'GATE_ENABLED' in os.environ:
    gate['enabled'] = os.environ['GATE_ENABLED'].lower() == 'true'
inference_config['gate'] = gate


class ClientState:
    def __init__(self):
        self.lock = threading.Lock()
        self.extractor = predictor.make_extractor()
        self.stream = StreamState(predictor, inference_config)
        self.last_seen = time.monotonic()
        self.frame_id = -1
        self.mode = None


clients = {}
clients_lock = threading.Lock()
# Shared neural model is serialized; MediaPipe state is always client-local.
model_lock = threading.Lock()


def get_client(client_id):
    with clients_lock:
        now = time.monotonic()
        for key, state in list(clients.items()):
            if now-state.last_seen > 300 and state.lock.acquire(blocking=False):
                try:
                    state.extractor.close()
                    del clients[key]
                finally:
                    state.lock.release()
        if client_id not in clients:
            if len(clients) >= 16:
                raise HTTPException(503, 'Too many active camera sessions')
            clients[client_id] = ClientState()
        state = clients[client_id]
        state.last_seen = now
        return state


class PredictRequest(BaseModel):
    client_id: str = Field(min_length=1, max_length=128)
    image_b64: str
    frame_id: int = Field(ge=0)
    timestamp: float = Field(ge=0, allow_inf_nan=False)
    mode: Literal['stream', 'segment'] = 'stream'
    finalize: bool = False


class ResetRequest(BaseModel):
    client_id: str


@asynccontextmanager
async def lifespan(app):
    yield
    with clients_lock:
        for state in clients.values():
            with state.lock:
                state.extractor.close()
        clients.clear()


app = FastAPI(title='ISL-Sense', version='2.0', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=ROOT/'static'), name='static')


@app.get('/')
def frontend():
    return FileResponse(ROOT/'static/index.html')


@app.get('/health')
def health():
    return dict(status='ok', classes=len(predictor.names), device=device,
                feature_version=predictor.config['preprocessing'].get('feature_version', 1),
                window_seconds=inference_config.get('window_seconds', 2.56))


@app.get('/classes')
def classes():
    return {'classes': predictor.names}


@app.post('/predict')
def predict(req: PredictRequest):
    state = get_client(req.client_id)
    with state.lock:
        if req.frame_id <= state.frame_id:
            raise HTTPException(409, 'Out-of-order frame; reset the session before restarting IDs')
        if state.stream.last_timestamp is not None and req.timestamp <= state.stream.last_timestamp:
            raise HTTPException(409, 'Out-of-order capture timestamp')
        try:
            data = base64.b64decode(req.image_b64.split(',')[-1], validate=True)
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError('Invalid image')
        except Exception as exc:
            raise HTTPException(400, 'Invalid image') from exc
        gap = state.stream.last_timestamp is not None and req.timestamp-state.stream.last_timestamp > state.stream.max_gap
        if state.mode != req.mode or gap:
            state.stream.reset()
            state.extractor.reset_anchor()
            state.extractor.mp_holistic.reset()
        state.mode = req.mode
        feature = state.extractor.process_frame(frame)
        with model_lock:
            result = state.stream.push(feature, req.timestamp, state.extractor.anchor_valid, req.mode, req.finalize)
        state.frame_id = req.frame_id
        state.last_seen = time.monotonic()
        return dict(client_id=req.client_id, frame_id=req.frame_id, **result)


@app.post('/reset')
def reset(req: ResetRequest):
    state = get_client(req.client_id)
    with state.lock:
        state.stream.reset()
        state.extractor.reset_anchor()
        state.extractor.mp_holistic.reset()
        state.mode = None
        # Keep frame IDs monotonic so delayed pre-reset requests cannot be accepted.
    return {'status': 'reset', 'client_id': req.client_id}
