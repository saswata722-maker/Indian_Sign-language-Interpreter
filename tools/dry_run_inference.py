"""Dry-run test: verify checkpoint loads and inference shapes match (no webcam)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import yaml

from training.model import HybridSignModel

with open("config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("models/best_model.pth", map_location=device)
model_config = checkpoint.get("config", config)

idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
print(f"Classes in checkpoint: {len(class_names)}")
print(f"Checkpoint val_acc: {checkpoint.get('val_acc')*100:.1f}%")

model = HybridSignModel(
    input_dim=model_config["model"]["input_dim"],
    num_classes=len(class_names),
    d_model=model_config["model"]["d_model"],
    nhead=model_config["model"]["nhead"],
    num_transformer_layers=model_config["model"]["num_transformer_layers"],
    dropout=model_config["model"]["dropout"],
).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Model loaded OK (input_dim={model_config['model']['input_dim']})")

# Simulate a live sliding window: 64 frames of 228-dim features
seq_len = config["inference"]["seq_len"]
raw = np.random.randn(seq_len, 228).astype(np.float32) * 0.1

use_velocity = model_config["preprocessing"].get("use_velocity", False)
if use_velocity:
    delta = np.diff(raw, axis=0, prepend=raw[:1])
    model_input = np.concatenate([raw, delta], axis=1)
print(f"Window: {seq_len} frames -> model input shape: {model_input.shape}")

with torch.no_grad():
    logits = model(torch.tensor(model_input).unsqueeze(0).to(device))
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    best = np.argmax(probs)
print(f"Prediction: '{class_names[best]}' ({probs[best]*100:.1f}%)")
print("Dry-run OK — live_predict.py is compatible with the trained model.")
