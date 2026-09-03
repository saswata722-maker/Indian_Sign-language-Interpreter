"""
Real-Time Webcam Inference Module
Connects to webcam, extracts landmarks in real-time, and recognizes ISL gestures.

Pipeline:
  1. Capture frame from webcam
  2. Extract normalized landmark features via MediaPipe
  3. Buffer frames in a sliding window (deque)
  4. Run model inference when buffer is full
  5. Temporal debounce: majority vote over recent predictions
  6. Render recognized sign as subtitle overlay

Based on AI4Bharat OpenHands live inference approach.
"""

import cv2
import numpy as np
import torch
import collections
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.extract_landmarks import LandmarkExtractor
from training.model import HybridSignModel


def run_live(model_path="models/best_model.pth", config_path="config.yaml",
             class_names=None, seq_len=40, threshold=0.75):
    """
    Run live ISL recognition from webcam.

    Args:
        model_path: Path to trained model checkpoint.
        config_path: Path to configuration YAML file.
        class_names: List of class names (overrides checkpoint class names).
        seq_len: Sliding window length (number of frames).
        confidence_threshold: Minimum confidence to accept a prediction.
    """
    # Load config
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint.get('config', config)

    # Determine class names
    if class_names is None and 'class_to_idx' in checkpoint:
        idx_to_class = {v: k for k, v in checkpoint['class_to_idx'].items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    if class_names is None:
        class_names = ["loud", "quiet", "happy", "sad",
                       "Beautiful", "Ugly", "Deaf", "Blind"]
        print("Warning: No class names found. Using defaults.")

    # Initialize model
    model = HybridSignModel(
        input_dim=model_config['model']['input_dim'],
        num_classes=len(class_names),
        d_model=model_config['model']['d_model'],
        nhead=model_config['model']['nhead'],
        num_transformer_layers=model_config['model']['num_transformer_layers'],
        dropout=model_config['model']['dropout']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Initialize extractor
    extractor = LandmarkExtractor(
        include_face=config['extraction']['include_face'],
        model_complexity=config['extraction']['model_complexity'],
        min_detection_confidence=config['extraction']['min_detection_confidence'],
        min_tracking_confidence=config['extraction']['min_tracking_confidence']
    )

    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Buffers
    frame_buffer = collections.deque(maxlen=seq_len)
    recent_preds = collections.deque(maxlen=config['inference']['smoothing_window'])
    current_label = "Waiting for gesture..."
    confidence = 0.0

    print("=" * 50)
    print("ISL-Sense Live Recognition")
    print("=" * 50)
    print(f"Classes: {class_names}")
    print(f"Window size: {seq_len} frames")
    print(f"Confidence threshold: {threshold}")
    print("Press 'q' to quit.")
    print("=" * 50)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror for natural interaction
        h, w, _ = frame.shape

        # Extract features for current frame
        feat = extractor.process_frame(frame)
        frame_buffer.append(feat)

        # Run inference once buffer is full
        if len(frame_buffer) == seq_len:
            seq = np.array(frame_buffer)
            tensor_x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor_x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                best_idx = np.argmax(probs)
                best_prob = probs[best_idx]

                if best_prob >= threshold:
                    recent_preds.append(class_names[best_idx])
                    # Temporal debounce: majority vote
                    majority = max(set(recent_preds), key=recent_preds.count)
                    current_label = majority
                    confidence = best_prob

        # Render subtitle overlay
        cv2.rectangle(frame, (0, h - 70), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, f"Sign: {current_label} ({confidence*100:.1f}%)",
                    (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 128), 2, cv2.LINE_AA)

        # Show buffer status
        cv2.putText(frame, f"Buffer: {len(frame_buffer)}/{seq_len}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Indian Sign Language Recognition (Live)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print("Session ended.")


if __name__ == "__main__":
    run_live()
