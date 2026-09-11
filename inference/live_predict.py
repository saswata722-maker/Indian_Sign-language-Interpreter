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

import mediapipe as mp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.extract_landmarks import LandmarkExtractor
from training.model import HybridSignModel

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def run_live(model_path="models/best_model.pth", config_path="config.yaml",
             class_names=None, seq_len=None, threshold=None):
    """
    Run live ISL recognition from webcam.

    Args:
        model_path: Path to trained model checkpoint.
        config_path: Path to configuration YAML file.
        class_names: List of class names (overrides checkpoint class names).
        seq_len: Sliding window length (defaults to config inference.seq_len).
        threshold: Minimum confidence to accept a prediction.
    """
    # Load config
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if seq_len is None:
        seq_len = config['inference']['seq_len']
    if threshold is None:
        threshold = config['inference']['confidence_threshold']

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
    last_probs = []  # top-3 (name, prob) for on-screen display

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

        h, w, _ = frame.shape

        # Extract features from the UNMIRRORED frame — mirror-flipping swaps
        # MediaPipe's left/right hand labels vs. the (unmirrored) training
        # videos, which silently corrupts the feature layout.
        feat = extractor.process_frame(frame)
        frame_buffer.append(feat)

        # --- Skeleton overlay: show exactly what the camera is tracking ---
        res = getattr(extractor, "last_result", None)
        if res is not None:
            spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            hand_spec = mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=3)
            if res.pose_landmarks:
                mp_drawing.draw_landmarks(frame, res.pose_landmarks,
                                          mp_holistic.POSE_CONNECTIONS, spec, spec)
            if res.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, res.left_hand_landmarks,
                                          mp_holistic.HAND_CONNECTIONS, hand_spec, hand_spec)
            if res.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, res.right_hand_landmarks,
                                          mp_holistic.HAND_CONNECTIONS, hand_spec, hand_spec)

        # Tracking flags (last 3 features: pose, left hand, right hand)
        flags = feat[-3:]
        status = (f"Pose: {'OK' if flags[0] > 0.5 else '--'}   "
                  f"L-Hand: {'OK' if flags[1] > 0.5 else '--'}   "
                  f"R-Hand: {'OK' if flags[2] > 0.5 else '--'}")
        status_color = (0, 255, 0) if (flags[1] > 0.5 or flags[2] > 0.5) else (0, 0, 255)
        cv2.putText(frame, status, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    status_color, 2, cv2.LINE_AA)

        # Run inference once buffer is full
        if len(frame_buffer) == seq_len:
            seq = np.array(frame_buffer, dtype=np.float32)

            # Append velocity (frame-to-frame deltas) to match training input
            use_velocity = model_config['preprocessing'].get('use_velocity', False)
            if use_velocity:
                delta = np.diff(seq, axis=0, prepend=seq[:1])
                seq = np.concatenate([seq, delta], axis=1)

            tensor_x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor_x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                top3_idx = np.argsort(probs)[::-1][:3]
                top3 = [(class_names[i], probs[i]) for i in top3_idx]
                best_idx = top3_idx[0]
                best_prob = probs[best_idx]

                last_probs = top3  # for on-screen display

                if best_prob >= threshold:
                    recent_preds.append(class_names[best_idx])
                    # Temporal debounce: majority vote
                    majority = max(set(recent_preds), key=recent_preds.count)
                    current_label = majority
                    confidence = best_prob

        # Render subtitle overlay
        cv2.rectangle(frame, (0, h - 110), (w, h), (20, 20, 20), -1)
        if confidence > 0:
            color = (0, 255, 128) if confidence >= threshold else (0, 200, 255)
            cv2.putText(frame, f"Sign: {current_label} ({confidence*100:.1f}%)",
                        (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        color, 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"Sign: {current_label}",
                        (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (160, 160, 160), 2, cv2.LINE_AA)

        # Show live top-3 model guesses (always visible, even below threshold)
        for k, (name, prob) in enumerate(last_probs[:3]):
            bar = "#" * int(prob * 30)
            cv2.putText(frame, f"{name[:28]:28s} {prob*100:4.1f}% {bar}",
                        (20, h - 80 + k * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 200, 200), 1, cv2.LINE_AA)

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
