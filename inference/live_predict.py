"""Webcam recognition: SPACE starts/finishes one sign; S toggles streaming."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
import torch
import yaml
from inference.runtime import Predictor, StreamState


def run_live(model_path='models/best_model.pth', config_path='config.yaml',
             class_names=None, seq_len=None, threshold=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    predictor = Predictor(model_path, config, 'cuda' if torch.cuda.is_available() else 'cpu')
    if class_names is not None and list(class_names) != predictor.names:
        raise ValueError('Class order must match the checkpoint')
    inf = dict(config['inference'])
    if threshold is not None:
        inf['confidence_threshold'] = threshold
    if seq_len is not None:
        inf['window_seconds'] = seq_len/25.0
    state = StreamState(predictor, inf)
    extractor = predictor.make_extractor()
    from mediapipe import solutions
    cap = cv2.VideoCapture(0)
    mode, recording = 'segment', False
    result = state.result('ready', 'Press SPACE to record one sign')
    try:
        if not cap.isOpened():
            raise RuntimeError('Could not open webcam')
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = time.monotonic()
            key = cv2.waitKey(1) & 0xff
            if key == ord('q'):
                break
            finalize = False
            if key in (ord('r'), ord('s')):
                state.reset()
                extractor.reset_anchor()
                extractor.mp_holistic.reset()
                recording = False
                if key == ord('s'):
                    mode = 'stream' if mode == 'segment' else 'segment'
                result = state.result('ready', 'Ready')
            if key == ord(' ') and mode == 'segment':
                if recording:
                    finalize = True
                else:
                    state.reset()
                    recording = True
            if mode == 'stream' or recording:
                feature = extractor.process_frame(frame)
                tracked = extractor.last_result
                for landmarks, connections in [
                    (tracked.pose_landmarks, solutions.holistic.POSE_CONNECTIONS),
                    (tracked.left_hand_landmarks, solutions.holistic.HAND_CONNECTIONS),
                    (tracked.right_hand_landmarks, solutions.holistic.HAND_CONNECTIONS),
                ]:
                    if landmarks:
                        solutions.drawing_utils.draw_landmarks(frame, landmarks, connections)
                if mode == 'segment' and state.buffer:
                    finalize |= timestamp-state.buffer[0][0] >= inf.get('max_segment_seconds', 8)
                result = state.push(feature, timestamp, extractor.anchor_valid, mode, finalize)
                if finalize:
                    recording = False
            label = result['label']
            if result['confidence']:
                label += f" ({result['confidence']:.1%})"
            cv2.putText(frame, label, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 128), 2)
            for index, (name, probability) in enumerate(result['top3']):
                cv2.putText(frame, f'{name}: {probability:.1%}', (15, 65+index*20),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            cv2.putText(frame, f"{mode}: {result['status']} | SPACE record/finish | S mode | R reset | Q quit",
                        (15, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            cv2.imshow('ISL-Sense', frame)
    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    run_live()
