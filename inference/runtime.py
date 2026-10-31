"""Shared model adapter and timestamp-bounded live prediction state."""
from collections import deque
from copy import deepcopy
import numpy as np
import torch
from preprocessing.features import prepare_sequence, window_quality
from training.model import HybridSignModel


class Predictor:
    def __init__(self, path, fallback_config, device='cpu'):
        self.device = torch.device(device)
        self.checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.config = deepcopy(self.checkpoint.get('config', fallback_config))
        mapping = self.checkpoint['class_to_idx']
        if sorted(mapping.values()) != list(range(len(mapping))):
            raise ValueError('Checkpoint class indices must be contiguous')
        self.names = [name for name, idx in sorted(mapping.items(), key=lambda item: item[1])]
        mc = self.config['model']
        self.model = HybridSignModel(num_classes=len(mapping), **{
            k: mc[k] for k in ('input_dim', 'd_model', 'nhead', 'num_transformer_layers', 'dropout')
        }).to(self.device)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()

    def probabilities(self, raw, timestamps=None):
        seq = prepare_sequence(raw, self.config, timestamps)
        with torch.inference_mode():
            x = torch.from_numpy(seq).unsqueeze(0).to(self.device)
            return torch.softmax(self.model(x), dim=1)[0].cpu().numpy()

    def make_extractor(self):
        from preprocessing.extract_landmarks import LandmarkExtractor
        return LandmarkExtractor(**self.config['extraction'], feature_version=
                                 self.config['preprocessing'].get('feature_version', 1))


class StreamState:
    """One stream only. Call under its owner's lock; timestamps are capture seconds."""
    def __init__(self, predictor, inference_config):
        self.predictor = predictor
        self.config = inference_config
        self.window_seconds = float(inference_config.get('window_seconds', 2.56))
        self.max_gap = float(inference_config.get('max_frame_gap_seconds', 1.0))
        self.min_frames = int(inference_config.get('min_frames', 8))
        self.buffer = deque(maxlen=int(inference_config.get('max_buffer_frames', 256)))
        self.recent = deque(maxlen=int(inference_config.get('smoothing_window', 5)))
        self.last_timestamp = None

    def reset(self):
        self.buffer.clear()
        self.recent.clear()
        self.last_timestamp = None

    def result(self, status, label='Waiting for gesture...', **values):
        return dict(label=label, confidence=0.0, status=status, top3=[],
                    pose_coverage=0.0, hand_coverage=0.0, motion=0.0,
                    calibrated=False, buffer_len=len(self.buffer), **values)

    def push(self, feature, timestamp, calibrated, mode='stream', finalize=False):
        if not np.isfinite(timestamp):
            raise ValueError('Timestamp must be finite')
        if self.last_timestamp is not None:
            if timestamp <= self.last_timestamp:
                raise ValueError('Out-of-order frame')
            if timestamp-self.last_timestamp > self.max_gap:
                self.reset()
        self.last_timestamp = timestamp
        self.buffer.append((timestamp, feature))
        duration = self.window_seconds if mode == 'stream' else float(self.config.get('max_segment_seconds', 8))
        while self.buffer and timestamp-self.buffer[0][0] > duration:
            self.buffer.popleft()
        result = self.result('buffering' if mode == 'stream' else 'recording')
        result['calibrated'] = bool(calibrated)
        span = timestamp-self.buffer[0][0]
        if mode == 'segment' and not finalize:
            return result
        if len(self.buffer) < self.min_frames or (mode == 'stream' and span < duration*.8):
            if finalize:
                self.reset()
                result.update(status='too_short', label='Record a longer sign')
            return result
        times, features = zip(*self.buffer)
        raw = np.asarray(features, dtype=np.float32)
        pose, hands, motion = window_quality(raw, times)
        result.update(pose_coverage=pose, hand_coverage=hands, motion=motion)
        gate = self.config.get('gate', {})
        # Complete manually delimited signs may be static. Do not motion-reject them.
        accepted = calibrated and (not gate.get('enabled', True) or (
            pose >= gate.get('min_pose_coverage', .6) and
            hands >= gate.get('min_hand_coverage', .4) and
            (mode == 'segment' or motion >= gate.get('min_motion', .003))))
        if not accepted:
            self.recent.clear()
            result.update(status='idle', label='Show hands & sign')
        else:
            probs = self.predictor.probabilities(raw, times)
            threshold = self.config.get('confidence_threshold', .35)
            if float(probs.max()) < threshold:
                self.recent.clear()
                result.update(status='below_threshold', label='Uncertain')
            else:
                if mode == 'segment':
                    self.recent.clear()
                # Forget stale votes, even if low-confidence frames were skipped.
                ttl = self.config.get('prediction_ttl_seconds', 1.0)
                while self.recent and timestamp-self.recent[0][0] > ttl:
                    self.recent.popleft()
                self.recent.append((timestamp, probs))
                averaged = np.mean([p for _, p in self.recent], axis=0)
                idx = int(averaged.argmax())
                if averaged[idx] >= threshold:
                    result.update(status='predicting', label=self.predictor.names[idx],
                                  confidence=float(averaged[idx]))
                else:
                    result.update(status='below_threshold', label='Uncertain')
                probs = averaged
            result['top3'] = [(self.predictor.names[i], float(probs[i]))
                              for i in np.argsort(probs)[::-1][:3]]
        if finalize:
            self.reset()
        return result
