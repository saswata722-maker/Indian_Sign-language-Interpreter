"""Shared MediaPipe Holistic landmark extractor.

This module is used by BOTH `preprocessing/extract_landmarks.py` (offline,
over the INCLUDE dataset) and `inference/live_predict.py` (live webcam), so
that training and inference see exactly the same feature representation.

Per-frame feature layout (float32):

    [ pose(33*3) | left_hand(21*3) | right_hand(21*3) | face(468*3) | flags(3) ]

    flags = [pose_detected, left_hand_detected, right_hand_detected]

With include_face=True the feature dim is 99 + 126 + 1404 + 3 = 1632.
With include_face=False it is 99 + 126 + 3 = 234.

Normalization: all coordinates are re-expressed relative to the mid-shoulder
point and divided by the shoulder width, making the representation
translation- and scale-invariant. When the pose is missing in a frame, the
previous frame's origin/scale are reused (state is reset at the start of
every clip).
"""

from __future__ import annotations

import contextlib
from typing import Optional

import cv2
import numpy as np

POSE_LM = 33
HAND_LM = 21
FACE_LM = 468

POSE_DIM = POSE_LM * 3          # 99
HAND_DIM = HAND_LM * 3          # 63  (per hand)
HANDS_DIM = 2 * HAND_DIM        # 126
FACE_DIM = FACE_LM * 3          # 1404
FLAGS_DIM = 3                   # pose, left hand, right hand


def feature_dim(include_face: bool) -> int:
    """Total per-frame feature dimensionality."""
    return POSE_DIM + HANDS_DIM + (FACE_DIM if include_face else 0) + FLAGS_DIM


def segment_offsets(include_face: bool) -> dict:
    """Start offsets of each feature segment within a frame vector."""
    off = {"pose": 0, "left_hand": POSE_DIM, "right_hand": POSE_DIM + HAND_DIM}
    face_start = POSE_DIM + HANDS_DIM
    flags_start = face_start + (FACE_DIM if include_face else 0)
    if include_face:
        off["face"] = face_start
    off["flags"] = flags_start
    return off


def _to_array(landmarks, n: int) -> np.ndarray:
    """MediaPipe landmark list -> (n, 3) float32 array, zeros if absent."""
    out = np.zeros((n, 3), dtype=np.float32)
    if landmarks is not None:
        for i, lm in enumerate(landmarks.landmark[:n]):
            out[i, 0] = lm.x
            out[i, 1] = lm.y
            out[i, 2] = lm.z
    return out


class LandmarkExtractor:
    """Stateful MediaPipe Holistic wrapper producing normalized feature frames.

    Use as a context manager::

        with LandmarkExtractor(include_face=True) as ext:
            vec = ext.process_frame(bgr_frame)   # -> (D,) float32
            seq = ext.extract_video("clip.mov")  # -> (T, D) float32
    """

    def __init__(
        self,
        include_face: bool = True,
        model_complexity: int = 1,
        static_image_mode: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        import mediapipe as mp  # deferred so importing this module is cheap

        self.include_face = include_face
        self._mp = mp
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        # Normalization state, reset per clip.
        self._origin = np.zeros(3, dtype=np.float32)
        self._scale = 1.0

    # -- context manager ----------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        self._holistic.close()

    def reset(self):
        """Reset normalization calibration (call at the start of a clip)."""
        self._origin = np.zeros(3, dtype=np.float32)
        self._scale = 1.0

    # -- core ----------------------------------------------------------------
    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Run Holistic on one BGR frame -> (D,) float32 feature vector."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._holistic.process(rgb)

        pose = _to_array(res.pose_landmarks, POSE_LM)
        left = _to_array(res.left_hand_landmarks, HAND_LM)
        right = _to_array(res.right_hand_landmarks, HAND_LM)
        face = _to_array(res.face_landmarks, FACE_LM) if self.include_face else None

        pose_ok = res.pose_landmarks is not None
        left_ok = res.left_hand_landmarks is not None
        right_ok = res.right_hand_landmarks is not None

        # --- normalization calibration -------------------------------------
        if pose_ok:
            ls, rs = pose[11], pose[12]  # left / right shoulder
            width = float(np.linalg.norm(ls - rs))
            if width > 1e-3:
                self._origin = (ls + rs) / 2.0
                self._scale = width

        def norm(arr: np.ndarray) -> np.ndarray:
            return ((arr - self._origin) / self._scale).astype(np.float32)

        parts = [
            norm(pose).reshape(-1),
            norm(left).reshape(-1),
            norm(right).reshape(-1),
        ]
        if face is not None:
            parts.append(norm(face).reshape(-1))
        parts.append(np.array([pose_ok, left_ok, right_ok], dtype=np.float32))
        return np.concatenate(parts, axis=0)

    def extract_video(self, video_path: str, max_frames: Optional[int] = 200) -> np.ndarray:
        """Decode a video file -> (T, D) float32 landmark sequence."""
        self.reset()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        feats = []
        try:
            while True:
                if max_frames is not None and len(feats) >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                feats.append(self.process_frame(frame))
        finally:
            cap.release()
        if not feats:
            return np.zeros((0, feature_dim(self.include_face)), dtype=np.float32)
        return np.stack(feats, axis=0)
