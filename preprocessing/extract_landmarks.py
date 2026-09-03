"""
Landmark Extraction Module
Processes videos in raw_data/ and saves .npy sequences into landmarks/.

Uses MediaPipe Holistic to extract:
  - Pose landmarks (33 x 3 = 99 dims)
  - Left hand landmarks (21 x 3 = 63 dims)
  - Right hand landmarks (21 x 3 = 63 dims)
  - Face mesh (468 x 3 = 1404 dims, optional)
  - Detection flags (3 dims)

Applies torso-centered & scale-invariant normalization:
  - Anchor: mid-shoulder point (left_shoulder + right_shoulder) / 2
  - Scale: shoulder Euclidean distance ||left_shoulder - right_shoulder||

Output: .npy files with shape (T_frames, feature_dim) float32
  - include_feature=False → 234 dims (recommended for training/live)
  - include_face=True → 1632 dims (research, subtle facial expressions)
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm


class LandmarkExtractor:
    def __init__(self, include_face=False, model_complexity=1,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """Initialize MediaPipe Holistic landmark extractor."""
        self.include_face = include_face
        self.mp_holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.origin = np.zeros(3, dtype=np.float32)
        self.scale = 1.0

    def process_frame(self, frame_bgr):
        """Process a single BGR frame and return normalized landmark features."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mp_holistic.process(rgb)

        def to_arr(lms, count):
            if lms is None:
                return np.zeros((count, 3), dtype=np.float32)
            return np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark[:count]],
                            dtype=np.float32)

        pose = to_arr(res.pose_landmarks, 33)
        lh = to_arr(res.left_hand_landmarks, 21)
        rh = to_arr(res.right_hand_landmarks, 21)

        pose_ok = res.pose_landmarks is not None
        lh_ok = res.left_hand_landmarks is not None
        rh_ok = res.right_hand_landmarks is not None

        # Torso calibration (per-frame)
        if pose_ok:
            ls, rs = pose[11], pose[12]
            width = float(np.linalg.norm(ls - rs))
            if width > 1e-3:
                self.origin = (ls + rs) / 2.0
                self.scale = width

        def norm(arr):
            return ((arr - self.origin) / self.scale).astype(np.float32)

        parts = [norm(pose).reshape(-1), norm(lh).reshape(-1), norm(rh).reshape(-1)]
        if self.include_face:
            face = to_arr(res.face_landmarks, 468)
            parts.append(norm(face).reshape(-1))
        parts.append(np.array([pose_ok, lh_ok, rh_ok], dtype=np.float32))

        return np.concatenate(parts, axis=0)

    def extract_video(self, video_path):
        """Extract landmark sequence from a video file."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(self.process_frame(frame))
        cap.release()
        return np.stack(frames, axis=0) if frames else None

    def close(self):
        """Release MediaPipe resources."""
        self.mp_holistic.close()


def batch_extract(raw_root="raw_data", save_root="landmarks", include_face=False):
    """Batch-process all videos and save .npy files."""
    extractor = LandmarkExtractor(include_face=include_face)
    raw_path = Path(raw_root)
    save_path = Path(save_root)

    video_files = (
        list(raw_path.rglob("*.MOV"))
        + list(raw_path.rglob("*.mp4"))
        + list(raw_path.rglob("*.mov"))
        + list(raw_path.rglob("*.MP4"))
    )

    print(f"Found {len(video_files)} videos in {raw_root}. Extracting...")
    for v_path in tqdm(video_files, desc="Extracting"):
        rel = v_path.relative_to(raw_path).with_suffix(".npy")
        out_file = save_path / rel
        if out_file.exists():
            continue
        out_file.parent.mkdir(parents=True, exist_ok=True)
        feats = extractor.extract_video(v_path)
        if feats is not None:
            np.save(out_file, feats)
    extractor.close()
    print(f"Done. Landmarks saved to {save_root}/")


if __name__ == "__main__":
    batch_extract(raw_root="raw_data", save_root="landmarks", include_face=False)