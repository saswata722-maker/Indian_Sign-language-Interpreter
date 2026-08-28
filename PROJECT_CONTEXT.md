# Project Context: Real-Time Indian Sign Language (ISL) Detection

Use this document as full context to scaffold the project described below.

## 1. Goal

Build a system that:
- Captures **live webcam video**
- Detects **Indian Sign Language (ISL)** gestures in real time
- Displays the recognized gesture/word as a **subtitle overlay** on the video feed

## 2. High-Level Pipeline

1. **Landmark extraction**: Use MediaPipe Holistic to extract hand, face, and pose keypoints per frame (do NOT feed raw pixels into the classifier).
2. **Sequence modeling**: Feed the per-frame landmark vectors, as a time sequence, into a hybrid LSTM + Transformer model.
3. **Classification**: Output a predicted gesture/word class per sequence window.
4. **Live inference**: Run the trained model on a sliding window of webcam frames and render the predicted word as subtitle text using OpenCV.

## 3. Model Architecture (required design)

Hybrid LSTM → Transformer encoder:
1. Input: sequence of per-frame landmark vectors (concatenated x,y,z for hands/pose/face keypoints from MediaPipe)
2. Linear projection layer (per-frame embedding)
3. 1–2 layer LSTM — captures short-term local motion between adjacent frames
4. 2–4 layer Transformer Encoder, 4–8 attention heads — captures long-range dependencies across the full gesture sequence
5. Classification head: dense layer + softmax over gesture/word vocabulary

Framework: **PyTorch** (chosen over TensorFlow for custom architecture flexibility).

## 4. Dataset

- Primary: **INCLUDE dataset** (IIIT Bombay) — isolated word-level ISL videos
- Supplementary: self-recorded videos for target vocabulary (ISL datasets are sparse compared to ASL)
- Data type: RGB video clips → converted to landmark keypoint sequences (.npy) before training

## 5. Folder Structure to Generate

```
isl-detection/
├── data/
│   ├── raw_videos/
│   └── landmarks/
├── preprocessing/
│   ├── extract_landmarks.py
│   └── augment_data.py
├── models/
│   ├── model.py
│   └── checkpoints/
├── training/
│   ├── train.py
│   ├── config.yaml
│   └── dataset.py
├── inference/
│   └── live_predict.py
├── utils/
│   ├── metrics.py
│   └── label_map.json
├── requirements.txt
└── README.md
```

## 6. File-by-File Responsibilities

- **preprocessing/extract_landmarks.py**: Loop over `data/raw_videos/`, run MediaPipe Holistic per frame, save landmark sequences as `.npy` into `data/landmarks/`.
- **preprocessing/augment_data.py**: Optional augmentation (horizontal flip, time-warp/speed variation, small coordinate noise) on landmark sequences.
- **training/dataset.py**: PyTorch `Dataset`/`DataLoader` that loads landmark `.npy` files + labels from `utils/label_map.json`, pads/truncates sequences to fixed length.
- **models/model.py**: Defines the LSTM+Transformer hybrid class described in section 3.
- **training/train.py**: Training loop — loss (cross-entropy), optimizer (Adam), scheduler, checkpoint saving to `models/checkpoints/`.
- **training/config.yaml**: Hyperparameters — epochs, batch size, learning rate, sequence length, hidden dims, number of transformer heads/layers.
- **inference/live_predict.py**: Opens webcam via OpenCV, extracts landmarks per frame via MediaPipe, buffers a sliding window of frames, runs the trained model, overlays predicted word as subtitle text on the video frame.
- **utils/metrics.py**: Accuracy, confusion matrix helpers using scikit-learn.
- **utils/label_map.json**: Maps class index ↔ gesture/word label.

## 7. Python Dependencies

```
opencv-python
mediapipe
torch
torchvision
numpy
pandas
scikit-learn
onnx
onnxruntime
pyyaml
matplotlib
```

## 8. Key Constraints / Decisions Already Made

- Input to the model is **landmark keypoints**, not raw video frames (lighter, faster, needs less data).
- Model must be a **hybrid LSTM + Transformer**, not either alone.
- Target language: **Python**, framework: **PyTorch**.
- Final deliverable includes a **real-time webcam inference script** with subtitle overlay via OpenCV.
