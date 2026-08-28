# Implementation Plan - Real-Time Indian Sign Language Video Translator

## [Overview]
Build an end-to-end, fully local real-time ISL translator: webcam video in -> recognized word as subtitle overlay out. Videos from the INCLUDE dataset (already downloaded in raw_data/, 4284 videos, 277 word folders) are converted to per-frame MediaPipe landmark sequences; a hybrid LSTM->Transformer PyTorch model is trained on INCLUDE-50 (50 word classes) first, with the full 263-class vocabulary as a later extension. All heavy work (extraction, training) runs locally on the RTX 5050 Laptop GPU (8 GB) using Python 3.12 + torch 2.11.0+cu128 + mediapipe 0.10.21 (legacy mp.solutions.holistic).

Environment decisions:
- Use C:/Users/saswa/AppData/Local/Programs/Python/Python312/python.exe everywhere (the ONLY env with working CUDA torch; verified sm_120 GPU matmul).
- mediapipe 0.10.21 provides mp.solutions.holistic (pose 33, hands 2x21, face 468 landmarks) matching PROJECT_CONTEXT.
- Train_Test_Split.zip is NOT on Zenodo (404) -> custom stratified split 80/10/10 at video level.
- INCLUDE-50 word list: the 50 most-populated word folders (robust fallback since official CSVs are unavailable).

## [Types]
- Per-frame feature vector (float32): pose 33x3=99 + hands 2x21x3=126 (zeros+flag when absent) + face 468x3=1404 (config include_face) + 3 detection flags. include_face=true -> D=1632; false -> D=234.
- Normalization: origin at mid-shoulder, scale by shoulder width; identical for train and inference (shared extractor).
- config.yaml schema: data{raw_dir, landmarks_dir, seq_len:96, include_face, num_classes:50}, model{d_model:256, lstm_layers:2, transformer_layers:4, nhead:8, ff_dim:512, dropout:0.3}, train{epochs:80, batch_size:32, lr:1e-3, weight_decay:1e-4, patience:15, num_workers:4, seed:42}, inference{window:96, stride:8, smooth_k:5, conf_threshold:0.6}.
- utils/label_map.json {idx: word}; .npy landmark files as (T,D) float32, T<=200.

## [Files]
New: models/landmark_extractor.py, preprocessing/extract_landmarks.py, preprocessing/augment_data.py, training/config.yaml, training/dataset.py, training/train.py, training/evaluate.py, models/model.py, inference/live_predict.py, utils/metrics.py, utils/make_splits.py, README.md.
Modified: requirements.txt.

## [Functions]
New: extract_video(path)->(T,D), build_feature_frame(...) (landmark_extractor.py); augment(seq) (augment_data.py); build_label_map(), make_splits() (make_splits.py); pad_or_truncate() (dataset.py); train_one_epoch(), evaluate() (train.py/evaluate.py); predict_window() (live_predict.py).

## [Classes]
- HybridLSTMTransformer(nn.Module) (models/model.py): Linear projection -> LSTM -> positional encoding -> TransformerEncoder -> masked mean-pool -> classifier.
- LandmarkDataset(Dataset) (training/dataset.py).
- LandmarkExtractor (models/landmark_extractor.py): owns mp.solutions.holistic.Holistic; method extract(video_path)->(T,D).
- LiveRecognizer (inference/live_predict.py): ring buffer, sliding window prediction, moving-majority smoothing, cv2 subtitle overlay.

## [Dependencies]
- Existing: torch 2.11.0+cu128, numpy.
- Installed: mediapipe==0.10.21, opencv-contrib-python, pandas, scikit-learn, pyyaml, matplotlib, tqdm, onnx, onnxruntime.

## [Testing]
Extraction smoke test (shapes/NaNs/missing-hand flags) -> dataset tests -> overfit sanity run -> full INCLUDE-50 training (target 75-85% test acc) -> confusion matrix -> live webcam demo (>=10 fps).

## [Implementation Order]
1. Env install; 2. scaffold; 3. landmark extractor + smoke test; 4. full extraction; 5. splits + label map; 6. model; 7. dataset+config; 8. train; 9. evaluate; 10. live demo; 11. README/ONNX/263-class scale-up.
