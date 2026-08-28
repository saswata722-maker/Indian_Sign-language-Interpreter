# ISL Real-Time Video Translator (Indian Sign Language)

Converts live webcam video into on-screen subtitles by recognizing ISL
word signs. Landmark-based pipeline: MediaPipe Holistic per-frame
landmarks -> hybrid LSTM + Transformer (PyTorch) -> predicted word.

## 1. Environment (already verified on this machine)

- Python 3.12: `C:\Users\saswa\AppData\Local\Programs\Python\Python312\python.exe`
  (the only env with CUDA torch — always use this absolute path)
- torch 2.11.0+cu128 (RTX 5050 Laptop GPU, sm_120 — verified working)
- Other deps: `pip install -r requirements.txt` (mediapipe 0.10.21 with Holistic)

## 2. Pipeline

```
raw_data/  (INCLUDE videos, already downloaded)
   |  python preprocessing/extract_landmarks.py
   v
data/landmarks/<Category>/<Word>/*.npy      (T, 1632) float32 sequences
   |  python utils/make_splits.py --top_n 50
   v
data/splits/{train,val,test}.csv + utils/label_map.json   (INCLUDE-50)
   |  python training/train.py --config training/config.yaml
   v
models/checkpoints/best.pt
   |  python training/evaluate.py
   |  python inference/live_predict.py
   v
Webcam window with live subtitle overlay
```

## 3. Quick start

```powershell
$py = "C:\Users\saswa\AppData\Local\Programs\Python\Python312\python.exe"

# 1) Landmark extraction (resumable; ~3-6 h for all 4284 videos; run overnight)
& $py preprocessing\extract_landmarks.py --limit_videos 5   # smoke test first
& $py preprocessing\extract_landmarks.py                    # full run

# 2) Splits + label map (INCLUDE-50 = 50 most-populated word classes)
& $py utils\make_splits.py --top_n 50

# 3) Sanity training run (tiny subset, should overfit fast)
& $py training\train.py --include 10

# 4) Full INCLUDE-50 training
& $py training\train.py

# 5) Test-set evaluation + confusion matrix (runs\confusion_test.png)
& $py training\evaluate.py

# 6) Live webcam translator
& $py inference\live_predict.py
```

## 4. Configuration

All hyperparameters: `training/config.yaml` (seq_len, d_model, LSTM/Transformer
depths, heads, dropout, batch size, lr, early stopping, inference window/stride).
Set `data.include_face: false` for a lighter 234-dim model (no facial landmarks)
— re-run extraction with `--no_face` first.

## 5. Notes

- The official Zenodo Train_Test_Split.zip is unavailable (404); we use our own
  stratified 80/10/10 split at video level (`utils/make_splits.py`).
- Training/inference share `models/landmark_extractor.py` so features match exactly.
- Augmentation (flip / time-warp / jitter / crop) is applied on the fly in
  `training/dataset.py` via `preprocessing/augment_data.py`.
- INCLUDE-50 paper baseline: 94.5% (test). Realistic target for this from-scratch
  landmark model: 75-90%.
