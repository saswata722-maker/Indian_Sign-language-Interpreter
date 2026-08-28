# Kaggle Landmark Extraction (cloud) -> Local Training (GPU)

Extract INCLUDE video landmarks in the free Kaggle cloud so your local CPU stays
free, then **train on your RTX 5050 locally** with the same code.

## Why
- **Extraction is CPU-bound** (MediaPipe Holistic) -> do it on Kaggle's free CPU.
- **Training is GPU-bound** -> do it on your local GPU.
- The `.npy` landmark files are portable float32 arrays; `make_splits.py` and
  `train.py` read them by path, so nothing changes locally.

## Steps

### 1. Create the notebook on Kaggle
1. Go to https://www.kaggle.com -> **Create -> New Notebook**.
2. Set **Accelerator = CPU** (not GPU — MediaPipe is CPU only). This keeps your
   free GPU quota untouched.
3. **Import** / paste the contents of `ISL_landmark_extraction.ipynb`
   (File -> Import Notebook -> upload this file).
4. Click **+ Add Input** (right pane) -> **Kaggle Datasets** -> search
   `daskoushik/include` -> **Add**. This mounts the INCLUDE videos under
   `/kaggle/input/include`.

### 2. Run the cells in order
1. **Setup** — `pip install` the pinned deps.
2. **Write source** (x2) — recreates `landmark_extractor.py` and
   `extract_landmarks.py` in the notebook (kept byte-identical to the repo).
3. **Locate dataset** — auto-finds the video directory.
4. **Smoke test** — extracts 5 videos; verify shapes/NaNs printouts.
5. **Full extraction** — set `INCLUDE_FACE` (see *Face dimension warning*
   below). ~1–4 h on Kaggle CPU, resumable if a session timeout interrupts it.
6. **Zip** — creates `landmarks.zip`, download it from the output panel.

### 3. Train locally
```powershell
$py = "C:\Users\saswa\AppData\Local\Programs\Python\Python312\python.exe"

# unzip the downloaded landmarks into the project
Expand-Archive "$HOME\Downloads\landmarks.zip" -DestinationPath data\landmarks

# verify it looks right
(Get-ChildItem data\landmarks -Recurse -Filter *.npy).Count   # expect ~4284

# build INCLUDE-50 splits + label map, then train on the GPU
& $py utils\make_splits.py --top_n 50
& $py training\train.py
```

## Face dimension warning (MUST match)
Whichever `INCLUDE_FACE` you pick in the notebook MUST match
`training/config.yaml` -> `data.include_face`, because the feature dimension
changes:
- `INCLUDE_FACE = True`  -> `D = 1632`, config `include_face: true`  (default)
- `INCLUDE_FACE = False` -> `D = 234`,  config `include_face: false` (3x faster)

Since `best.pt` stores `config.data.include_face` and `live_predict.py` reads it
from the checkpoint, inference stays consistent automatically — just keep
`train.py` on the matching `include_face`.

## Reproducing the notebook from source
The notebook is generated from the repo by `kaggle/_build_notebook.py`, which
embeds `models/landmark_extractor.py` and `preprocessing/extract_landmarks.py`
so the cloud copy is byte-identical to what you'd run locally. Re-run it after
changing either source file:
```powershell
& $py kaggle\_build_notebook.py
```
