# ISL-Sense — Indian Sign Language Recognition

MediaPipe landmarks → temporal classifier → isolated-sign predictions.
The repository includes a legacy 262-class checkpoint. It is an isolated-sign
classifier, not a complete continuous sign-language translator.

## Run locally

Use Python 3.12. The MediaPipe/OpenCV versions in `requirements.txt` match the
versions used to verify landmark extraction; install only one OpenCV distribution.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Start the camera, choose **Record one sign**, perform
one complete sign, and select **Finish sign**. Recording automatically finishes
after about 7.5 seconds. The backend also bounds recording history to eight
seconds. The browser sends one request at a time with capture timestamps.
**Continuous preview** is available but is still a rolling-window heuristic.

For the desktop webcam:

```powershell
.venv\Scripts\python.exe -m inference.live_predict
```

SPACE starts/finishes a sign, S changes between segment/stream modes, R resets
tracking and the buffer, and Q quits. Keep shoulders and both hands in frame.

The API keeps per-client state in memory: run one Uvicorn worker, or configure
sticky routing if you build a multi-instance deployment. The browser and API
must be updated together: `/predict` now requires `frame_id` and `timestamp`.

## Training and evaluation

`config.yaml` trains corrected feature version 2 into `models_v2/`, preserving
`models/best_model.pth`. Inference always uses the checkpoint's own preprocessing
and class map. Checkpoints without a feature version use legacy version 1.

For a fresh landmark cache:

```powershell
.venv\Scripts\python.exe -m preprocessing.extract_landmarks --output landmarks_v2
```

Set `landmarks_dir: "landmarks_v2"` in `config.yaml` before training on this cache.
Existing legacy landmarks can also be used: V2 masks missing coordinates on load,
but fresh extraction additionally removes cross-video tracker history.

Create group-disjoint splits if signer/session metadata is available:

```powershell
.venv\Scripts\python.exe -m utils.make_splits --landmarks landmarks_v2 --groups groups.json
```

Otherwise training creates and saves a clip-random split (65% train, 20% validation,
15% test). This does not establish unseen-signer performance. Existing split files
are reused and never silently overwritten. See the research/migration notes for
the group mapping format.

```powershell
.venv\Scripts\python.exe -m training.train
.venv\Scripts\python.exe -m training.evaluate --checkpoint models_v2/best_model.pth --landmarks landmarks_v2 --split test
```

Training now reads paths from `config.yaml`. Evaluation uses only checkpoint
split membership and class mapping. For the shipped checkpoint, only reconstructed
validation is available; there is no saved independent test split:

```powershell
.venv\Scripts\python.exe -m training.evaluate --split val
.venv\Scripts\python.exe tools/pipeline_check.py --split val --limit 30 --fps 10
```

The replay script processes held-out raw videos with the actual streaming runtime.
It reports complete-clip accuracy separately from the final rolling-window result
and rejection rate. It is not a continuous-webcam or unseen-signer benchmark.

To use a newly trained checkpoint in the API:

```powershell
$env:MODEL_PATH = "models_v2/best_model.pth"
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

## Verification

```powershell
.venv\Scripts\python.exe -m pip install httpx
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover geometry/masks, monotonic time warps, version compatibility,
timestamp resampling, bounded buffers, stale predictions, HTTP request ordering,
client isolation, and a miniature train/checkpoint/test-split round trip.

The old all-data report's 97.51% includes training examples. The shipped
checkpoint's validation accuracy is 88.73%; neither is a live-webcam accuracy
measurement. Source fixes still need validation on labeled webcam recordings.

## Research and next steps

See [accuracy fixes, migration, and pretrained-model research](docs/ACCURACY_AND_MODELS.md)
for verified OpenHands INCLUDE checkpoint links, SignCLIP, Uni-Sign, NLA-SLR,
CISLR, and a practical comparison plan. No replacement model is installed.

This project builds on [INCLUDE](https://github.com/AI4Bharat/INCLUDE),
[OpenHands](https://github.com/AI4Bharat/OpenHands), and MediaPipe Holistic.
Respect the individual dataset and checkpoint licenses when redistributing assets.
