import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
KAGGLE = pathlib.Path(__file__).resolve().parent

src_extractor = (ROOT / "models" / "landmark_extractor.py").read_text(encoding="utf-8")
src_extract = (ROOT / "preprocessing" / "extract_landmarks.py").read_text(encoding="utf-8")


def markdown(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}


def write_py_cell(relpath: str, content: str) -> str:
    """Cell source that writes <content> to <relpath>.

    Uses a RAW triple-quoted string so backslashes (e.g. the '\\n' inside
    f-strings of the source) are preserved literally when the cell runs.
    """
    esc = content.replace("'''", "\\'\\'\\'")
    return (
        f"import pathlib\n"
        f"_p = pathlib.Path({relpath!r})\n"
        f"_p.parent.mkdir(parents=True, exist_ok=True)\n"
        f"_p.write_text(r'''{esc}''', encoding='utf-8')\n"
        f"print('wrote', _p, _p.stat().st_size, 'bytes')\n"
    )


cells = []


cells.append(
    markdown(
        "# ISL Landmark Extraction (Kaggle)\n\n"
        "Runs INCLUDE video -> MediaPipe Holistic -> `.npy` landmarks in the cloud so your "
        "local CPU stays free. **Train locally afterwards** on your GPU.\n\n"
        "## Workflow\n"
        "1. Run the **Setup** cell (installs pinned deps).\n"
        "2. Run the two **Write source** cells.\n"
        "3. Run **Locate dataset** (mount the INCLUDE dataset via right-side "
        "`+ Add data` -> Kaggle Datasets -> `daskoushik/include`).\n"
        "4. **Smoke test** 5 videos.\n"
        "5. **Full extraction** (about 1-4 h on Kaggle CPU; resumable).\n"
        "6. **Zip** the landmarks, then download `landmarks.zip` from the output panel.\n\n"
        "Unzip locally into `data/landmarks/`, then run `utils/make_splits.py --top_n 50` "
        "and `training/train.py` on your GPU.\n"
    )
)

cells.append(code("!pip install -q mediapipe==0.10.21 opencv-contrib-python numpy tqdm"))
cells.append(code(write_py_cell("models/landmark_extractor.py", src_extractor)))
cells.append(code(write_py_cell("preprocessing/extract_landmarks.py", src_extract)))

cells.append(
    code(
        "import sys, pathlib\n"
        "sys.path.insert(0, '.')\n"
        "from preprocessing.extract_landmarks import find_videos, resolve_raw_dir\n\n"
        "def locate_raw_dir(base=pathlib.Path('/kaggle/input'), "
        "exts=('.MOV','.MP4','.AVI','.MKV')):\n"
        "    hits = [p for p in base.rglob('*') if p.is_file() and p.suffix.upper() in exts]\n"
        "    if not hits:\n"
        "        raise SystemExit('No videos under ' + str(base) + ' - mount INCLUDE dataset')\n"
        "    parents = [set(p.parents) for p in hits]\n"
        "    common = set.intersection(*parents)\n"
        "    return max(common, key=lambda d: len(str(d))), len(hits)\n\n"
        "raw_dir, n = locate_raw_dir()\n"
        "raw_dir = resolve_raw_dir(raw_dir)\n"
        "print('raw_dir ->', raw_dir)\n"
        "jobs = list(find_videos(raw_dir, None))\n"
        "print('videos:', len(jobs), '| unique word folders:', "
        "len({str(p.parent) for p,_ in jobs}))\n"
        "_RAW_DIR = str(pathlib.Path('/kaggle/input'))"
    )
)

cells.append(
    markdown(
        "# 4) Smoke test (5 videos)\n\n"
        "Quick sanity check of the extraction and saved `.npy` format before the long run."
    )
)

cells.append(
    code(
        "import subprocess, sys\n"
        "r = subprocess.run([sys.executable, 'preprocessing/extract_landmarks.py',\n"
        "                    '--raw_dir', _RAW_DIR, '--out_dir', 'data/landmarks',\n"
        "                    '--limit_videos', '5'])\n"
        "print('exit', r.returncode)"
    )
)

cells.append(
    code(
        "import numpy as np, pathlib\n"
        "files = sorted(pathlib.Path('data/landmarks').rglob('*.npy'))\n"
        "print('npy files:', len(files))\n"
        "for f in files[:5]:\n"
        "    a = np.load(f); print(f, a.shape, a.dtype, 'NaN', int(np.isnan(a).sum()))"
    )
)

cells.append(
    markdown(
        "# 5) Full extraction\n\n"
        "Set `INCLUDE_FACE` below. **True -> D=1632 (face on)** is the default and matches "
        "local `include_face: true`. **False -> D=234 (`--no_face`)** is ~3x faster/smaller. "
        "Whichever you pick MUST match `training/config.yaml` `data.include_face` locally. "
        "The job is resumable (skips existing `.npy`). If Kaggle times out, just re-run this cell."
    )
)

cells.append(
    code(
        "import subprocess, sys\n"
        "INCLUDE_FACE = True   # True -> D=1632 ; False -> D=234 (faster)\n"
        "cmd = [sys.executable, 'preprocessing/extract_landmarks.py',\n"
        "       '--raw_dir', _RAW_DIR, '--out_dir', 'data/landmarks']\n"
        "if not INCLUDE_FACE:\n"
        "    cmd.append('--no_face')\n"
        "r = subprocess.run(cmd)\n"
        "print('exit', r.returncode)"
    )
)

cells.append(
    code(
        "import pathlib, shutil\n"
        "src = pathlib.Path('data/landmarks')\n"
        "files = list(src.rglob('*.npy'))\n"
        "print('total npy:', len(files))\n"
        "print('classes (word folders):', "
        "len({f.parent.name for f in files if f.parent.name.lower()!='extra'}))\n"
        "shutil.make_archive('landmarks', 'zip', src)\n"
        "z = pathlib.Path('landmarks.zip')\n"
        "print('zip ready:', z, round(z.stat().st_size/1e6,1), 'MB -> download from output panel')"
    )
)

cells.append(
    markdown(
        "## Download & train locally\n"
        "1. Download `landmarks.zip` from the output panel (File icon, top-right).\n"
        "2. Unzip into the project: "
        "`Expand-Archive landmarks.zip -DestinationPath data/landmarks`.\n"
        "3. Ensure `data.include_face` in `training/config.yaml` equals `INCLUDE_FACE` above.\n"
        "4. `python utils/make_splits.py --top_n 50`\n"
        "5. `python training/train.py` (on your RTX 5050)\n"
        "6. `python training/evaluate.py` + `python inference/live_predict.py`\n"
    )
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = KAGGLE / "ISL_landmark_extraction.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote", out, out.stat().st_size, "bytes")

