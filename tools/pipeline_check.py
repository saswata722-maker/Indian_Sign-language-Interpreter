"""Replay held-out raw videos through the same StreamState used by webcam/API.

Reports complete-clip accuracy separately from the final rolling-window result.
This is an isolated-video replay, NOT an unseen-signer or continuous-stream test.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
import numpy as np
import torch
import yaml
from inference.runtime import Predictor, StreamState
from utils.make_splits import select_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='models/best_model.pth')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--landmarks', default='landmarks')
    parser.add_argument('--raw', default='raw_data')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--limit', type=int, default=30, help='0 evaluates all selected clips')
    parser.add_argument('--fps', type=float, default=10, help='Replay sampling cadence, not source-video FPS')
    parser.add_argument('--output', default='pipeline_replay.json')
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error('--fps must be positive')
    with open(args.config) as f:
        config = yaml.safe_load(f)
    predictor = Predictor(args.checkpoint, config, 'cuda' if torch.cuda.is_available() else 'cpu')
    files, labels = select_split(predictor.checkpoint, args.landmarks, args.split)
    indices = np.random.default_rng(42).permutation(len(files))
    if args.limit:
        indices = indices[:args.limit]
    rows = []
    for index in indices:
        landmark = files[index]
        base = Path(args.raw)/landmark.relative_to(args.landmarks).with_suffix('')
        candidates = [base.with_suffix(ext) for ext in ('.MOV', '.mp4', '.mov', '.MP4')]
        video = next((p for p in candidates if p.exists()), None)
        if video is None:
            raise FileNotFoundError(f'No raw video for {landmark}')
        extractor = predictor.make_extractor()  # independent tracker per video
        stream = StreamState(predictor, config['inference'])
        cap = cv2.VideoCapture(str(video))
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0:
            raise ValueError(f'Missing frame rate: {video}')
        features, frame_no, next_time = [], 0, 0.0
        result = stream.result('buffering')
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = frame_no/source_fps
                frame_no += 1
                if timestamp+1e-6 < next_time:
                    continue
                next_time += 1/args.fps
                feat = extractor.process_frame(frame)
                features.append(feat)
                result = stream.push(feat, timestamp, extractor.anchor_valid)
        finally:
            cap.release()
            extractor.close()
        if not features:
            raise ValueError(f'Empty video: {video}')
        probs = predictor.probabilities(np.asarray(features))
        true = predictor.names[labels[index]]
        row = dict(video=str(video), true=true, frames=len(features),
                   clip_prediction=predictor.names[int(probs.argmax())],
                   stream_prediction=result['label'], stream_status=result['status'])
        rows.append(row)
        print(row, flush=True)
    if not rows:
        raise ValueError('No clips selected')
    accepted = [r for r in rows if r['stream_status'] == 'predicting']
    report = dict(protocol='isolated held-out clip replay; final window only', split=args.split,
        fps=args.fps, count=len(rows),
        clip_accuracy=sum(r['clip_prediction']==r['true'] for r in rows)/len(rows),
        stream_effective_accuracy=sum(r['stream_prediction']==r['true'] for r in accepted)/len(rows),
        stream_rejection_rate=1-len(accepted)/len(rows), rows=rows)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print({k:v for k,v in report.items() if k != 'rows'})


if __name__ == '__main__':
    main()
