"""Clip evaluation on saved splits; never silently score the training set."""
import argparse
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from training.dataset import ISLLandmarkDataset
from inference.runtime import Predictor
from utils.make_splits import select_split


def evaluate_model(model_path, landmarks_dir='landmarks', config_path='config.yaml', split='val', threshold=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    predictor = Predictor(model_path, config, 'cuda' if torch.cuda.is_available() else 'cpu')
    files, labels = select_split(predictor.checkpoint, landmarks_dir, split)
    prep = predictor.config['preprocessing']
    loader = DataLoader(ISLLandmarkDataset(files, labels, target_len=prep['target_seq_len'],
        use_velocity=prep.get('use_velocity', False), feature_version=prep.get('feature_version', 1)),
        batch_size=64, shuffle=False)
    predictions, confidence = [], []
    with torch.inference_mode():
        for x, _ in loader:
            probs = predictor.model(x.to(predictor.device)).softmax(1)
            values, indices = probs.max(1)
            predictions.extend(indices.cpu().tolist())
            confidence.extend(values.cpu().tolist())
    predictions, labels = np.asarray(predictions), np.asarray(labels)
    correct = predictions == labels
    print(f'{split}: {correct.sum()}/{len(labels)} = {correct.mean():.2%} (complete clips)')
    print(classification_report(labels, predictions, labels=list(range(len(predictor.names))),
                                target_names=predictor.names, zero_division=0))
    if threshold is not None:
        accepted = np.asarray(confidence) >= threshold
        print(f'Accepted-only: {correct[accepted].mean() if accepted.any() else 0:.2%}; '
              f'effective: {(correct & accepted).mean():.2%}; rejection: {(~accepted).mean():.2%}')
        print('Threshold-only clip evaluation; use tools/pipeline_check.py for streaming replay.')
    return float(correct.mean()), predictions, labels


def evaluate_live_style(model_path, landmarks_dir='landmarks', config_path='config.yaml', split='val'):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return evaluate_model(model_path, landmarks_dir, config_path, split,
                          config['inference']['confidence_threshold'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', choices=['clip', 'live'], default='clip')
    parser.add_argument('--checkpoint', default='models/best_model.pth')
    parser.add_argument('--landmarks', default='landmarks')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    args = parser.parse_args()
    fn = evaluate_live_style if args.mode == 'live' else evaluate_model
    fn(args.checkpoint, args.landmarks, args.config, args.split)
