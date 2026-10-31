import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import yaml
from utils.make_splits import make_splits, validate_splits, select_split
from training.train import train_isl


class SplitTests(unittest.TestCase):
    def test_train_save_and_evaluate_manifest(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            landmarks = root/'landmarks'
            rng = np.random.default_rng(42)
            for name in ('a', 'b'):
                folder = landmarks/name
                folder.mkdir(parents=True)
                for i in range(12):
                    data = rng.normal(size=(8, 228)).astype(np.float32)
                    data[:, -3:] = 1
                    np.save(folder/f'{i}.npy', data)
            config = yaml.safe_load(Path('config.yaml').read_text())
            config['model'].update(d_model=16, nhead=2, num_transformer_layers=1)
            config['preprocessing']['target_seq_len'] = 8
            config['training'].update(num_epochs=1, batch_size=8, splits_path=str(root/'splits.json'), early_stopping_min_delta=-1)
            config['model_save_dir'] = str(root/'models')
            path = root/'config.yaml'
            path.write_text(yaml.safe_dump(config))
            with contextlib.redirect_stdout(io.StringIO()):
                train_isl(str(landmarks), str(path))
            checkpoint = torch.load(root/'models/best_model.pth', map_location='cpu', weights_only=True)
            manifest = validate_splits(checkpoint['splits'])
            self.assertEqual(checkpoint['config']['preprocessing']['feature_version'], 2)
            files, labels = select_split(checkpoint, landmarks, 'test')
            self.assertTrue(files)
            self.assertEqual(len(files), len(labels))
            self.assertFalse(set(manifest['test']['files']) & set(manifest['train']['files']))
            manifest['test']['files'][0] = manifest['train']['files'][0]
            with self.assertRaises(ValueError):
                validate_splits(manifest)

    def test_legacy_checkpoint_cannot_claim_test_accuracy(self):
        with self.assertRaises(ValueError):
            select_split({}, 'landmarks', 'test')


if __name__ == '__main__':
    unittest.main()
