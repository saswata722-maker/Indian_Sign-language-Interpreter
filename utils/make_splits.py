"""Persistent splits. Optional JSON maps relative .npy paths to signer/session groups."""
import argparse
import json
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from training.dataset import load_dataset


def validate_splits(manifest):
    seen = set()
    all_groups = set()
    for split in ('train', 'val', 'test'):
        entries = manifest[split]
        paths = entries['files']
        if len(paths) != len(entries['labels']) or not paths:
            raise ValueError(f'Invalid or empty {split} split')
        if len(set(paths)) != len(paths) or seen.intersection(paths):
            raise ValueError('Duplicate or overlapping split paths')
        seen.update(paths)
        if 'groups' in entries:
            groups = set(entries['groups'])
            if len(entries['groups']) != len(paths) or all_groups.intersection(groups):
                raise ValueError('Signer/session groups overlap')
            all_groups.update(groups)
    mapping = manifest['class_to_idx']
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError('Invalid class mapping')
    train_labels = set(manifest['train']['labels'])
    if train_labels != set(mapping.values()):
        raise ValueError('Training split must contain every class')
    for split in ('train', 'val', 'test'):
        for path, label in zip(manifest[split]['files'], manifest[split]['labels']):
            if mapping.get(Path(path).parent.name) != label:
                raise ValueError(f'Label mapping mismatch: {path}')
    return manifest


def make_splits(landmarks_dir='landmarks', val_split=.2, test_split=.15,
                random_seed=42, output_file='splits.json', groups_file=None):
    if not (0 < val_split < 1 and 0 < test_split < 1 and val_split+test_split < 1):
        raise ValueError('Invalid split fractions')
    root = Path(landmarks_dir)
    files, labels, mapping = load_dataset(root)
    paths = [f.relative_to(root).as_posix() for f in files]
    ids = list(range(len(paths)))
    groups = None
    if groups_file:
        group_map = json.loads(Path(groups_file).read_text())
        groups = [group_map[p] for p in paths]  # fail rather than invent signer IDs
        def split(indices, size, seed):
            # Select a group-disjoint partition with class coverage when feasible.
            splitter = GroupShuffleSplit(n_splits=100, test_size=size, random_state=seed)
            for a, b in splitter.split(indices, groups=[groups[i] for i in indices]):
                left, right = [indices[i] for i in a], [indices[i] for i in b]
                if set(labels[i] for i in left) == set(mapping.values()) and set(labels[i] for i in right) == set(mapping.values()):
                    return left, right
            raise ValueError('Cannot form group-disjoint splits with every class; collect more signer/session data')
    else:
        warnings.warn('Clip-random split: this does NOT measure unseen-signer accuracy')
        def split(indices, size, seed):
            return train_test_split(indices, test_size=size, stratify=[labels[i] for i in indices], random_state=seed)
    train_val, test = split(ids, test_split, random_seed)
    train, val = split(train_val, val_split/(1-test_split), random_seed)
    manifest = {'class_to_idx': mapping, 'protocol': 'group-disjoint' if groups else 'clip-random',
                'seed': random_seed, 'path_format': 'relative-to-landmarks'}
    for name, indices in [('train', train), ('val', val), ('test', test)]:
        manifest[name] = {'files': [paths[i] for i in indices], 'labels': [labels[i] for i in indices]}
        if groups:
            manifest[name]['groups'] = [groups[i] for i in indices]
    validate_splits(manifest)
    # Never overwrite a previously chosen test set by accident.
    with open(output_file, 'x') as f:
        json.dump(manifest, f, indent=2)
    return manifest


def select_split(checkpoint, landmarks_dir, split='val'):
    manifest = checkpoint.get('splits')
    if manifest:
        validate_splits(manifest)
        if manifest['class_to_idx'] != checkpoint['class_to_idx']:
            raise ValueError('Split and checkpoint class mappings differ')
        entry = manifest[split]
        files = [Path(landmarks_dir)/p for p in entry['files']]
        labels = entry['labels']
    else:
        if split != 'val':
            raise ValueError('Legacy checkpoint has no held-out test manifest; use val or retrain with saved splits')
        warnings.warn('Reconstructed legacy validation split; valid only if dataset contents/order are unchanged. Not an independent test set.')
        files, labels, mapping = load_dataset(landmarks_dir)
        if mapping != checkpoint['class_to_idx']:
            raise ValueError('Dataset class mapping differs from checkpoint')
        cfg = checkpoint['config']['training']
        _, files, _, labels = train_test_split(files, labels, test_size=cfg['val_split'],
                                              stratify=labels, random_state=cfg['random_seed'])
    if not all(p.is_file() for p in files):
        raise FileNotFoundError('Split references missing landmark files')
    return files, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--landmarks', default='landmarks')
    parser.add_argument('--groups', help='JSON: relative landmark path -> signer/session group')
    parser.add_argument('--output', default='splits.json')
    args = parser.parse_args()
    make_splits(args.landmarks, output_file=args.output, groups_file=args.groups)
