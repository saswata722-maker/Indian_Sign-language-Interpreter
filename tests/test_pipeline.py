import unittest
import numpy as np
from preprocessing.features import mask_missing, add_velocity, prepare_sequence, window_quality
from preprocessing.augment_data import random_rotation, random_translation, random_scaling, spatial_jitter, time_warp
from inference.runtime import StreamState


def features(n=64):
    data = np.random.default_rng(42).normal(size=(n, 228)).astype(np.float32)
    data[:, -3:] = 1
    return data


class FakePredictor:
    names = ['a', 'b']
    def __init__(self):
        self.probs = np.array([.9, .1])
        self.calls = 0
    def probabilities(self, raw, timestamps=None):
        self.calls += 1
        return self.probs.copy()


class FeatureTests(unittest.TestCase):
    def test_augmentation_preserves_flags_missing_and_rotation_geometry(self):
        data = features()
        data[:, -2] = 0
        data[:, 99:162] = -2  # legacy sentinel must not be spatially transformed
        for fn in (random_rotation, random_translation, random_scaling, spatial_jitter):
            out = fn(data)
            np.testing.assert_array_equal(out[:, -3:], data[:, -3:])
            np.testing.assert_array_equal(out[:, 99:162], data[:, 99:162])
        out = random_rotation(data)
        np.testing.assert_allclose(out[:, 2:99:3], data[:, 2:99:3])
        np.testing.assert_allclose(np.linalg.norm(out[:, :3]-out[:, 3:6], axis=1),
                                   np.linalg.norm(data[:, :3]-data[:, 3:6], axis=1), rtol=1e-5)

    def test_time_warp_never_reverses(self):
        data = np.repeat(np.arange(64, dtype=np.float32)[:, None], 228, axis=1)
        for _ in range(100):
            out = time_warp(data)
            self.assertTrue(np.all(np.diff(out[:, 0]) >= 0))
            self.assertEqual(out[0, 0], 0)
            self.assertEqual(out[-1, 0], 63)

    def test_v2_missing_and_velocity_transitions(self):
        data = features(3)
        data[1, -2] = 0
        out = mask_missing(data)
        self.assertTrue(np.all(out[1, 99:162] == 0))
        vel = add_velocity(out, 2)[:, 228:]
        self.assertTrue(np.all(vel[:, -3:] == 0))
        self.assertTrue(np.all(vel[:, 99:162] == 0))

    def test_legacy_preprocessing_and_irregular_timestamps(self):
        data = features(64)
        cfg = {'preprocessing': {'target_seq_len':64, 'use_velocity':True}, 'model': {'input_dim':456}}
        expected = np.concatenate([data, np.diff(data, axis=0, prepend=data[:1])], axis=1)
        np.testing.assert_array_equal(prepare_sequence(data, cfg), expected)
        cfg['preprocessing']['target_seq_len'] = 3
        data = np.zeros((3, 228), np.float32)
        data[:, 0] = [0, 1, 4]
        out = prepare_sequence(data, cfg, [0, 1, 4])
        np.testing.assert_allclose(out[:, 0], [0, 2, 4])

    def test_gate_ignores_flags_and_missing_jumps(self):
        data = np.zeros((4, 228), np.float32)
        data[:, -3] = 1
        data[::2, -2] = 1
        data[1::2, 99:162] = -100
        self.assertEqual(window_quality(data)[2], 0)


class StreamTests(unittest.TestCase):
    def make_state(self, **extra):
        cfg = dict(window_seconds=1, max_buffer_frames=20, min_frames=3,
                   smoothing_window=5, gate={'enabled':False}, confidence_threshold=.5)
        cfg.update(extra)
        return StreamState(FakePredictor(), cfg)

    def test_rejection_cannot_grow_buffer(self):
        state = self.make_state()
        for i in range(500):
            state.push(features(1)[0], i/10, False)
            self.assertLessEqual(len(state.buffer), 11)
        self.assertEqual(state.predictor.calls, 0)

    def test_reject_out_of_order_and_reset_on_gap(self):
        state = self.make_state()
        state.push(features(1)[0], 1, True)
        with self.assertRaises(ValueError):
            state.push(features(1)[0], .9, True)
        state.push(features(1)[0], 10, True)
        self.assertEqual(len(state.buffer), 1)

    def test_low_confidence_clears_label(self):
        state = self.make_state()
        for i in range(11):
            out = state.push(features(1)[0], i/10, True)
        self.assertEqual(out['label'], 'a')
        state.predictor.probs = np.array([.49, .48])
        out = state.push(features(1)[0], 1.1, True)
        self.assertEqual(out['label'], 'Uncertain')
        self.assertEqual(out['confidence'], 0)
        self.assertEqual(len(state.recent), 0)

    def test_complete_segment_static_sign(self):
        state = self.make_state(gate={'enabled':True, 'min_motion':1})
        data = features(1)[0]
        for i in range(10):
            out = state.push(data, i/10, True, 'segment', i == 9)
        self.assertEqual(out['label'], 'a')
        self.assertEqual(state.predictor.calls, 1)
        self.assertEqual(len(state.buffer), 0)

    def test_streams_are_independent(self):
        first, second = self.make_state(), self.make_state()
        first.push(features(1)[0], 0, True)
        self.assertEqual(len(second.buffer), 0)


if __name__ == '__main__':
    unittest.main()
