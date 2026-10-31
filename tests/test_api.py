"""Integration tests exercise HTTP validation, ordering, reset, and stream isolation."""
import base64
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from fastapi.testclient import TestClient
from api import main


class FakeExtractor:
    def __init__(self):
        self.anchor_valid = True
        self.mp_holistic = self
        self.reset_count = 0
    def process_frame(self, frame):
        feature = np.zeros(228, np.float32)
        feature[-3:] = 1
        return feature
    def reset_anchor(self):
        self.reset_count += 1
    def reset(self):
        pass
    def close(self):
        pass


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.patch = patch.object(main.predictor, 'make_extractor', side_effect=FakeExtractor)
        self.patch.start()
        self.client = TestClient(main.app)
        self.client.__enter__()
        _, image = cv2.imencode('.jpg', np.zeros((32, 32, 3), np.uint8))
        self.image = base64.b64encode(image).decode()
    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.patch.stop()
    def send(self, client_id, frame_id, timestamp, **extra):
        return self.client.post('/predict', json=dict(client_id=client_id, frame_id=frame_id,
            timestamp=timestamp, image_b64=self.image, **extra))
    def test_sessions_and_order(self):
        self.assertEqual(self.send('a', 0, 0).status_code, 200)
        self.assertEqual(self.send('a', 0, .1).status_code, 409)
        self.assertEqual(self.send('a', 1, 0).status_code, 409)
        self.assertEqual(self.send('b', 0, 0).status_code, 200)
        self.assertIsNot(main.clients['a'].extractor, main.clients['b'].extractor)
        self.client.post('/reset', json={'client_id':'a'})
        self.assertEqual(len(main.clients['a'].stream.buffer), 0)
        self.assertEqual(len(main.clients['b'].stream.buffer), 1)
    def test_segment_prediction_and_no_stale_result(self):
        with patch.object(main.predictor, 'probabilities', return_value=np.r_[.9, np.full(261, .1/261)]):
            for i in range(8):
                response = self.send('a', i, i*.1, mode='segment', finalize=i == 7)
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertEqual(result['status'], 'predicting')
            self.assertAlmostEqual(result['confidence'], .9)
        self.assertEqual(len(main.clients['a'].stream.buffer), 0)
    def test_invalid_image_and_schema(self):
        self.assertEqual(self.client.post('/predict', json={'client_id':'x'}).status_code, 422)
        response = self.client.post('/predict', json=dict(client_id='x', frame_id=0,
                                     timestamp=0, image_b64='bad base64'))
        self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main()
