import contextlib
import csv
import hashlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

import alpr_auto as app


class AutomaticPlateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def row(self, box=(10, 10, 60, 40), text='59A00123', score=.9):
        row = dict.fromkeys(app.DETECTION_FIELDS, '')
        row.update(source='test.mp4', frame_index=0, text=text, ocr_score=score,
                   x1=box[0], y1=box[1], x2=box[2], y2=box[3], needs_review=0, status='ok')
        return row

    def test_letterbox_inverse_coordinates_and_empty_or_invalid_output(self):
        frame = np.zeros((100, 200, 3), np.uint8)
        frame[:, :, 2] = 255
        tensor, ratio, padding = app.letterbox(frame, 100)
        self.assertEqual(tensor.shape, (1, 3, 100, 100))
        np.testing.assert_allclose(tensor[0, :, 40, 40], [1, 0, 0])
        rows = app.decode_boxes(np.array([[0, 5, 30, 25, 40, 0, .8]]), ratio, padding, frame.shape, .35)
        self.assertEqual(rows[0]['box'], [10, 10, 50, 30])
        self.assertEqual(app.decode_boxes(np.empty((0, 7)), ratio, padding, frame.shape, .35), [])
        clipped = app.decode_boxes(np.array([[0, -10, 0, 500, 500, 0, .8]]), ratio, padding, frame.shape, .35)
        self.assertEqual(clipped[0]['box'], [0, 0, 200, 100])
        with self.assertRaisesRegex(ValueError, 'output'):
            app.decode_boxes(np.zeros((1, 5)), ratio, padding, frame.shape, .35)
        with self.assertRaisesRegex(ValueError, 'NaN'):
            app.decode_boxes(np.full((1, 7), np.nan), ratio, padding, frame.shape, .35)

    def test_one_and_two_row_geometry_preserves_both_rows(self):
        crop = np.full((64, 96, 3), 255, np.uint8)
        crop[8:22, 15:80] = 0
        crop[42:56, 15:80] = 0
        regions, split = app.line_regions(crop)
        self.assertEqual(len(regions), 2)
        self.assertGreater(split, 22)
        self.assertLess(split, 42)
        self.assertGreaterEqual(regions[0][1], 22)
        self.assertLessEqual(regions[1][0], 42)
        self.assertEqual(app.line_regions(crop, 'one'), ([(0, 64)], None))
        self.assertEqual(len(app.line_regions(np.full((32, 160, 3), 255, np.uint8))[0]), 1)

    def test_rectification_boundary_and_missing_boundary_fallback(self):
        crop = np.full((100, 180, 3), 60, np.uint8)
        polygon = np.array([[15, 25], [160, 10], [170, 85], [25, 95]], np.int32)
        cv2.fillConvexPoly(crop, polygon, (240, 240, 240))
        cv2.polylines(crop, [polygon], True, (0, 0, 0), 2)
        rectified, applied = app.rectify_plate(crop)
        self.assertTrue(applied)
        self.assertLess(rectified.shape[0], crop.shape[0])
        self.assertLess(rectified.shape[1], crop.shape[1])
        self.assertGreater(float(rectified[rectified.shape[0] // 2, rectified.shape[1] // 2].mean()), 200)
        blank = np.full((64, 100, 3), 255, np.uint8)
        result, applied = app.rectify_plate(blank)
        self.assertFalse(applied)
        np.testing.assert_array_equal(result, blank)

    def test_tracking_one_to_one_votes_expiry_and_no_mutable_alias(self):
        tracker = app.PlateTracks('test.mp4')
        first = [self.row(), self.row((100, 10, 150, 40), '30A12345')]
        self.assertEqual(tracker.update(first, 0), [])
        self.assertEqual([r['track_id'] for r in first], [1, 2])
        second = [self.row((102, 10, 152, 40), '30A12345'), self.row((12, 10, 62, 40), '59A0012S')]
        tracker.update(second, .1)
        self.assertEqual([r['track_id'] for r in second], [2, 1])
        third = [self.row((14, 10, 64, 40))]
        tracker.update(third, .2)
        third[0]['text'] = 'MUTATED'
        events = tracker.update([], 2)
        winner = next(e for e in events if e['track_id'] == 1)
        self.assertEqual(winner['text'], '59A00123')
        self.assertEqual(winner['votes'], 2)
        self.assertEqual(winner['observations'], 3)
        self.assertEqual(len(events), 2)
        self.assertEqual(tracker.finish(), [])
        row = self.row()
        tracker.update([row], 3)
        self.assertEqual(row['track_id'], 3)

    def test_default_download_checksum_and_partial_file_cleanup(self):
        content = b'test-only-model-bytes'
        with patch.object(app, '__file__', str(self.root / 'alpr_auto.py')), \
             patch.object(app, 'MODEL_SHA256', hashlib.sha256(content).hexdigest()), \
             patch.object(app, 'urlopen', return_value=io.BytesIO(content)), contextlib.redirect_stdout(io.StringIO()):
            path = app.detector_model()
            self.assertEqual(path.read_bytes(), content)
        path.unlink()
        with patch.object(app, '__file__', str(self.root / 'alpr_auto.py')), \
             patch.object(app, 'urlopen', return_value=io.BytesIO(content)), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                app.detector_model()
        self.assertFalse(path.exists())
        self.assertEqual(list(path.parent.glob('*.part')), [])

    def run_mocked(self, source, output, no_plate=False, failure=False, extra=None):
        model = self.root / 'mock.onnx'
        model.write_bytes(b'mock')
        class Detector:
            def __init__(self, *args):
                self.session = types.SimpleNamespace(get_providers=lambda: ['CPUExecutionProvider'])
                self.calls = 0
            def predict(self, frame):
                self.calls += 1
                if failure and self.calls > 1:
                    raise RuntimeError('model execution failed')
                return [] if no_plate else [{'box': [10, 10, 80, 40], 'score': .9}]
        class Reader:
            def __init__(self, *args):
                self.torch = types.SimpleNamespace(__version__='TEST_STUB')
            def read(self, crop, layout): return ['59A00123'], .95, None
        with patch.object(app, 'detector_model', return_value=model), \
             patch.object(app, 'PlateDetector', Detector), patch.object(app, 'ParseqReader', Reader), \
             patch('builtins.input', side_effect=AssertionError('Automatic inference must never ask for labels')), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = app.main(['--source', str(source), '--output', str(output)] + (extra or []))
        report = json.loads((output / 'summary.json').read_text())
        def read(name):
            with (output / name).open(encoding='utf-8-sig', newline='') as handle:
                return list(csv.DictReader(handle))
        return status, report, read

    def test_image_cli_is_automatic_and_no_detection_is_explicit(self):
        source = self.root / 'biển số.png'
        Image.new('RGB', (100, 60), 'white').save(source)
        status, report, read = self.run_mocked(source, self.root / 'image')
        self.assertEqual(status, 0)
        self.assertIsNone(report['accuracy'])
        self.assertEqual(report['detections'], 1)
        self.assertEqual(read('events.csv')[0]['text'], '59A00123')
        self.assertTrue((self.root / 'image' / '001_biển số.jpg').is_file())
        status, report, read = self.run_mocked(source, self.root / 'empty', no_plate=True)
        self.assertEqual(status, 0)
        self.assertEqual(read('detections.csv'), [])
        self.assertEqual(read('frames.csv')[0]['status'], 'no_plate')
        self.assertEqual(report['events'], 0)

    def test_video_sampling_preview_csv_consensus_and_failure_reporting(self):
        source = self.root / 'test.avi'
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10, (100, 60))
        self.assertTrue(writer.isOpened())
        for _ in range(6):
            writer.write(np.full((60, 100, 3), 255, np.uint8))
        writer.release()
        status, report, read = self.run_mocked(source, self.root / 'video', extra=['--every-n-frames', '2', '--max-frames', '2'])
        self.assertEqual(status, 0)
        self.assertEqual([r['frame_index'] for r in read('frames.csv')], ['0', '2'])
        self.assertEqual([r['timestamp_ms_approx'] for r in read('frames.csv')], ['0.0', '200.0'])
        self.assertEqual(report['processed_frames'], 2)
        self.assertEqual(read('events.csv')[0]['observations'], '2')
        capture = cv2.VideoCapture(str(self.root / 'video' / '001_preview.mp4'))
        self.assertTrue(capture.isOpened())
        self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
        capture.release()
        status, report, read = self.run_mocked(source, self.root / 'failed', failure=True)
        self.assertEqual(status, 1)
        self.assertEqual(report['status'], 'failed')
        self.assertIn('model execution failed', report['error'])
        self.assertEqual(read('frames.csv')[0]['status'], 'error')


if __name__ == '__main__':
    unittest.main()
