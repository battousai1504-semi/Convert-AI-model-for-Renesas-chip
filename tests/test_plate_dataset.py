import csv
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

import annotate_plates as annotation
import evaluate_plate_groups as evaluation


class PlateDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'images').mkdir()
        self.data = {'version': 1, 'plates': []}
        # Distinct pixels ensure duplicate detection does not confuse two lines.
        self.frame = np.arange(40 * 100 * 3, dtype=np.uint16).reshape(40, 100, 3).astype(np.uint8)
        self.info = {'source_name': 'xe.png', 'source_sha256': 'test-source',
                     'group_id': 'test-source', 'frame_index': None, 'timestamp_ms_approx': None}

    def save_two_lines(self):
        return annotation.save_plate(self.root, self.data, self.frame,
                                      [[0, 0, 100, 18], [0, 20, 100, 40]], ['59-A', '001.23'], self.info)

    def test_label_roundtrip_unicode_path_duplicates_and_tamper(self):
        plate = self.save_two_lines()
        loaded = annotation.load_dataset(self.root)
        self.assertEqual(loaded, self.data)
        with (self.root / 'labels.csv').open(encoding='utf-8-sig', newline='') as handle:
            self.assertEqual([r['text'] for r in csv.DictReader(handle)], ['59-A', '001.23'])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.save_two_lines()
        self.assertEqual(len(annotation.load_dataset(self.root)['plates']), 1)
        image = self.root / 'images' / plate['lines'][0]['image']
        Image.new('RGB', (100, 18), 'white').save(image)
        with self.assertRaisesRegex(ValueError, 'modified'):
            annotation.load_dataset(self.root)

    def test_box_scale_cancellation_order_and_rollback(self):
        self.assertEqual(annotation.map_box((10, 5, 20, 10), (50, 100), (100, 200)), [20, 10, 60, 30])
        with self.assertRaises(ValueError):
            annotation.map_box((0, 0, 0, 0), (50, 100), (100, 200))
        with self.assertRaisesRegex(ValueError, 'TOP'):
            annotation.save_plate(self.root, self.data, self.frame,
                                   [[0, 20, 100, 40], [0, 0, 100, 18]], ['00123', '59A'], self.info)
        with patch.object(annotation, 'atomic_text', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.save_two_lines()
        self.assertEqual(list((self.root / 'images').iterdir()), [])
        self.assertEqual(self.data['plates'], [])

    def test_video_sampling_and_unicode_image(self):
        image = self.root / 'biển số.png'
        Image.fromarray(self.frame).save(image)
        frames = list(annotation.sample_frames(image))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0][0].shape, self.frame.shape)
        video = self.root / 'sample.avi'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 5, (100, 40))
        self.assertTrue(writer.isOpened(), 'MJPG test codec unavailable')
        try:
            for _ in range(12):
                writer.write(self.frame)
        finally:
            writer.release()
        frames = list(annotation.sample_frames(video, every_seconds=.4, max_frames=3))
        self.assertEqual([r[1] for r in frames], [0, 2, 4])
        self.assertEqual([r[2] for r in frames], [0, 400, 800])

    def test_whole_plate_metrics_errors_mismatch_and_duplicate(self):
        plate = self.save_two_lines()
        predictions = [{'image': line['image'], 'device': 'cuda', 'prediction': text,
                        'expected_text': line['text'], 'status': 'ok'}
                       for line, text in zip(plate['lines'], ['59A', '0012S'])]
        rows, report = evaluation.evaluate(self.data, predictions)
        self.assertEqual(rows[0]['prediction'], '59A0012S')
        self.assertEqual(report['overall']['exact_plate_accuracy'], 0)
        self.assertEqual(report['overall']['character_error_rate'], 1 / 8)
        self.assertIsNone(report['one_line']['exact_plate_accuracy'])
        predictions[1]['prediction'] = '00123'
        self.assertEqual(evaluation.evaluate(self.data, predictions)[1]['overall']['exact_plate_accuracy'], 1)
        predictions[1]['status'] = 'error'
        self.assertEqual(evaluation.evaluate(self.data, predictions)[1]['overall']['exact_plate_accuracy'], 0)
        with self.assertRaisesRegex(ValueError, 'lists'):
            evaluation.evaluate(self.data, predictions[:1])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            evaluation.evaluate(self.data, predictions + predictions[:1])
        predictions[0]['expected_text'] = 'WRONG'
        with self.assertRaisesRegex(ValueError, 'labels'):
            evaluation.evaluate(self.data, predictions)

    def test_annotation_cli_quit_and_evaluation_files(self):
        source = self.root / 'full scene.png'
        Image.fromarray(self.frame).save(source)
        destination = self.root / 'dataset'
        with contextlib.redirect_stdout(io.StringIO()), \
             patch('builtins.input', side_effect=['2', '59A', '00123', 'q']), \
             patch.object(annotation, 'select_line', side_effect=[[0, 0, 100, 18], [0, 20, 100, 40]]):
            self.assertEqual(annotation.main(['--source', str(source), '--output', str(destination)]), 0)
        data = annotation.load_dataset(destination)
        with (destination / 'labels.csv').open(encoding='utf-8-sig', newline='') as handle:
            labels = list(csv.DictReader(handle))
        predictions = self.root / 'ocr.csv'
        with predictions.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['image', 'prediction', 'expected_text', 'status', 'device'])
            writer.writeheader()
            writer.writerows(dict(image=r['image'], prediction=r['text'], expected_text=r['text'], status='ok', device='cpu') for r in labels)
        output = self.root / 'evaluation'
        with contextlib.redirect_stdout(io.StringIO()):
            evaluation.main(['--dataset', str(destination / 'dataset.json'), '--predictions', str(predictions), '--output', str(output)])
        report = json.loads((output / 'plate_summary.json').read_text())
        self.assertEqual(report['overall']['exact_plate_accuracy'], 1)
        self.assertEqual(len(data['plates']), 1)
        self.assertTrue((output / 'plate_evaluation.csv').is_file())
        data['plates'][0]['lines'][1]['text'] = '00001'
        (destination / 'dataset.json').write_text(json.dumps(data), encoding='utf-8')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(annotation.main(['--output', str(destination), '--export-only']), 0)
        with (destination / 'labels.csv').open(encoding='utf-8-sig', newline='') as handle:
            self.assertEqual(list(csv.DictReader(handle))[1]['text'], '00001')


if __name__ == '__main__':
    unittest.main()
