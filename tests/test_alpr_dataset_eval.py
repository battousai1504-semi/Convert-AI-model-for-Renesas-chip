"""Evaluation regressions use real image files and synthetic predictions, no model/NPU."""
import contextlib
import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate_alpr_dataset as evaluation
import prepare_vn_eval as preparation


def annotation(identity='p1', text='30A12345', box=(0, 0, 10, 10)):
    return {'id': identity, 'text': text, 'box_xyxy': list(box), 'layout': 'one'}


def image_record(plates):
    return {'id': 'im1', 'split': 'development', 'group_id': 'capture1', 'annotations': plates}


def prediction(text='30A12345', box=(0, 0, 10, 10), status='ok'):
    return {'box': box, 'text': text, 'status': status, 'error': ''}


class MetricTests(unittest.TestCase):

    def score(self, plates, predictions):
        image = image_record(plates)
        rows = evaluation.score_images([image], {'im1': predictions}, 0.5)
        return rows, evaluation.metrics(rows, [image])

    def test_maximum_cardinality_avoids_greedy_false_miss(self):
        matching = evaluation.match_boxes([(0, 0, 10, 10), (5, 0, 15, 10)], [(2, 0, 12, 10), (0, 0, 5, 10)], 0.5)
        self.assertEqual(matching, {0: 1, 1: 0})

    def test_matching_never_uses_plate_text(self):
        rows, score = self.score(
            [annotation('a', 'ABC123', (0, 0, 10, 10)), annotation('b', 'DEF456', (20, 0, 30, 10))],
            [prediction('DEF456'), prediction('ABC123', (20, 0, 30, 10))],
        )
        self.assertEqual(score['detection_tp'], 2)
        self.assertEqual(score['exact_plates'], 0)
        self.assertEqual([row['category'] for row in rows], ['ocr_wrong', 'ocr_wrong'])

    def test_duplicate_box_is_false_positive_not_extra_true_positive(self):
        rows, score = self.score([annotation()], [prediction(), prediction()])
        self.assertEqual((score['detection_tp'], score['detection_fp'], score['detection_fn']), (1, 1, 0))
        self.assertEqual(score['end_to_end_exact_precision'], 0.5)
        self.assertEqual(score['end_to_end_exact_recall'], 1)
        self.assertEqual(len(rows), 2)

    def test_misses_and_failed_ocr_remain_in_denominators(self):
        rows, score = self.score(
            [annotation('a', 'ABC'), annotation('b', 'DEF', (20, 0, 30, 10)), annotation('c', 'GHI', (40, 0, 50, 10))],
            [prediction('ABC'), prediction('DEF', (20, 0, 30, 10), 'error')],
        )
        self.assertEqual(score['end_to_end_exact_recall'], 1 / 3)
        self.assertEqual(score['ocr_exact_on_matched_boxes'], 0.5)
        self.assertEqual(score['ocr_cer_on_matched_boxes'], 0.5)
        self.assertEqual(score['gt_cer_including_missed_plates'], 2 / 3)
        self.assertEqual(rows[1]['prediction'], '')
        self.assertEqual(rows[1]['raw_prediction'], 'DEF')

    def test_no_detections_and_negative_images_do_not_invent_perfect_accuracy(self):
        _, positive = self.score([annotation()], [])
        self.assertEqual(positive['end_to_end_exact_recall'], 0)
        self.assertIsNone(positive['ocr_exact_on_matched_boxes'])
        _, negative = self.score([], [])
        self.assertIsNone(negative['end_to_end_exact_recall'])
        self.assertIsNone(negative['detection_precision'])
        _, false_alarm = self.score([], [prediction()])
        self.assertEqual(false_alarm['detection_fp'], 1)
        self.assertEqual(false_alarm['detection_precision'], 0)

    def test_normalization_does_not_fix_lookalikes_or_clip_cer(self):
        self.assertEqual(evaluation.normalize('30a-123.45'), '30A12345')
        _, score = self.score([annotation(text='O1')], [prediction('0I')])
        self.assertEqual(score['exact_plates'], 0)
        _, long = self.score([annotation(text='AB')], [prediction('ABCDEFG')])
        self.assertGreater(long['ocr_cer_on_matched_boxes'], 1)

    def test_threshold_and_box_validation(self):
        for threshold in (0, -1, float('nan'), float('inf'), 1.1):
            with self.assertRaises(ValueError):
                evaluation.match_boxes([], [], threshold)
        for box in [(0, 0, 0, 1), (0, 0, float('nan'), 1), (-1, 0, 10, 10), (0, 0, 50, 10)]:
            with self.assertRaises(ValueError):
                evaluation.valid_box(box, 32, 32)


class DatasetIntegrationTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / 'dataset'
        (self.data / 'images').mkdir(parents=True)
        self.images = []
        for index in range(2):
            path = self.data / 'images' / f'image{index}.png'
            Image.new('RGB', (32, 32), color=(index * 100, 20, 30)).save(path)
            self.images.append(
                dict(
                    image_record([annotation(text='30A12345')]),
                    id=f'im{index}',
                    file='images/' + path.name,
                    width=32,
                    height=32,
                    sha256=evaluation.sha256(path),
                )
            )
        self.dataset = {
            'schema_version': 1,
            'name': 'fixture',
            'scope': 'Synthetic predictions only',
            'label_policy': 'test fixtures',
            'images': self.images,
        }
        self.manifest = self.data / 'dataset.json'
        self.write_json(self.manifest, self.dataset)

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    def make_run(self, backend, texts=('30A12345', '30A12345')):
        folder = self.root / backend
        folder.mkdir()
        frames, detections = [], []
        for image, text in zip(self.images, texts):
            source = str((self.data / image['file']).resolve())
            frames.append({'source': source, 'frame_index': 0, 'status': 'ok', 'detections': 1, 'compute_ms': 10})
            detections.append(
                {'source': source, 'frame_index': 0, 'status': 'ok', 'text': text, 'x1': 0, 'y1': 0, 'x2': 10, 'y2': 10}
            )
        evaluation.write_csv(folder / 'frames.csv', frames)
        evaluation.write_csv(folder / 'detections.csv', detections)
        self.write_json(
            folder / 'summary.json',
            {
                'status': 'complete',
                'processed_frames': 2,
                'detections': 2,
                'ocr_device': 'cpu',
                'ocr_encoder_backend': 'DX-M1' if backend == 'dxm1' else 'PyTorch',
                'ocr_decoder_device': 'cpu',
                'deepx_encoder': {'dxnn_sha256': 'FIXTURE_NOT_A_REAL_MODEL'},
                'detector_sha256': 'FIXTURE_DETECTOR',
                'checkpoint': 'fixture',
                'arguments': {'confidence': 0.35, 'layout': 'auto', 'no_rectify': False},
            },
        )
        return folder

    def test_cli_writes_wrong_plate_errors_review_and_honest_accuracy(self):
        cpu = self.make_run('cpu')
        dx = self.make_run('dxm1', ('30A12345', '30A12346'))
        destination = self.root / 'report'
        with contextlib.redirect_stdout(io.StringIO()):
            code = evaluation.main([
                '--dataset',
                str(self.manifest),
                '--cpu-run',
                str(cpu),
                '--dxm1-run',
                str(dx),
                '--output',
                str(destination),
            ])
        self.assertEqual(code, 0)  # Scoring a wrong plate is successful evaluation, not an execution error.
        report = evaluation.read_json(destination / 'summary.json')
        self.assertEqual(report['backends']['cpu']['overall']['end_to_end_exact_recall'], 1)
        self.assertEqual(report['backends']['dxm1']['overall']['end_to_end_exact_recall'], 0.5)
        self.assertEqual(report['conversion_comparison']['same_nonempty_successful_text'], 1)
        self.assertEqual(len(evaluation.read_csv(destination / 'dxm1_errors.csv')), 1)
        self.assertTrue((destination / 'review.html').is_file())
        self.assertEqual(len(list((destination / 'review').glob('*.jpg'))), 4)

    def test_both_backends_missing_every_plate_is_zero_accuracy_and_zero_agreement(self):
        folders = [self.make_run(name) for name in ('cpu', 'dxm1')]
        for folder in folders:
            frames = evaluation.read_csv(folder / 'frames.csv')
            for frame in frames:
                frame.update(status='no_plate', detections=0)
            evaluation.write_csv(folder / 'frames.csv', frames)
            (folder / 'detections.csv').write_text('source,frame_index,status,text,x1,y1,x2,y2\n')
            report = evaluation.read_json(folder / 'summary.json')
            report['detections'] = 0
            self.write_json(folder / 'summary.json', report)
        destination = self.root / 'report'
        with contextlib.redirect_stdout(io.StringIO()):
            evaluation.main([
                '--dataset',
                str(self.manifest),
                '--cpu-run',
                str(folders[0]),
                '--dxm1-run',
                str(folders[1]),
                '--output',
                str(destination),
            ])
        report = evaluation.read_json(destination / 'summary.json')
        for backend in ('cpu', 'dxm1'):
            self.assertEqual(report['backends'][backend]['overall']['end_to_end_exact_recall'], 0)
        self.assertEqual(report['conversion_comparison']['paired_geometric_matches'], 0)
        self.assertEqual(report['conversion_comparison']['same_nonempty_successful_text'], 0)

    def test_prepare_downloads_verified_dataset_and_review_sheet_idempotently(self):
        recipe = copy.deepcopy(self.dataset)
        contents = {}
        for index, image in enumerate(recipe['images']):
            data = (self.data / image['file']).read_bytes()
            image['file'] = Path(image['file']).name
            image['url'] = f'https://example.invalid/image{index}.png'
            label_url = f'https://example.invalid/label{index}.txt'
            label = b'0 0.15625 0.15625 0.3125 0.3125'
            image['annotation_provenance'] = {
                'box_label_url': label_url,
                'box_label_sha256': hashlib.sha256(label).hexdigest(),
            }
            contents[image['url']], contents[label_url] = data, label
        recipe_path, target = self.root / 'recipe.json', self.root / 'downloaded'
        self.write_json(recipe_path, recipe)
        calibration = self.root / 'calibration.json'
        self.write_json(calibration, {'calibration': [{'sha256': '0' * 64}]})
        args = ['--recipe', str(recipe_path), '--output', str(target), '--calibration-manifest', str(calibration)]
        with patch.object(
            preparation, 'urlopen', side_effect=lambda url, timeout: io.BytesIO(contents[url])
        ), contextlib.redirect_stdout(io.StringIO()):
            preparation.main(args)
        generated = evaluation.load_dataset(target / 'dataset.json')
        self.assertEqual(len(generated['images']), 2)
        self.assertTrue((target / 'label_review.jpg').is_file())
        before = (target / 'dataset.json').read_bytes()
        with patch.object(
            preparation, 'urlopen', side_effect=AssertionError('verified cache should be reused')
        ), contextlib.redirect_stdout(io.StringIO()):
            preparation.main(args)
        self.assertEqual((target / 'dataset.json').read_bytes(), before)
        self.write_json(calibration, {'calibration': [{'sha256': recipe['images'][0]['sha256']}]})
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            preparation.main(args)
        self.assertEqual((target / 'dataset.json').read_bytes(), before)

    def test_missing_frame_is_rejected_instead_of_dropping_sample(self):
        folder = self.make_run('cpu')
        rows = evaluation.read_csv(folder / 'frames.csv')
        evaluation.write_csv(folder / 'frames.csv', rows[:1])
        with self.assertRaisesRegex(ValueError, 'Missing frame'):
            evaluation.load_run(folder, self.dataset, self.manifest, 'cpu')

    def test_foreign_prediction_source_and_incomplete_run_are_rejected(self):
        folder = self.make_run('cpu')
        rows = evaluation.read_csv(folder / 'detections.csv')
        rows[0]['source'] = str(self.root / 'foreign.png')
        evaluation.write_csv(folder / 'detections.csv', rows)
        with self.assertRaisesRegex(ValueError, 'Unexpected image'):
            evaluation.load_run(folder, self.dataset, self.manifest, 'cpu')
        report = evaluation.read_json(folder / 'summary.json')
        report['status'] = 'failed'
        self.write_json(folder / 'summary.json', report)
        with self.assertRaisesRegex(ValueError, 'did not finish'):
            evaluation.load_run(folder, self.dataset, self.manifest, 'cpu')

    def test_hash_modification_and_group_leak_are_rejected(self):
        (self.data / self.images[0]['file']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            evaluation.load_dataset(self.manifest)
        # Restore the image and then check group splitting independently.
        Image.new('RGB', (32, 32), color=(0, 20, 30)).save(self.data / self.images[0]['file'])
        self.dataset['images'][1]['split'] = 'test'
        self.write_json(self.manifest, self.dataset)
        with self.assertRaisesRegex(ValueError, 'leaks across'):
            evaluation.load_dataset(self.manifest)

    def test_duplicate_images_empty_dataset_and_dimensions_rejected(self):
        value = copy.deepcopy(self.dataset)
        value['images'] = []
        self.write_json(self.manifest, value)
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            evaluation.load_dataset(self.manifest)
        value = copy.deepcopy(self.dataset)
        value['images'][1]['file'] = value['images'][0]['file']
        self.write_json(self.manifest, value)
        with self.assertRaisesRegex(ValueError, 'Duplicate image file'):
            evaluation.load_dataset(self.manifest)
        value = copy.deepcopy(self.dataset)
        value['images'][0]['width'] = 100
        self.write_json(self.manifest, value)
        with self.assertRaisesRegex(ValueError, 'dimensions differ'):
            evaluation.load_dataset(self.manifest)

    def test_download_checksum_failure_preserves_existing_file(self):
        path = self.root / 'download.jpg'
        path.write_bytes(b'user data')
        with patch.object(preparation, 'urlopen', side_effect=AssertionError('must not download existing file')):
            with self.assertRaisesRegex(ValueError, 'Checksum differs'):
                preparation.download_verified('https://example.invalid/image', '0' * 64, path)
        self.assertEqual(path.read_bytes(), b'user data')

    def test_verified_download_with_retry_and_no_overwrite(self):
        data = b'new image bytes'

        class Response(io.BytesIO):
            pass

        path = self.root / 'download.jpg'
        with patch.object(preparation, 'urlopen', side_effect=[OSError('transient'), Response(data)]):
            preparation.download_verified('https://example.invalid/image', hashlib.sha256(data).hexdigest(), path)
        self.assertEqual(path.read_bytes(), data)
        with patch.object(preparation, 'urlopen', side_effect=AssertionError('should reuse verified file')):
            preparation.download_verified('https://example.invalid/image', hashlib.sha256(data).hexdigest(), path)


if __name__ == '__main__':
    unittest.main()
