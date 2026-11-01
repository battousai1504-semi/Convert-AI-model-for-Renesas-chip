import contextlib
import csv
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'ocr_batch.py'
spec = importlib.util.spec_from_file_location('ocr_batch', SCRIPT)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

export_spec = importlib.util.spec_from_file_location('export_parseq_encoder', SCRIPT.parent / 'export_parseq_encoder.py')
export_app = importlib.util.module_from_spec(export_spec)
export_spec.loader.exec_module(export_app)


class BaselineTests(unittest.TestCase):
    def test_encoder_comparison_rejects_shape_and_nonfinite_mismatches(self):
        self.assertFalse(export_app.compare_arrays(np.zeros(2), np.zeros(3), rtol=1e-3, atol=1e-4)['close'])
        self.assertFalse(export_app.compare_arrays(np.array([np.nan]), np.array([np.nan]), rtol=1e-3, atol=1e-4)['close'])
        self.assertTrue(export_app.compare_arrays(np.array([1.0]), np.array([1.00001]), rtol=1e-3, atol=1e-4)['close'])

    def test_export_report_and_image_discovery(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            Image.new('RGB', (80, 24), 'white').save(root / 'plate.PNG')
            (root / 'notes.txt').write_text('not an image')
            discovered_root, images = export_app.collect_images(root)
            self.assertEqual(discovered_root, root)
            self.assertEqual([p.name for p in images], ['plate.PNG'])
            row = dict.fromkeys(export_app.CSV_FIELDS, '')
            row.update(image='plate.PNG', status='pass')
            export_app.save_validation(root, [row], {'all_validations_passed': True})
            self.assertTrue(json.loads((root / 'export_report.json').read_text())['all_validations_passed'])
            with (root / 'encoder_validation.csv').open(encoding='utf-8-sig', newline='') as handle:
                self.assertEqual(next(csv.DictReader(handle))['image'], 'plate.PNG')

    def test_metrics_preserve_errors_and_do_not_invent_accuracy(self):
        rows = [app.make_row('a.png', 'cpu', '59A-123.45', '59a12345'),
                app.make_row('b.png', 'cpu', '51F12345', '51F1234S'),
                app.make_row('c.png', 'cpu', '30A12345', error='bad image')]
        summary = app.summarize(rows)
        self.assertEqual(summary['exact_plate_accuracy'], 1 / 3)
        self.assertEqual(summary['character_error_rate'], 9 / 24)
        self.assertEqual(rows[1]['prediction'], '51F1234S')
        self.assertEqual(app.normalize('O0 I1 B8'), 'O0I1B8')
        self.assertIsNone(app.summarize([app.make_row('x', 'cpu', prediction='ABC')])['exact_plate_accuracy'])

    def test_labels_utf8_bom_duplicates_and_filename_normalization(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'labels.csv'
            path.write_text('image,text\nsub\\a.png,00123\n', encoding='utf-8-sig')
            self.assertEqual(app.load_labels(path), {'sub/a.png': '00123'})
            path.write_text('image,text\na.png,00123\na.png,00001\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                app.load_labels(path)

    def test_cli_csv_and_comparison_with_stubbed_inference(self):
        # Exercise real image decoding, CLI, both device paths, bad-file reporting,
        # CSV output and summary. Only model/torch execution is stubbed.
        class Tensor:
            def unsqueeze(self, axis): return self
            def to(self, device): return self
        class Logits:
            def softmax(self, axis): return self
        class Model:
            hparams = types.SimpleNamespace(img_size=(32, 128))
            tokenizer = types.SimpleNamespace(decode=lambda _: (['59A12345'], [None]))
            def eval(self): return self
            def to(self, device): return self
            def __call__(self, tensor): return Logits()
        torch = types.ModuleType('torch')
        torch.__version__ = 'TEST_STUB'
        torch.version = types.SimpleNamespace(cuda='TEST_STUB')
        torch.cuda = types.SimpleNamespace(is_available=lambda: True,
                    synchronize=lambda: None, get_device_name=lambda _: 'TEST_STUB',
                    empty_cache=lambda: None)
        torch.get_num_threads = lambda: 1
        torch.inference_mode = contextlib.nullcontext
        data_module = types.ModuleType('strhub.data.module')
        data_module.SceneTextDataModule = types.SimpleNamespace(
            get_transform=lambda size: lambda image: Tensor())
        model_module = types.ModuleType('strhub.models.utils')
        model_module.load_from_checkpoint = lambda checkpoint: Model()
        modules = {'torch': torch, 'strhub.data.module': data_module,
                   'strhub.models.utils': model_module}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'images'
            root.mkdir()
            Image.new('RGB', (80, 24), 'white').save(root / 'one.PNG')
            (root / 'two.png').write_bytes(b'not an image')
            labels = Path(td) / 'labels.csv'
            labels.write_text('image,text\none.PNG,59A12345\ntwo.png,30A12345\n')
            out = Path(td) / 'output'
            argv = [str(SCRIPT), '--input', str(root), '--output', str(out),
                    '--device', 'both', '--labels', str(labels)]
            with patch.dict(sys.modules, modules), patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                status = app.main()
            self.assertEqual(status, 1)
            for device in ('cpu', 'cuda'):
                with (out / f'ocr_{device}.csv').open(encoding='utf-8-sig', newline='') as handle:
                    records = list(csv.DictReader(handle))
                self.assertEqual([r['status'] for r in records], ['ok', 'error'])
                self.assertEqual(records[0]['prediction'], '59A12345')
            report = json.loads((out / 'summary.json').read_text())
            self.assertEqual(report['results']['cpu']['exact_plate_accuracy'], 0.5)
            self.assertEqual(report['cpu_gpu_comparison']['paired_successful_images'], 1)
            self.assertEqual(report['cpu_gpu_comparison']['raw_prediction_disagreements'], [])


if __name__ == '__main__':
    unittest.main()
