"""Demo launcher tests use fake inference and actual media tools, never real NPU calls."""
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import demo_alpr as demo


class DemoFlowTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.run = self.make_model('dxm1_full_encoder_mm_verified', b'verified fixture model')
        evidence = self.repo / 'docs/evidence/dxm1_20261101.json'
        evidence.parent.mkdir(parents=True)
        evidence.write_text(json.dumps({'dxnn_sha256': demo.sha256(self.run / 'model.dxnn')}))

    def make_model(self, name, binary):
        folder = self.repo / 'artifacts' / name
        folder.mkdir(parents=True)
        model = folder / 'model.dxnn'
        model.write_bytes(binary)
        (folder / 'matmul_contract.json').write_text(
            json.dumps({
                'status': 'compiled',
                'dxnn': 'model.dxnn',
                'dxnn_sha256': hashlib.sha256(binary).hexdigest(),
                'placement': {'projection_nodes': ['project'], 'partitions': {'npu_0': 'NPU'}},
            })
        )
        return folder

    def test_default_prefers_verified_hash_over_newer_experiment(self):
        self.make_model('dxm1_full_encoder_mm_zz_new', b'new unverified model')
        self.assertEqual(demo.resolve_run(self.repo), self.run)

    def test_explicit_model_is_respected_and_tampering_rejected(self):
        other = self.make_model('custom_full', b'custom model')
        self.assertEqual(demo.resolve_run(self.repo, str(other)), other)
        (other / 'model.dxnn').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'sai hash'):
            demo.resolve_run(self.repo, str(other))

    def test_ambiguous_verified_copies_need_explicit_selection(self):
        self.make_model('dxm1_full_encoder_mm_duplicate', b'verified fixture model')
        with self.assertRaisesRegex(ValueError, 'nhiều encoder'):
            demo.resolve_run(self.repo)

    def test_paths_with_spaces_quotes_and_file_uri(self):
        photo = self.repo / 'ảnh mới có dấu cách.png'
        Image.new('RGB', (40, 30)).save(photo)
        for value in (str(photo), "'" + str(photo) + "'", str(photo).replace(' ', '\\ '), photo.as_uri()):
            self.assertEqual(demo.parse_source(value), photo)
        with self.assertRaisesRegex(ValueError, 'cục bộ'):
            demo.parse_source('file://remote-host/tmp/photo.png')

    def test_missing_and_unsupported_source_rejected(self):
        with self.assertRaises(ValueError):
            demo.parse_source(str(self.repo / 'missing.png'))
        text = self.repo / 'notes.txt'
        text.write_text('not media')
        with self.assertRaisesRegex(ValueError, 'không phải ảnh/video'):
            demo.parse_source(str(text))
        with self.assertRaisesRegex(ValueError, 'không có ảnh/video'):
            demo.parse_source(str(self.repo))

    def test_file_picker_cancel_does_not_start_inference(self):
        response = subprocess.CompletedProcess([], 1, stdout='', stderr='')
        with patch.object(demo.shutil, 'which', return_value='/usr/bin/zenity'), patch.dict(
            os.environ, {'DISPLAY': ':1'}
        ), patch.object(demo.subprocess, 'run', return_value=response):
            self.assertIsNone(demo.choose_source())

    def test_command_uses_actual_source_and_cpu_decoder_without_shell_interpolation(self):
        source = self.repo / 'new $(bad) image.png'
        command = demo.command_for(self.repo, self.run, source, self.repo / 'out', 120, 1, 'auto')
        self.assertEqual(command[command.index('--source') + 1], str(source))
        self.assertEqual(command[command.index('--device') + 1], 'cpu')
        self.assertEqual(command[command.index('--max-frames') + 1], '120')
        self.assertNotIn('--no-preview', command)

    def test_check_rejects_success_exit_without_actual_m1_device(self):
        for filename in ('alpr_auto.py', 'alpr_auto_dxm1.py', 'dxm1_parseq_backend.py'):
            (self.repo / filename).write_text('# fake entry')
        with patch.object(demo.importlib.util, 'find_spec', return_value=object()), patch.object(
            demo.shutil, 'which', return_value='/fake/tool'
        ), patch.object(
            demo.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout='No devices', stderr='')
        ):
            with self.assertRaisesRegex(ValueError, 'Chưa nhận DX-M1'):
                demo.check_environment(self.repo, self.run)

    def fake_pipeline(self, command, **kwargs):
        output = Path(command[command.index('--output') + 1])
        source = Path(command[command.index('--source') + 1])
        output.mkdir(parents=True)
        if source.suffix == '.mp4':
            subprocess.run(
                [
                    'ffmpeg',
                    '-v',
                    'error',
                    '-f',
                    'lavfi',
                    '-i',
                    'color=c=blue:s=128x96:r=4',
                    '-frames:v',
                    '4',
                    '-c:v',
                    'mpeg4',
                    str(output / '001_preview.mp4'),
                ],
                check=True,
            )
            count = 4
        else:
            Image.open(source).save(output / '001_new_image.jpg')
            count = 1
        report = {
            'status': 'complete',
            'ocr_encoder_backend': 'DX-M1',
            'ocr_decoder_device': 'cpu',
            'processed_frames': count,
            'detections': count,
            'events': 1,
            'ocr_errors': 0,
            'sources': [{'processed_frames': count}],
            'deepx_encoder': {'dxnn_sha256': demo.sha256(self.run / 'model.dxnn')},
        }
        (output / 'summary.json').write_text(json.dumps(report))
        (output / 'detections.csv').write_text('status,text\nok,59D208525\n')
        (output / 'events.csv').write_text('text\n59D208525\n')
        return subprocess.CompletedProcess(command, 0)

    def test_fresh_image_flow_creates_gallery_and_never_claims_labeled_accuracy(self):
        photo = self.repo / 'new image.png'
        Image.new('RGB', (32, 32), 'green').save(photo)
        with patch.object(demo, 'check_environment', return_value='FIXTURE DEVICE'), patch.object(
            demo.subprocess, 'run', side_effect=self.fake_pipeline
        ), contextlib.redirect_stdout(io.StringIO()):
            code = demo.main(['--repo', str(self.repo), '--source', str(photo), '--no-open'])
        self.assertEqual(code, 0)
        output = next((self.repo / 'results').iterdir())
        self.assertTrue((output / '001_new_image.jpg').is_file())
        gallery = (output / 'index.html').read_text()
        self.assertIn('chưa có accuracy', gallery)
        self.assertIn('59D208525', gallery)
        metadata = demo.read_json(output / 'demo_run.json')
        self.assertEqual(metadata['source'], str(photo))
        self.assertFalse((photo.parent / 'labels.csv').exists())

    def test_pipeline_failure_is_not_presented_as_success(self):
        photo = self.repo / 'new.png'
        Image.new('RGB', (32, 32)).save(photo)
        with patch.object(demo, 'check_environment', return_value='FIXTURE DEVICE'), patch.object(
            demo.subprocess, 'run', return_value=subprocess.CompletedProcess([], 2)
        ), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as errors:
            code = demo.main(['--repo', str(self.repo), '--source', str(photo), '--no-open'])
        self.assertEqual(code, 1)
        self.assertIn('Pipeline dừng', errors.getvalue())
        self.assertFalse((self.repo / 'results').exists())

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'Actual FFmpeg media check')
    def test_new_video_flow_produces_verified_h264(self):
        source = self.repo / 'new video.mp4'
        source.write_bytes(b'fixture source; fake inference writes a real preview')
        original_run = subprocess.run

        def boundary(command, **kwargs):
            if 'alpr_auto_dxm1.py' in str(command[1]):
                # Only inference is faked. FFmpeg/ffprobe remain actual processes.
                with patch.object(demo.subprocess, 'run', original_run):
                    return self.fake_pipeline(command, **kwargs)
            return original_run(command, **kwargs)

        with patch.object(demo, 'check_environment', return_value='FIXTURE DEVICE'), patch.object(
            demo.subprocess, 'run', side_effect=boundary
        ), contextlib.redirect_stdout(io.StringIO()):
            code = demo.main(['--repo', str(self.repo), '--source', str(source), '--no-open', '--max-frames', '4'])
        self.assertEqual(code, 0)
        output = next((self.repo / 'results').iterdir())
        self.assertTrue((output / '001_preview_H264.mp4').is_file())
        self.assertEqual(demo.read_json(output / 'demo_run.json')['video_frames'], {'001_preview_H264.mp4': 4})

    def test_zero_detection_and_untrusted_text_are_shown_honestly_and_escaped(self):
        output = self.repo / 'gallery'
        output.mkdir()
        (output / 'detections.csv').write_text('status,text\nok,<script>alert(1)</script>\n')
        report = {'processed_frames': 1, 'detections': 0, 'events': 0, 'ocr_errors': 1}
        demo.write_gallery(output, report, [], [])
        page = (output / 'index.html').read_text()
        self.assertIn('Không phát hiện biển', page)
        self.assertIn('Có lỗi OCR', page)
        self.assertIn('&lt;script&gt;', page)
        self.assertNotIn('<script>', page)


if __name__ == '__main__':
    unittest.main()
