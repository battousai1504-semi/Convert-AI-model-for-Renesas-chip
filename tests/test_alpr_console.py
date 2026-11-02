"""Queue/menu contracts use recorded fake inference; no driver or NPU calls."""
import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import alpr_console as console


class ConsoleTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / 'repo'
        self.repo.mkdir()
        self.photo = Path(self.tmp.name) / 'ảnh mới có dấu cách.jpg'
        self.photo.write_bytes(b'photo fixture: no model used')
        self.video = Path(self.tmp.name) / 'new video.mp4'
        self.video.write_bytes(b'video fixture: no model used')

    def add(self, *sources):
        with contextlib.redirect_stdout(io.StringIO()):
            return console.add_inputs(self.repo, [str(source) for source in sources])

    def test_add_copies_originals_and_deduplicates_content(self):
        self.add(self.photo, self.video)
        self.add(self.photo)
        state = console.read_state(self.repo)
        self.assertEqual(len(state['items']), 2)
        self.assertEqual(console.media_path(self.repo, state['items'][0]).read_bytes(), self.photo.read_bytes())
        self.assertEqual(self.photo.read_bytes(), b'photo fixture: no model used')
        self.assertEqual(state['items'][0]['name'], self.photo.name)
        self.assertEqual(console.read_state(self.repo), state)

    def test_cancel_add_preserves_queue(self):
        self.add(self.photo)
        before = console.read_state(self.repo)
        with patch.object(console.demo, 'choose_source', return_value=None), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(console.add_inputs(self.repo, []), 0)
        self.assertEqual(console.read_state(self.repo), before)

    def test_directory_batch_and_clear_never_delete_originals_or_previous_output(self):
        folder = Path(self.tmp.name) / 'incoming'
        folder.mkdir()
        (folder / 'a.jpg').write_bytes(b'image-a')
        (folder / 'b.mp4').write_bytes(b'video-b')
        (folder / 'readme.txt').write_text('not input')
        self.add(folder)
        state = console.read_state(self.repo)
        self.assertEqual(len(state['items']), 2)
        state['latest_output'] = 'results/previous'
        console.save_state(self.repo, state)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(console.main(['--repo', str(self.repo), 'clear']), 0)
        after = console.read_state(self.repo)
        self.assertEqual(after['items'], [])
        self.assertEqual(after['latest_output'], 'results/previous')
        self.assertTrue((folder / 'a.jpg').exists())
        self.assertTrue(console.media_path(self.repo, state['items'][0]).exists())

    def fake_demo(self, args):
        self.scanned_names = sorted(p.name for p in Path(args[args.index('--source') + 1]).iterdir())
        self.scan_args = args
        output = Path(args[args.index('--output') + 1])
        output.mkdir(parents=True)
        (output / 'index.html').write_text('fixture result')
        (output / 'summary.json').write_text(json.dumps({'status': 'complete', 'ocr_encoder_backend': 'DX-M1'}))
        return 0

    def test_scan_uses_only_selected_inputs_and_remembers_exact_completed_output(self):
        self.add(self.photo, self.video)
        with contextlib.redirect_stdout(io.StringIO()):
            console.main(['--repo', str(self.repo), 'clear'])
        self.add(self.photo)  # old video remains cached, but must never leak into this scan.
        with patch.object(console.demo, 'main', side_effect=self.fake_demo), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(console.scan(self.repo, no_open=True), 0)
        self.assertEqual(len(self.scanned_names), 1)
        self.assertTrue(self.scanned_names[0].endswith('.jpg'))
        self.assertEqual(self.scan_args[self.scan_args.index('--max-frames') + 1], '120')
        self.assertEqual(self.scan_args[self.scan_args.index('--every-n-frames') + 1], '1')
        self.assertTrue((console.latest_output(self.repo) / 'index.html').is_file())

    def test_all_frames_option_and_auto_open(self):
        self.add(self.video)
        with patch.object(console.demo, 'main', side_effect=self.fake_demo), patch.object(
            console, 'open_output', return_value=0
        ) as opened, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(console.main(['--repo', str(self.repo), 'scan', '--all']), 0)
        self.assertEqual(self.scan_args[self.scan_args.index('--max-frames') + 1], '0')
        opened.assert_called_once_with(self.repo)

    def test_failed_scan_keeps_previous_success_and_active_queue(self):
        self.add(self.photo)
        state = console.read_state(self.repo)
        state['latest_output'] = 'results/previous'
        console.save_state(self.repo, state)
        with patch.object(console.demo, 'main', return_value=1), contextlib.redirect_stdout(
            io.StringIO()
        ), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(console.scan(self.repo, no_open=True), 1)
        self.assertEqual(console.read_state(self.repo), state)

    def test_modified_cached_input_is_rejected_before_inference(self):
        self.add(self.photo)
        state = console.read_state(self.repo)
        console.media_path(self.repo, state['items'][0]).write_bytes(b'changed')
        with patch.object(console.demo, 'main', side_effect=AssertionError('must not infer modified input')):
            with self.assertRaisesRegex(ValueError, 'thiếu/đã đổi'):
                console.scan(self.repo)
        self.assertFalse((self.repo / 'inputs/demo_batches').exists())

    def test_bad_state_paths_do_not_escape_private_inbox_or_results(self):
        console.save_state(self.repo, {'version': 1, 'items': [], 'latest_output': '../../outside'})
        with self.assertRaisesRegex(ValueError, 'Output state'):
            console.latest_output(self.repo)
        console.save_state(
            self.repo, {'version': 1, 'items': [{'file': '../../outside.jpg', 'sha256': 'abc', 'name': 'x'}]}
        )
        with self.assertRaisesRegex(ValueError, 'inbox'):
            console.read_state(self.repo)

    def test_open_uses_saved_output_and_optional_folder_without_requiring_card(self):
        output = self.repo / 'results/completed'
        output.mkdir(parents=True)
        (output / 'index.html').write_text('fixture result')
        (output / '001_preview_H264.mp4').write_bytes(b'fixture media')
        console.save_state(self.repo, {'version': 1, 'items': [], 'latest_output': 'results/completed'})
        with patch.object(console.shutil, 'which', return_value='/fake/tool'), patch.object(
            console.subprocess, 'Popen'
        ) as opened, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(console.open_output(self.repo), 0)
            self.assertEqual(opened.call_args_list[0].args[0], ['xdg-open', str(output / 'index.html')])
            self.assertEqual(opened.call_args_list[1].args[0], ['vlc', '--loop', str(output / '001_preview_H264.mp4')])
            opened.reset_mock()
            self.assertEqual(console.open_output(self.repo, folder=True), 0)
            self.assertEqual(opened.call_count, 1)
            self.assertEqual(opened.call_args.args[0], ['xdg-open', str(output)])

    def test_start_requests_only_driver_loading_and_then_read_only_check(self):
        with patch.object(console.subprocess, 'run') as run, patch.object(
            console.demo, 'main', return_value=0
        ) as checked:
            self.assertEqual(console.start(self.repo), 0)
        self.assertEqual(
            [call.args[0] for call in run.call_args_list],
            [['sudo', 'modprobe', 'dx_dma'], ['sudo', 'modprobe', 'dxrt_driver']],
        )
        self.assertIn('--check', checked.call_args.args[0])

    def test_empty_scan_does_not_start_inference(self):
        with patch.object(console.demo, 'main', side_effect=AssertionError('empty queue must not infer')):
            with self.assertRaisesRegex(ValueError, 'Chưa chọn input'):
                console.scan(self.repo)

    def test_explicit_output_cannot_overwrite_existing_directory(self):
        output = self.repo / 'results/existing'
        output.mkdir(parents=True)
        marker = output / 'keep.txt'
        marker.write_text('user file')
        with patch.object(console.demo, 'resolve_run', return_value=self.repo / 'fixture_model'), patch.object(
            console.demo, 'check_environment', return_value='FIXTURE DEVICE'
        ), patch.object(
            console.subprocess, 'run', side_effect=AssertionError('must not start pipeline')
        ), contextlib.redirect_stdout(
            io.StringIO()
        ), contextlib.redirect_stderr(
            io.StringIO()
        ):
            code = console.demo.main(
                ['--repo', str(self.repo), '--source', str(self.photo), '--output', str(output), '--no-open']
            )
        self.assertEqual(code, 1)
        self.assertEqual(marker.read_text(), 'user file')

    def test_menu_lists_current_queue_and_exits_without_inference(self):
        self.add(self.photo)
        with patch.object(console.sys.stdin, 'isatty', return_value=True), patch(
            'builtins.input', side_effect=['5', '0']
        ), patch.object(
            console.demo, 'main', side_effect=AssertionError('status menu must not infer')
        ), contextlib.redirect_stdout(
            io.StringIO()
        ) as text:
            self.assertEqual(console.menu(self.repo), 0)
        self.assertIn(self.photo.name, text.getvalue())

    def test_wrapper_runs_help_with_venv_python_from_another_working_directory(self):
        source = Path(__file__).resolve().parents[1]
        for name in ('alpr', 'alpr_console.py', 'demo_alpr.py'):
            shutil.copyfile(source / name, self.repo / name)
        executable = self.repo / '.venv-linux/bin/python'
        executable.parent.mkdir(parents=True)
        executable.symlink_to(sys.executable)
        result = subprocess.run(
            ['bash', str(self.repo / 'alpr'), 'help'], cwd=self.tmp.name, check=True, text=True, capture_output=True
        )
        self.assertIn('scan', result.stdout)
        self.assertIn('open', result.stdout)
        self.assertFalse((self.repo / 'inputs').exists())


if __name__ == '__main__':
    unittest.main()
