#!/usr/bin/env python3
"""Choose a new local image/video and demonstrate the existing DX-M1 ALPR CLI."""
import argparse
import csv
import hashlib
import html
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_run(path):
    path = Path(path).resolve()
    manifest = read_json(path / 'matmul_contract.json')
    require(manifest.get('status') == 'compiled', 'Encoder run chưa compile xong.')
    model = (path / manifest['dxnn']).resolve()
    model.relative_to(path)
    require(model.is_file() and sha256(model) == manifest['dxnn_sha256'], 'DXNN thiếu hoặc sai hash.')
    require(
        manifest.get('placement', {}).get('projection_nodes')
        and 'NPU' in manifest.get('placement', {}).get('partitions', {}).values(),
        'Manifest chưa xác nhận projection trên NPU.',
    )
    return path


def resolve_run(repo, explicit=None):
    if explicit:
        path = Path(explicit).expanduser()
        return validate_run(path if path.is_absolute() else repo / path)
    # Prefer the model hash from the real hardware milestone; never silently
    # select a newer experimental run merely because its timestamp is later.
    evidence = repo / 'docs/evidence/dxm1_20261101.json'
    expected = read_json(evidence).get('dxnn_sha256') if evidence.is_file() else None
    require(expected, 'Không có hash encoder đã kiểm chứng. Chọn rõ full encoder bằng --dx-run.')
    candidates = []
    for manifest_path in sorted((repo / 'artifacts').glob('dxm1_full_encoder_mm_*/matmul_contract.json')):
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError):
            print('Bỏ qua manifest thử nghiệm chưa đọc được:', manifest_path)
            continue
        if manifest.get('status') == 'compiled' and manifest.get('dxnn_sha256') == expected:
            candidates.append(manifest_path.parent)
    require(candidates, 'Không tìm thấy full encoder đã kiểm chứng. Truyền --dx-run theo docs/DEMO.md.')
    require(len(candidates) == 1, 'Có nhiều encoder phù hợp. Chọn rõ một thư mục bằng --dx-run.')
    return validate_run(candidates[0])


def parse_source(value):
    value = value.strip()
    if value.startswith('file://'):
        parsed = urlparse(value)
        require(parsed.netloc in ('', 'localhost'), 'Chỉ nhận file cục bộ, không nhận URL mạng.')
        value = unquote(parsed.path)
    elif not Path(value).expanduser().exists():
        parts = shlex.split(value)
        require(len(parts) == 1, 'Chọn một file hoặc một thư mục; đặt đường dẫn có dấu cách trong dấu nháy.')
        value = parts[0]
    path = Path(value).expanduser().resolve()
    require(path.exists(), f'Không tìm thấy đầu vào: {path}')
    if path.is_file():
        require(path.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS, 'File không phải ảnh/video được hỗ trợ.')
    else:
        require(path.is_dir(), 'Đầu vào phải là file hoặc thư mục.')
        require(
            any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS for p in path.rglob('*')),
            'Thư mục không có ảnh/video được hỗ trợ.',
        )
    return path


def choose_source():
    if shutil.which('zenity') and (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        result = subprocess.run(
            [
                'zenity',
                '--file-selection',
                '--title=Chọn ảnh hoặc video mới để demo DX-M1',
                '--file-filter=Ảnh/video | *.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.mp4 *.avi *.mov *.mkv *.webm *.m4v',
                '--file-filter=Tất cả | *',
            ],
            text=True,
            capture_output=True,
        )
        if result.returncode == 0:
            return parse_source(result.stdout)
        if result.returncode == 1:
            return None
        print('Không mở được hộp chọn file. Dán đường dẫn trong terminal.')
    require(sys.stdin.isatty(), 'Dùng --source với đường dẫn file; terminal này không nhận nhập tương tác.')
    value = input('Kéo/thả ảnh hoặc video vào terminal (hoặc dán đường dẫn), rồi Enter: ').strip()
    return parse_source(value) if value else None


def check_environment(repo, run, verbose=False):
    require(
        sys.platform.startswith('linux'),
        'Flow DX-M1 này dùng Ubuntu/Linux. Xem docs/SETUP_DUAL_BOOT.md cho baseline Windows.',
    )
    for name in ('alpr_auto.py', 'alpr_auto_dxm1.py', 'dxm1_parseq_backend.py'):
        require((repo / name).is_file(), f'Thiếu {name}; chạy từ repo đã cập nhật.')
    missing = [
        name
        for name in ('torch', 'cv2', 'PIL', 'onnx', 'onnxruntime', 'strhub', 'dx_engine')
        if importlib.util.find_spec(name) is None
    ]
    require(not missing, 'Thiếu thư viện: ' + ', '.join(missing) + '. Dùng .venv-linux/bin/python.')
    require(shutil.which('dxrt-cli'), 'Thiếu dxrt-cli. Xem docs/DEEPX_UBUNTU_PREP.md.')
    status = subprocess.run(['dxrt-cli', '-s'], text=True, capture_output=True, timeout=20)
    require(
        status.returncode == 0 and re.search(r'Device\s+\d+\s*:\s*M1\b', status.stdout),
        'Chưa nhận DX-M1. Chạy sudo modprobe dx_dma && sudo modprobe dxrt_driver rồi dxrt-cli -s.\n'
        + status.stdout
        + status.stderr,
    )
    print('DX-M1: OK | Decoder: CPU | Detector: CPU', flush=True)
    print('Encoder run:', run, flush=True)
    if verbose:
        print(status.stdout)
        for tool in ('zenity', 'ffmpeg', 'ffprobe', 'vlc'):
            print(f'{tool}: ' + (shutil.which(tool) or 'chưa cài'))
        print('Thiếu zenity: dùng --source hoặc dán đường dẫn. Video cần ffmpeg + ffprobe; VLC dùng để xem.')
    return status.stdout


def command_for(repo, run, source, output, max_frames, every_n_frames, layout):
    return [
        sys.executable,
        str(repo / 'alpr_auto_dxm1.py'),
        '--dx-run',
        str(run),
        '--source',
        str(source),
        '--device',
        'cpu',
        '--output',
        str(output),
        '--max-frames',
        str(max_frames),
        '--every-n-frames',
        str(every_n_frames),
        '--layout',
        layout,
    ]


def transcode_preview(source):
    target = source.with_name(source.stem + '_H264.mp4')
    require(not target.exists(), f'Đã có output: {target}')
    subprocess.run(
        [
            'ffmpeg',
            '-hide_banner',
            '-loglevel',
            'error',
            '-nostdin',
            '-xerror',
            '-n',
            '-i',
            str(source),
            '-map',
            '0:v:0',
            '-c:v',
            'libx264',
            '-preset',
            'veryfast',
            '-crf',
            '20',
            '-threads',
            '2',
            '-pix_fmt',
            'yuv420p',
            '-movflags',
            '+faststart',
            '-an',
            str(target),
        ],
        check=True,
    )
    probed = subprocess.run(
        [
            'ffprobe',
            '-v',
            'error',
            '-select_streams',
            'v:0',
            '-count_frames',
            '-show_entries',
            'stream=codec_name,pix_fmt,nb_read_frames',
            '-of',
            'json',
            str(target),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    stream = json.loads(probed.stdout)['streams'][0]
    require(
        stream['codec_name'] == 'h264' and stream['pix_fmt'] == 'yuv420p' and int(stream['nb_read_frames']) > 0,
        'Video H.264 không có frame hợp lệ.',
    )
    return target, int(stream['nb_read_frames'])


def write_gallery(output, report, images, videos):
    def escaped(value):
        return html.escape(str(value), quote=True)

    with (output / 'detections.csv').open(encoding='utf-8-sig', newline='') as stream:
        detections = list(csv.DictReader(stream))
    counts = Counter(row.get('text', '') for row in detections if row.get('status') == 'ok' and row.get('text'))
    parts = [
        '<!doctype html><html lang="vi"><meta charset="utf-8"><title>DX-M1 ALPR demo</title>',
        '<style>body{font:18px system-ui;max-width:1100px;margin:30px auto;padding:0 18px;background:#111827;color:#f9fafb}img,video{max-width:100%;max-height:75vh}figure{margin:24px 0}a{color:#93c5fd}li{margin:8px 0}</style>',
        '<h1>Demo nhận diện biển số Việt Nam</h1>',
        '<p>Encoder PARSeq: DX-M1 · Detector và decoder: CPU</p>',
        '<p>Ảnh/frame tự detect và OCR; không có đáp án chuẩn đưa vào inference.</p>',
        f"<p>Frame đã xử lý: {report['processed_frames']} · Detections: {report['detections']} · "
        f"Events: {report['events']} · Lỗi OCR: {report['ocr_errors']}</p>",
        '<p>Số detections/events không phải phép đếm xe duy nhất đã kiểm chứng. Nguồn mới chưa có nhãn nên chưa có accuracy.</p>',
    ]
    if report['ocr_errors']:
        parts.append('<p><strong>Có lỗi OCR; xem summary.json và detections.csv.</strong></p>')
    if not report['detections']:
        parts.append('<p><strong>Không phát hiện biển trong phần dữ liệu đã xử lý.</strong></p>')
    if counts:
        parts.append('<details><summary>Text đã đọc và số lần quan sát</summary><ul>')
        parts.extend(f'<li>{escaped(text)}: {count} lần</li>' for text, count in counts.most_common())
        parts.append('</ul></details>')
    for path in images:
        parts.append(f'<figure><img src="{quote(path.name)}"><figcaption>{escaped(path.name)}</figcaption></figure>')
    if videos:
        parts.append('<p>Video bên dưới đã xử lý rồi phát lại; tốc độ phát không phải FPS inference.</p>')
    for path in videos:
        parts.append(
            f'<figure><video controls preload="metadata" src="{quote(path.name)}"></video>'
            f'<figcaption><a href="{quote(path.name)}">{escaped(path.name)}</a></figcaption></figure>'
        )
    parts.append(
        '<p><a href="detections.csv">Detections CSV</a> · <a href="events.csv">Events CSV</a> · '
        '<a href="summary.json">Summary</a></p></html>'
    )
    (output / 'index.html').write_text('\n'.join(parts), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parent, help=argparse.SUPPRESS)
    parser.add_argument('--source', help='Ảnh, video hoặc thư mục mới; bỏ trống để chọn file')
    parser.add_argument('--dx-run', help='Full MatMul encoder run; mặc định tìm theo hash mốc đã kiểm chứng')
    parser.add_argument('--check', action='store_true', help='Chỉ kiểm tra môi trường/card, không inference')
    parser.add_argument('--max-frames', type=int, default=0, help='Giới hạn frame xử lý mỗi video; 0 = toàn bộ')
    parser.add_argument('--every-n-frames', type=int, default=1, help='1 = xử lý mọi frame; tăng để lấy mẫu')
    parser.add_argument('--layout', choices=('auto', 'one', 'two'), default='auto')
    parser.add_argument('--output', type=Path, help='Thư mục output MỚI; mặc định tạo trong results')
    parser.add_argument('--no-open', action='store_true', help='Chỉ tạo output, không mở trình duyệt/VLC')
    args = parser.parse_args(argv)
    try:
        require(args.max_frames >= 0 and args.every_n_frames >= 1, 'Giới hạn frame không âm; bước lấy mẫu >= 1.')
        repo = args.repo.resolve()
        run = resolve_run(repo, args.dx_run)
        status = check_environment(repo, run, args.check)
        if args.check:
            return 0
        source = parse_source(args.source) if args.source else choose_source()
        if source is None:
            print('Đã hủy chọn file. Không chạy inference.')
            return 0
        sources = [source] if source.is_file() else [p for p in source.rglob('*') if p.is_file()]
        has_video = any(p.suffix.lower() in VIDEO_EXTENSIONS for p in sources)
        if has_video:
            require(
                shutil.which('ffmpeg') and shutil.which('ffprobe'), 'Video cần ffmpeg: sudo apt-get install -y ffmpeg'
            )
        output = (
            args.output.expanduser()
            if args.output
            else repo / 'results' / datetime.now().strftime('demo_%Y%m%d_%H%M%S_%f')
        )
        output = (repo / output).resolve() if not output.is_absolute() else output.resolve()
        require(not output.exists(), 'Output đã tồn tại; chọn một thư mục mới.')
        require(
            not source.is_dir() or source not in output.parents,
            'Chọn thư mục dữ liệu riêng, không chọn repo/results chứa output.',
        )
        print('Nguồn mới:', source, flush=True)
        if has_video:
            print(
                f'Video: max_frames={args.max_frames} (0=toàn bộ), every_n_frames={args.every_n_frames}. '
                'Xử lý xong mới mở video; đây không phải camera live.',
                flush=True,
            )
        command = command_for(repo, run, source, output, args.max_frames, args.every_n_frames, args.layout)
        environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        result = subprocess.run(command, cwd=repo, env=environment)
        require(result.returncode == 0, f'Pipeline dừng, mã {result.returncode}. Giữ log và output tại {output}.')
        report = read_json(output / 'summary.json')
        require(
            report['status'] == 'complete'
            and report.get('ocr_encoder_backend') == 'DX-M1'
            and report.get('ocr_decoder_device') == 'cpu',
            'Báo cáo chưa xác nhận chạy DX-M1 + decoder CPU hoàn tất.',
        )
        require(
            report['deepx_encoder']['dxnn_sha256'] == read_json(run / 'matmul_contract.json')['dxnn_sha256'],
            'Báo cáo dùng DXNN hash khác encoder đã chọn.',
        )
        videos = []
        frame_counts = {}
        for raw in sorted(output.glob('*_preview.mp4')):
            print('Xuất H.264:', raw.name, flush=True)
            encoded, count = transcode_preview(raw)
            videos.append(encoded)
            frame_counts[encoded.name] = count
            index = int(raw.name.split('_', 1)[0]) - 1
            require(count == report['sources'][index]['processed_frames'], 'Số frame video khác số frame đã xử lý.')
        images = sorted(output.glob('*.jpg'))
        require(images or videos, 'Pipeline không tạo ảnh/video có khung; kiểm tra log và quyền ghi output.')
        write_gallery(output, report, images, videos)
        (output / 'demo_run.json').write_text(
            json.dumps(
                {
                    'command': command,
                    'source': str(source),
                    'dx_run': str(run),
                    'dxnn_sha256': report['deepx_encoder']['dxnn_sha256'],
                    'dxrt_status': status,
                    'video_frames': frame_counts,
                    'scope': 'New local input, recorded DX-M1 encoder + CPU decoder demo; no accuracy labels.',
                },
                ensure_ascii=False,
                indent=2,
            )
            + '\n',
            encoding='utf-8',
        )
        print('KẾT QUẢ:', output)
        print('MỞ:', output / 'index.html')
        if not args.no_open:
            if shutil.which('xdg-open'):
                subprocess.Popen(
                    ['xdg-open', str(output / 'index.html')], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            else:
                print('Không có xdg-open; mở index.html bằng trình duyệt hoặc xem output trực tiếp.')
            if videos and shutil.which('vlc'):
                subprocess.Popen(
                    ['vlc', '--loop', str(videos[0])], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(f'STOP: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
