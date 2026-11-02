#!/usr/bin/env python3
"""Small persistent input queue and menu for the existing DX-M1 demo launcher."""
import argparse
import json
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

import demo_alpr as demo


def state_path(repo):
    return repo / 'inputs/demo_inbox/state.json'


def read_state(repo):
    path = state_path(repo)
    if not path.exists():
        return {'version': 1, 'items': [], 'latest_output': None}
    state = demo.read_json(path)
    demo.require(
        state.get('version') == 1 and isinstance(state.get('items'), list),
        'State không hợp lệ; giữ file để kiểm tra, không tự reset.',
    )
    hashes = set()
    for item in state['items']:
        demo.require(item.get('sha256') and item['sha256'] not in hashes, 'State có input trùng hoặc thiếu hash.')
        hashes.add(item['sha256'])
        media_path(repo, item)
    return state


def save_state(repo, state):
    path = state_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('state_' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(state, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def media_path(repo, item):
    relative = Path(item['file'])
    demo.require(not relative.is_absolute() and '..' not in relative.parts, 'Input state phải nằm trong inbox.')
    path = (repo / relative).resolve()
    path.relative_to((repo / 'inputs/demo_inbox/files').resolve())
    demo.require(
        path.suffix.lower() in demo.IMAGE_EXTENSIONS | demo.VIDEO_EXTENSIONS, 'Input state không phải ảnh/video.'
    )
    return path


def add_inputs(repo, values):
    state = read_state(repo)
    sources = [demo.parse_source(value) for value in values] if values else []
    if not sources:
        source = demo.choose_source()
        if source is None:
            print('Đã hủy. Hàng đợi giữ nguyên.')
            return 0
        sources = [source]
    files = []
    for source in sources:
        files.extend(
            [source]
            if source.is_file()
            else sorted(
                p
                for p in source.rglob('*')
                if p.is_file() and p.suffix.lower() in demo.IMAGE_EXTENSIONS | demo.VIDEO_EXTENSIONS
            )
        )
    inbox = repo / 'inputs/demo_inbox/files'
    inbox.mkdir(parents=True, exist_ok=True)
    known = {item['sha256'] for item in state['items']}
    added = 0
    for source in files:
        digest = demo.sha256(source)
        if digest in known:
            print('Đã có trong hàng đợi:', source.name)
            continue
        target = inbox / (digest[:12] + '_' + source.stem[:40] + source.suffix.lower())
        if target.exists():
            demo.require(demo.sha256(target) == digest, f'File cache đã đổi: {target}')
        else:
            temporary = inbox / ('copy_' + uuid.uuid4().hex + '.tmp')
            try:
                shutil.copyfile(source, temporary)
                demo.require(
                    demo.sha256(temporary) == digest, 'File nguồn thay đổi trong lúc sao chép; thêm lại input.'
                )
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        state['items'].append({'name': source.name, 'file': str(target.relative_to(repo)), 'sha256': digest})
        known.add(digest)
        added += 1
    save_state(repo, state)
    print(f'Đã thêm {added} file. Hàng đợi: {len(state["items"])} file. Chạy ./alpr scan để quét.')
    return 0


def show_status(repo):
    state = read_state(repo)
    print(f'Input đang chọn: {len(state["items"])} file')
    for index, item in enumerate(state['items'], 1):
        path = media_path(repo, item)
        print(f'  {index}. {item["name"]}' + ('' if path.is_file() else ' [file cache đang thiếu]'))
    if state.get('latest_output'):
        print('Kết quả thành công gần nhất:', repo / state['latest_output'])
    else:
        print('Chưa có kết quả thành công được lưu bởi ./alpr.')
    return 0


def latest_output(repo):
    state = read_state(repo)
    demo.require(state.get('latest_output'), 'Chưa có kết quả. Thêm input và chạy ./alpr scan trước.')
    relative = Path(state['latest_output'])
    demo.require(not relative.is_absolute() and '..' not in relative.parts, 'Output state không hợp lệ.')
    output = (repo / relative).resolve()
    output.relative_to((repo / 'results').resolve())
    demo.require(
        (output / 'index.html').is_file(), 'Kết quả đã bị di chuyển/xóa. Giữ state và kiểm tra thư mục results.'
    )
    return output


def open_output(repo, folder=False):
    output = latest_output(repo)
    target = output if folder else output / 'index.html'
    demo.require(shutil.which('xdg-open'), f'Không có xdg-open. Mở trực tiếp: {target}')
    subprocess.Popen(['xdg-open', str(target)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not folder:
        videos = sorted(output.glob('*_H264.mp4'))
        if videos and shutil.which('vlc'):
            subprocess.Popen(['vlc', '--loop', str(videos[0])], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print('MỞ:', target)
    return 0


def scan(repo, max_frames=120, no_open=False, dx_run=None):
    demo.require(max_frames >= 0, 'max-frames phải >= 0; 0 nghĩa là toàn bộ video.')
    state = read_state(repo)
    demo.require(state['items'], 'Chưa chọn input. Chạy ./alpr add trước.')
    # Check every queued file before creating a new batch or invoking inference.
    for item in state['items']:
        path = media_path(repo, item)
        demo.require(
            path.is_file() and demo.sha256(path) == item['sha256'], f'Input cache thiếu/đã đổi: {item["name"]}'
        )
    tag = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    batch = repo / 'inputs/demo_batches' / tag
    output = repo / 'results' / ('alpr_' + tag)
    batch.mkdir(parents=True, exist_ok=False)
    for index, item in enumerate(state['items'], 1):
        source = media_path(repo, item)
        shutil.copyfile(source, batch / f'{index:03d}_{source.name}')
    print(f'QUÉT {len(state["items"])} input | Encoder DX-M1 | Detector/decoder CPU', flush=True)
    print(f'Video: tối đa {max_frames} frame/input (0=toàn bộ), xử lý mọi frame rồi phát kết quả.', flush=True)
    command = [
        '--repo',
        str(repo),
        '--source',
        str(batch),
        '--output',
        str(output),
        '--max-frames',
        str(max_frames),
        '--every-n-frames',
        '1',
        '--no-open',
    ]
    if dx_run:
        command += ['--dx-run', dx_run]
    result = demo.main(command)
    if result:
        print('Lượt mới chưa hoàn tất. Hàng đợi và kết quả thành công trước đó giữ nguyên.', file=sys.stderr)
        return result
    report = demo.read_json(output / 'summary.json')
    demo.require(
        report.get('status') == 'complete'
        and report.get('ocr_encoder_backend') == 'DX-M1'
        and (output / 'index.html').is_file(),
        'Lượt quét chưa có báo cáo DX-M1 hoàn tất.',
    )
    state['latest_output'] = str(output.relative_to(repo))
    save_state(repo, state)
    print('ĐÃ LƯU:', output)
    if not no_open:
        return open_output(repo)
    return 0


def start(repo, dx_run=None):
    subprocess.run(['sudo', 'modprobe', 'dx_dma'], check=True)
    subprocess.run(['sudo', 'modprobe', 'dxrt_driver'], check=True)
    args = ['--repo', str(repo), '--check']
    if dx_run:
        args += ['--dx-run', dx_run]
    return demo.main(args)


def menu(repo):
    demo.require(sys.stdin.isatty(), 'Dùng ./alpr add, scan, open hoặc help trong terminal này.')
    commands = {'1': 'start', '2': 'add', '3': 'scan', '4': 'open', '5': 'status', '6': 'clear'}
    while True:
        print(
            '\nDX-M1 ALPR\n1. Chuẩn bị card\n2. Thêm ảnh/video\n3. Quét input và mở kết quả\n'
            '4. Mở lại kết quả\n5. Xem input đang chọn\n6. Bỏ chọn toàn bộ input\n0. Thoát'
        )
        choice = input('Chọn số: ').strip()
        if choice == '0':
            return 0
        if choice in commands:
            main(['--repo', str(repo), commands[choice]])
        else:
            print('Chọn số từ 0 đến 6.')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='alpr', description='Menu thêm input, quét DX-M1 và mở kết quả.')
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parent, help=argparse.SUPPRESS)
    subs = parser.add_subparsers(dest='action')
    subs.add_parser('menu', help='Mở menu thao tác')
    add = subs.add_parser('add', help='Chọn file hoặc thêm các đường dẫn vào hàng đợi')
    add.add_argument('paths', nargs='*')
    prepare = subs.add_parser('start', help='Nạp driver (hỏi mật khẩu sudo) và kiểm tra DX-M1')
    prepare.add_argument('--dx-run')
    checking = subs.add_parser('check', help='Chỉ kiểm tra, không nạp driver')
    checking.add_argument('--dx-run')
    scanning = subs.add_parser('scan', help='Quét input đã thêm; mặc định 120 frame/video')
    limit = scanning.add_mutually_exclusive_group()
    limit.add_argument('--all', action='store_true', help='Quét toàn bộ video')
    limit.add_argument('--max-frames', type=int, default=120)
    scanning.add_argument('--no-open', action='store_true')
    scanning.add_argument('--dx-run')
    opening = subs.add_parser('open', help='Mở lại kết quả thành công gần nhất')
    opening.add_argument('--folder', action='store_true', help='Mở thư mục chứa CSV và media')
    subs.add_parser('status', help='Liệt kê input và kết quả gần nhất')
    subs.add_parser('clear', help='Bỏ chọn input; giữ file gốc, bản sao và kết quả')
    subs.add_parser('help', help='Xem các lệnh')
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    try:
        if args.action in (None, 'menu'):
            return menu(repo)
        if args.action == 'help':
            parser.print_help()
            return 0
        if args.action == 'add':
            return add_inputs(repo, args.paths)
        if args.action == 'status':
            return show_status(repo)
        if args.action == 'clear':
            state = read_state(repo)
            state['items'] = []
            save_state(repo, state)
            print('Đã bỏ chọn input. File gốc, bản sao trong inputs và kết quả vẫn được giữ.')
            return 0
        if args.action == 'start':
            return start(repo, args.dx_run)
        if args.action == 'check':
            return demo.main(['--repo', str(repo), '--check'] + (['--dx-run', args.dx_run] if args.dx_run else []))
        if args.action == 'scan':
            return scan(repo, 0 if args.all else args.max_frames, args.no_open, args.dx_run)
        return open_output(repo, args.folder)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(f'STOP: {exc}', file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print('\nĐã dừng. Input và kết quả đã lưu được giữ nguyên.')
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
