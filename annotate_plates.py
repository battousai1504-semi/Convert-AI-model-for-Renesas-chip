#!/usr/bin/env python3
"""Create manually labeled, single-line OCR crops from images or sampled video.

No detector or OCR runs here. Each saved record is one plate with one or two
manually selected line boxes, top to bottom. Originals are never modified.
"""
import argparse
import csv
import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from ocr_batch import EXTENSIONS, normalize

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}


def atomic_text(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(value, encoding='utf-8')
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def crop_hash(image):
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


def source_files(source):
    source = Path(source).resolve()
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = sorted(p for p in source.rglob('*') if p.is_file())
    else:
        raise ValueError(f'Source does not exist: {source}')
    paths = [p for p in paths if p.suffix.lower() in EXTENSIONS | VIDEO_EXTENSIONS]
    if not paths:
        raise ValueError(f'No supported images or videos: {source}')
    return paths


def sample_frames(path, every_seconds=2.0, max_frames=20):
    """Decode sequentially; video positions are approximate nominal-FPS times."""
    if path.suffix.lower() in EXTENSIONS:
        with Image.open(path) as original:
            frame = np.asarray(ImageOps.exif_transpose(original).convert('RGB'))
        yield cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), None, None
        return
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot open video: {path}')
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f'Video has no usable FPS: {path}')
        step = max(1, round(every_seconds * fps))
        index = emitted = 0
        while emitted < max_frames:
            ok, frame = capture.read()
            if not ok:
                if index == 0:
                    raise ValueError(f'Cannot decode first video frame: {path}')
                break
            if index % step == 0:
                yield frame, index, round(1000 * index / fps, 3)
                emitted += 1
            index += 1
    finally:
        capture.release()


def map_box(roi, display_shape, original_shape):
    """Map a displayed xywh selection to half-open original xyxy pixels."""
    x, y, width, height = roi
    if width <= 0 or height <= 0:
        raise ValueError('Selection cancelled')
    dh, dw = display_shape[:2]
    oh, ow = original_shape[:2]
    x1 = max(0, min(ow, math.floor(x * ow / dw)))
    y1 = max(0, min(oh, math.floor(y * oh / dh)))
    x2 = max(0, min(ow, math.ceil((x + width) * ow / dw)))
    y2 = max(0, min(oh, math.ceil((y + height) * oh / dh)))
    if x2 - x1 < 4 or y2 - y1 < 4:
        raise ValueError('Crop too small (minimum 4 x 4 original pixels)')
    return [x1, y1, x2, y2]


def select_line(frame, title):
    height, width = frame.shape[:2]
    ratio = min(1.0, 1280 / width, 720 / height)
    display = cv2.resize(frame, (max(1, round(width * ratio)), max(1, round(height * ratio))))
    try:
        roi = cv2.selectROI(title, display, showCrosshair=True, fromCenter=False)
    finally:
        cv2.destroyAllWindows()
    return map_box(roi, display.shape, frame.shape)


def load_dataset(destination):
    manifest = destination / 'dataset.json'
    if not manifest.exists():
        if any((destination / 'images').glob('*')):
            raise ValueError('Images exist without dataset.json; use a fresh output folder')
        return {'version': 1, 'plates': []}
    data = json.loads(manifest.read_text(encoding='utf-8'))
    if data.get('version') != 1 or not isinstance(data.get('plates'), list):
        raise ValueError('Unsupported dataset.json')
    expected = []
    ids = set()
    for plate in data['plates']:
        if plate['plate_id'] in ids or len(plate['lines']) not in (1, 2):
            raise ValueError('Invalid or duplicate plate record')
        ids.add(plate['plate_id'])
        for line in plate['lines']:
            name = line['image']
            if Path(name).name != name or not normalize(line['text']):
                raise ValueError('Invalid image name or label in dataset.json')
            expected.append(name)
            image_path = destination / 'images' / name
            with Image.open(image_path) as image:
                pixels = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2BGR)
            if crop_hash(pixels) != line['pixel_sha256']:
                raise ValueError(f'Crop modified since labeling: {name}')
    actual = {p.name for p in (destination / 'images').glob('*') if p.is_file()}
    if len(expected) != len(set(expected)) or set(expected) != actual:
        raise ValueError('Images and dataset.json disagree; restore the dataset before appending')
    return data


def export_labels(destination, data):
    """Regenerate a CSV compatible with ocr_batch.py; JSON is authoritative."""
    path = destination / 'labels.csv'
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['image', 'text'])
        writer.writeheader()
        for plate in data['plates']:
            writer.writerows({'image': line['image'], 'text': line['text']} for line in plate['lines'])
    temporary.replace(path)


def save_plate(destination, data, frame, boxes, texts, source_info):
    if len(boxes) != len(texts) or len(boxes) not in (1, 2):
        raise ValueError('A plate must have one or two labeled lines')
    if len(boxes) == 2 and sum(boxes[0][1::2]) >= sum(boxes[1][1::2]):
        raise ValueError('Select TOP line first, then BOTTOM line')
    pixels = []
    seen = {line['pixel_sha256'] for plate in data['plates'] for line in plate['lines']}
    for box, text in zip(boxes, texts):
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]):
            raise ValueError('Box outside image')
        if not normalize(text) or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .-' for c in text):
            raise ValueError('Use visible A-Z / 0-9 text (spaces, dots and hyphens allowed)')
        crop = frame[y1:y2, x1:x2].copy()
        digest = crop_hash(crop)
        if digest in seen:
            raise ValueError('Exact duplicate crop: not saved as a new evaluation sample')
        seen.add(digest)
        pixels.append((crop, digest))
    plate_id = uuid.uuid4().hex[:16]
    record = dict(source_info, plate_id=plate_id, width=frame.shape[1], height=frame.shape[0],
                  created_utc=datetime.now(timezone.utc).isoformat(), lines=[])
    written = []
    try:
        for index, (box, text, (crop, digest)) in enumerate(zip(boxes, texts, pixels), 1):
            name = f'{plate_id}_line{index}.png'
            path = destination / 'images' / name
            written.append(path)
            Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(path)
            record['lines'].append({'image': name, 'text': text, 'box_xyxy': box, 'pixel_sha256': digest})
        updated = dict(data, plates=data['plates'] + [record])
        atomic_text(destination / 'dataset.json', json.dumps(updated, ensure_ascii=False, indent=2))
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    data['plates'].append(record)
    export_labels(destination, data)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', help='Image/video file or folder (local)')
    parser.add_argument('--output', default='data/vn_ocr_v1')
    parser.add_argument('--export-only', action='store_true', help='Validate saved crops and rebuild labels.csv after editing dataset.json')
    parser.add_argument('--every-seconds', type=float, default=2.0)
    parser.add_argument('--max-frames', type=int, default=20, help='Sample limit per video')
    args = parser.parse_args(argv)
    if not math.isfinite(args.every_seconds) or args.every_seconds <= 0 or args.max_frames < 1:
        parser.error('Use positive --every-seconds and --max-frames')
    destination = Path(args.output).resolve()
    if args.export_only:
        try:
            if not (destination / 'dataset.json').is_file():
                raise ValueError('No dataset.json found')
            data = load_dataset(destination)
            export_labels(destination, data)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(str(exc))
        print(f'Rebuilt labels for {len(data["plates"])} plates: {destination / "labels.csv"}')
        return 0
    if not args.source:
        parser.error('--source is required unless --export-only is used')
    try:
        sources = source_files(args.source)
        if any(p == destination or destination in p.parents for p in sources):
            raise ValueError('Keep the source folder separate from the output dataset')
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'images').mkdir(exist_ok=True)
        data = load_dataset(destination)
        if not (destination / 'dataset.json').exists():
            atomic_text(destination / 'dataset.json', json.dumps(data, indent=2))
        export_labels(destination, data)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print('Manual labels only. Select ONE text line tightly; include all strokes.')
    print('ROI window: drag mouse, ENTER confirms, C cancels. Then return to this terminal.')
    print('For two-line plates: TOP first, then BOTTOM. Ctrl+C exits, saved work is kept.')
    print(f'Dataset: {destination}')
    try:
        for source in sources:
            # One source hash groups all frames of the same video. Never split
            # that group across calibration/training and held-out evaluation.
            digest = file_hash(source)
            for frame, frame_index, timestamp_ms in sample_frames(source, args.every_seconds, args.max_frames):
                print(f'\nSource: {source.name}; frame={frame_index}; approx_ms={timestamp_ms}')
                while True:
                    action = input('[1] one-line plate / [2] two-line plate / [n] next frame / [q] quit: ').strip().lower()
                    if action == 'q':
                        return 0
                    if action == 'n':
                        break
                    if action not in ('1', '2'):
                        continue
                    boxes, texts = [], []
                    try:
                        for line in range(1, int(action) + 1):
                            print(f'Select line {line}/{action} in the image window.')
                            boxes.append(select_line(frame, f'Plate line {line} - ENTER confirm - C cancel'))
                            texts.append(input(f'Visible text for line {line} (blank cancels this plate): ').strip())
                            if not texts[-1]:
                                raise ValueError('Plate cancelled; nothing saved')
                        record = save_plate(destination, data, frame, boxes, texts,
                            {'source_name': source.name, 'source_sha256': digest,
                             'group_id': digest, 'frame_index': frame_index,
                             'timestamp_ms_approx': timestamp_ms, 'label_method': 'manual'})
                        print(f"Saved plate {record['plate_id']}: {' / '.join(texts)}")
                    except ValueError as exc:
                        print(f'Not saved: {exc}')
    except (KeyboardInterrupt, EOFError):
        print('\nStopped; previously saved labels are kept.')
    except (OSError, cv2.error, ValueError) as exc:
        print(f'Stopped: {exc}')
        return 1
    finally:
        cv2.destroyAllWindows()
        print(f"Saved total: {len(data['plates'])} plates, {sum(len(p['lines']) for p in data['plates'])} lines.")
        print(f'Labels: {destination / "labels.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
