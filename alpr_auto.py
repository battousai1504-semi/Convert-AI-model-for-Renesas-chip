#!/usr/bin/env python3
"""Automatic plate detection + PARSeq OCR for local images and video.

No annotation/ground-truth input is required. Detector runs on ONNX Runtime
CPU; --device selects PARSeq CPU/CUDA. This is a PC baseline, not a DEEPX
runtime. Two-row splitting and IoU tracking are initial heuristics.

Detector input/output contract follows ankandrew/open-image-models (MIT),
see third_party/open-image-models-LICENSE.txt and docs/ALPR_AUTO_WINDOWS.md.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen

import cv2
import numpy as np
from PIL import Image, ImageOps

from ocr_batch import EXTENSIONS, normalize

MODEL_NAME = 'yolo-v9-t-640-license-plates-end2end.onnx'
MODEL_URL = 'https://github.com/ankandrew/open-image-models/releases/download/assets/' + MODEL_NAME
MODEL_SHA256 = 'c3c1026ca7d0585dd88084d68182dd897113712fa734ae1557ca70174440c076'
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
DETECTION_FIELDS = ['source', 'frame_index', 'timestamp_ms_approx', 'track_id',
                    'x1', 'y1', 'x2', 'y2', 'detector_score', 'rectified', 'layout', 'split_y',
                    'line1', 'line2', 'raw_text', 'text', 'ocr_score',
                    'needs_review', 'status', 'error']
EVENT_FIELDS = ['source', 'track_id', 'first_ms_approx', 'last_ms_approx',
                'observations', 'text', 'votes', 'vote_fraction', 'ocr_score',
                'best_frame_index', 'needs_review']
FRAME_FIELDS = ['source', 'frame_index', 'timestamp_ms_approx', 'detections',
                'ocr_errors', 'status', 'compute_ms', 'error']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def detector_model(custom=None):
    if custom:
        path = Path(custom).resolve()
        if not path.is_file():
            raise ValueError(f'Detector model not found: {path}')
        return path
    path = Path(__file__).resolve().parent / 'artifacts' / 'models' / MODEL_NAME
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.' + uuid.uuid4().hex + '.part')
        print(f'Downloading plate detector (7.8 MB): {MODEL_NAME}', flush=True)
        try:
            with urlopen(MODEL_URL, timeout=30) as response, temporary.open('wb') as output:
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    output.write(chunk)
            if sha256(temporary) != MODEL_SHA256:
                raise ValueError('Detector download failed SHA256 verification')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256(path) != MODEL_SHA256:
        raise ValueError(f'Detector checksum mismatch; remove the damaged cache file: {path}')
    return path


def letterbox(frame, size):
    height, width = frame.shape[:2]
    ratio = min(size / height, size / width)
    new_width, new_height = round(width * ratio), round(height * ratio)
    dx, dy = (size - new_width) / 2, (size - new_height) / 2
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, round(dy - .1), round(dy + .1),
                               round(dx - .1), round(dx + .1), cv2.BORDER_CONSTANT,
                               value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255, ratio, (dx, dy)


def decode_boxes(predictions, ratio, padding, shape, threshold):
    predictions = np.asarray(predictions)
    if predictions.ndim == 3 and predictions.shape[0] == 1:
        predictions = predictions[0]
    if predictions.ndim != 2 or predictions.shape[1] != 7:
        raise ValueError(f'Expected end-to-end YOLOv9 [N,7] output, received {predictions.shape}')
    height, width = shape[:2]
    boxes = []
    for row in predictions:
        if not np.isfinite(row).all():
            raise ValueError('Detector returned NaN/Inf')
        batch, x1, y1, x2, y2, category, score = map(float, row)
        if batch != 0 or category != 0 or score < threshold:
            continue
        x1, x2 = (x1 - padding[0]) / ratio, (x2 - padding[0]) / ratio
        y1, y2 = (y1 - padding[1]) / ratio, (y2 - padding[1]) / ratio
        box = [max(0, min(width, math.floor(x1))), max(0, min(height, math.floor(y1))),
               max(0, min(width, math.ceil(x2))), max(0, min(height, math.ceil(y2)))]
        if box[2] > box[0] and box[3] > box[1]:
            boxes.append({'box': box, 'score': score})
    return sorted(boxes, key=lambda item: item['score'], reverse=True)


class PlateDetector:
    def __init__(self, path, threshold=.35):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError('Detector must have one image input')
        shape = inputs[0].shape
        if len(shape) != 4 or shape[1] != 3 or not isinstance(shape[2], int) or shape[2] != shape[3]:
            raise ValueError(f'Expected detector NCHW square RGB input, received {shape}')
        if inputs[0].type != 'tensor(float)':
            raise ValueError('Detector requires FP32 input')
        self.input_name, self.size, self.threshold = inputs[0].name, shape[2], threshold

    def predict(self, frame):
        tensor, ratio, padding = letterbox(frame, self.size)
        output = self.session.run(None, {self.input_name: tensor})[0]
        return decode_boxes(output, ratio, padding, frame.shape, self.threshold)


def rectify_plate(crop):
    """Use a strong quadrilateral plate boundary when one is visible.

    Fall back to the original crop if the boundary is missing/ambiguous. This
    is geometric preprocessing, not a trained keypoint/rectification model.
    """
    height, width = crop.shape[:2]
    if height < 20 or width < 30:
        return crop, False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        area = cv2.contourArea(contour)
        if not .45 * width * height <= area <= .98 * width * height:
            continue
        polygon = cv2.approxPolyDP(contour, .025 * cv2.arcLength(contour, True), True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        points = polygon.reshape(4, 2).astype(np.float32)
        sums, differences = points.sum(axis=1), np.diff(points, axis=1).ravel()
        ordered = np.array([points[np.argmin(sums)], points[np.argmin(differences)],
                            points[np.argmax(sums)], points[np.argmax(differences)]], dtype=np.float32)
        if len(np.unique(ordered, axis=0)) != 4:
            continue
        tl, tr, br, bl = ordered
        new_width = round((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2)
        new_height = round((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2)
        if new_width < 20 or new_height < 10 or new_width / new_height < .9:
            continue
        target = np.array([[0, 0], [new_width - 1, 0], [new_width - 1, new_height - 1], [0, new_height - 1]], np.float32)
        transform = cv2.getPerspectiveTransform(ordered, target)
        return cv2.warpPerspective(crop, transform, (new_width, new_height)), True
    return crop, False


def line_regions(crop, layout='auto'):
    """Choose one/two rows by geometry; estimate the gap using Otsu projection.

    No text or legal plate pattern is used to force a desired prediction.
    This assumes near-upright plates; angled/occluded plates need later work.
    """
    height, width = crop.shape[:2]
    if layout == 'one' or (layout == 'auto' and width / height >= 2.5):
        return [(0, height)], None
    if height < 12:
        return [(0, height)], None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Exclude plate borders from the projection. Support light-on-dark plates
    # by treating the minority intensity as foreground.
    inner = mask[max(1, height // 10):max(2, height * 9 // 10), width // 10:max(width // 10 + 1, width * 9 // 10)]
    if np.mean(inner) > 127:
        mask = 255 - mask
    profile = (mask[:, width // 10:max(width // 10 + 1, width * 9 // 10)] > 0).mean(axis=1)
    kernel = max(1, round(height * .04))
    smooth = np.convolve(profile, np.ones(kernel) / kernel, mode='same')
    low, high = max(3, round(height * .30)), min(height - 3, round(height * .67))
    candidates = np.arange(low, high)
    # Prefer a low-ink gap, with a weak center preference to break flat ties.
    costs = smooth[candidates] + .04 * np.abs(candidates / height - .48)
    split = int(candidates[np.argmin(costs)])
    # Slight overlap preserves strokes when the estimated gap is imperfect.
    overlap = max(1, round(height * .02))
    return [(0, min(height, split + overlap)), (max(0, split - overlap), height)], split


class ParseqReader:
    def __init__(self, device='cpu', checkpoint='pretrained=parseq'):
        import torch
        from strhub.data.module import SceneTextDataModule
        from strhub.models.utils import load_from_checkpoint
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; use --device cpu or repair the GPU environment')
        self.torch, self.device = torch, device
        self.model = load_from_checkpoint(checkpoint).eval().to(device)
        self.transform = SceneTextDataModule.get_transform(self.model.hparams.img_size)

    def read(self, crop, layout='auto'):
        regions, split = line_regions(crop, layout)
        texts, probabilities = [], []
        with self.torch.inference_mode():
            for top, bottom in regions:
                image = Image.fromarray(cv2.cvtColor(crop[top:bottom], cv2.COLOR_BGR2RGB))
                tensor = self.transform(image).unsqueeze(0).to(self.device)
                predictions, scores = self.model.tokenizer.decode(self.model(tensor).softmax(-1))
                texts.append(predictions[0])
                probabilities.extend(scores[0].detach().cpu().reshape(-1).tolist())
        score = math.exp(sum(math.log(max(float(p), 1e-12)) for p in probabilities) / len(probabilities)) if probabilities else 0
        return texts, score, split


def iou(a, b):
    width, height = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0


class PlateTracks:
    """Small one-to-one IoU tracker, per source. Not a vehicle identity model."""
    def __init__(self, source, max_gap=1.0, threshold=.2):
        self.source, self.max_gap, self.threshold = source, max_gap, threshold
        self.active, self.next_id = {}, 1

    @staticmethod
    def event(track):
        votes = track['votes']
        if votes:
            winner = max(votes, key=lambda text: (votes[text], track['best'][text]['ocr_score'], text))
            best = track['best'][winner]
            count = votes[winner]
        else:
            winner, count, best = '', 0, track['first_row']
        fraction = count / track['observations']
        return dict(source=best['source'], track_id=track['id'],
                    first_ms_approx=round(track['first'] * 1000, 3), last_ms_approx=round(track['last'] * 1000, 3),
                    observations=track['observations'], text=winner, votes=count, vote_fraction=round(fraction, 4),
                    ocr_score=best['ocr_score'], best_frame_index=best['frame_index'],
                    needs_review=int(not winner or fraction < .6 or track['observations'] < 2 or bool(best['needs_review'])))

    def update(self, rows, seconds):
        events = []
        for identity in list(self.active):
            if seconds - self.active[identity]['last'] > self.max_gap:
                events.append(self.event(self.active.pop(identity)))
        pairs = []
        for identity, track in self.active.items():
            for index, row in enumerate(rows):
                overlap = iou(track['box'], [row[k] for k in ('x1', 'y1', 'x2', 'y2')])
                if overlap >= self.threshold:
                    pairs.append((overlap, identity, index))
        assigned_tracks, assigned_rows, matches = set(), set(), {}
        for _, identity, index in sorted(pairs, reverse=True):
            if identity not in assigned_tracks and index not in assigned_rows:
                matches[index] = identity
                assigned_tracks.add(identity)
                assigned_rows.add(index)
        for index, row in enumerate(rows):
            identity = matches.get(index)
            if identity is None:
                identity = self.next_id
                self.next_id += 1
                self.active[identity] = {'id': identity, 'first': seconds, 'last': seconds,
                    'observations': 0, 'votes': Counter(), 'best': {}, 'first_row': dict(row), 'box': None}
            row['track_id'] = identity
            track = self.active[identity]
            track['last'], track['box'] = seconds, [row[k] for k in ('x1', 'y1', 'x2', 'y2')]
            track['observations'] += 1
            if row['status'] == 'ok' and row['text']:
                text = row['text']
                track['votes'][text] += 1
                if text not in track['best'] or row['ocr_score'] > track['best'][text]['ocr_score']:
                    track['best'][text] = dict(row)
        return events

    def finish(self):
        events = [self.event(track) for track in self.active.values()]
        self.active.clear()
        return events


def recognize_frame(frame, detector, reader, source, frame_index, timestamp_ms, layout='auto', rectify=True):
    rows = []
    for item in detector.predict(frame):
        x1, y1, x2, y2 = item['box']
        row = dict.fromkeys(DETECTION_FIELDS, '')
        row.update(source=source, frame_index=frame_index, timestamp_ms_approx=timestamp_ms,
                   x1=x1, y1=y1, x2=x2, y2=y2, detector_score=round(item['score'], 5),
                   needs_review=1, status='error', ocr_score=0, rectified=0)
        if x2 - x1 < 12 or y2 - y1 < 6:
            row['error'] = 'Plate crop too small to read'
        else:
            # Keep a small margin to avoid cutting character strokes.
            margin_x, margin_y = max(1, round((x2 - x1) * .02)), max(1, round((y2 - y1) * .02))
            crop = frame[max(0, y1 - margin_y):min(frame.shape[0], y2 + margin_y),
                         max(0, x1 - margin_x):min(frame.shape[1], x2 + margin_x)]
            if rectify:
                crop, applied = rectify_plate(crop)
                row['rectified'] = int(applied)
            texts, score, split = reader.read(crop, layout)
            raw = ' / '.join(texts)
            text = ''.join(normalize(part) for part in texts)
            row.update(layout='two' if len(texts) == 2 else 'one', split_y=split if split is not None else '',
                       line1=texts[0], line2=texts[1] if len(texts) == 2 else '', raw_text=raw,
                       text=text, ocr_score=round(score, 5),
                       needs_review=int(score < .75 or not 6 <= len(text) <= 10 or any(not normalize(p) for p in texts)),
                       status='ok')
        rows.append(row)
    return rows


def draw_results(frame, rows):
    image = frame.copy()
    for row in rows:
        a, b = (row['x1'], row['y1']), (row['x2'], row['y2'])
        color = (0, 180, 255) if row['needs_review'] else (0, 200, 0)
        cv2.rectangle(image, a, b, color, 2)
        title = f"#{row['track_id']} {row['text'] or '[unreadable]'}"
        cv2.putText(image, title, (a[0], max(16, a[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
    return image


def input_sources(source):
    path = Path(source).resolve()
    if path.is_file():
        paths = [path]
    elif path.is_dir():
        paths = sorted(p for p in path.rglob('*') if p.is_file())
    else:
        raise ValueError(f'Input not found: {path}')
    paths = [p for p in paths if p.suffix.lower() in EXTENSIONS | VIDEO_EXTENSIONS]
    if not paths:
        raise ValueError('No supported image/video files in input')
    return paths


def frames(path, every_n=3, max_frames=0):
    """Yield frame, zero-based index, nominal-FPS seconds, preview FPS."""
    if path.suffix.lower() in EXTENSIONS:
        with Image.open(path) as original:
            image = np.asarray(ImageOps.exif_transpose(original).convert('RGB'))
        yield cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0, 0.0, None
        return
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot open video: {path}')
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f'Invalid FPS: {path}')
        index = emitted = 0
        while not max_frames or emitted < max_frames:
            if index % every_n == 0:
                ok, frame = capture.read()
                if not ok:
                    if index == 0:
                        raise ValueError(f'Cannot decode video: {path}')
                    break
                yield frame, index, index / fps, fps / every_n
                emitted += 1
            elif not capture.grab():
                break
            index += 1
    finally:
        capture.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Image/video or folder of images/videos')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='PARSeq device (detector always CPU in this baseline)')
    parser.add_argument('--output', help='New output folder; defaults to results/auto_TIMESTAMP')
    parser.add_argument('--detector', help='Optional compatible YOLOv9 end-to-end ONNX; default is verified pretrained download')
    parser.add_argument('--checkpoint', default='pretrained=parseq')
    parser.add_argument('--confidence', type=float, default=.35)
    parser.add_argument('--layout', choices=['auto', 'one', 'two'], default='auto')
    parser.add_argument('--every-n-frames', type=int, default=3)
    parser.add_argument('--max-frames', type=int, default=0, help='Processed frame limit per video; 0 = all')
    parser.add_argument('--max-gap-seconds', type=float, default=1.0)
    parser.add_argument('--no-preview', action='store_true', help='Do not save annotated images/video')
    parser.add_argument('--no-rectify', action='store_true', help='Disable geometric quadrilateral rectification')
    args = parser.parse_args(argv)
    if not 0 < args.confidence <= 1 or args.every_n_frames < 1 or args.max_frames < 0 or not math.isfinite(args.max_gap_seconds) or args.max_gap_seconds <= 0:
        parser.error('Invalid confidence / frame sampling / tracking gap arguments')
    try:
        sources = input_sources(args.source)
        destination = Path(args.output or Path('results') / datetime.now().strftime('auto_%Y%m%d_%H%M%S_%f')).resolve()
        if destination.exists() and any(destination.iterdir()):
            raise ValueError('Output folder is not empty; choose a fresh --output')
        source_root = Path(args.source).resolve()
        if source_root.is_dir() and (destination == source_root or source_root in destination.parents):
            raise ValueError('Keep output outside the input folder')
        model_path = detector_model(args.detector)
        detector = PlateDetector(model_path, args.confidence)
        reader = ParseqReader(args.device, args.checkpoint)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    destination.mkdir(parents=True, exist_ok=True)
    import onnxruntime as ort
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
              'platform': platform.platform(), 'torch': reader.torch.__version__, 'onnxruntime': ort.__version__,
              'detector_path': str(model_path), 'detector_sha256': sha256(model_path),
              'detector_provider': detector.session.get_providers(), 'ocr_device': args.device,
              'checkpoint': args.checkpoint, 'arguments': vars(args), 'sources': [],
              'processed_frames': 0, 'detections': 0, 'events': 0, 'ocr_errors': 0,
              'accuracy': None, 'accuracy_note': 'No ground truth is required or supplied. Model confidence is not accuracy.',
              'scope': 'Automatic PC baseline. Heuristic line splitting and IoU tracking; no DEEPX/Renesas execution.',
              'timing_note': 'Frame compute excludes decode/I/O/model load/warmup. Stream wall time includes decode and output writes. Sampling is not full-frame video processing.',
              'video_note': 'Timestamps use nominal FPS (approximate for variable frame rate). Preview is sampled, silent video.',
              'status': 'running'}
    warm = False
    handles = []
    exit_code = 0
    try:
        writers = {}
        for name, fields in [('detections', DETECTION_FIELDS), ('events', EVENT_FIELDS), ('frames', FRAME_FIELDS)]:
            handle = (destination / f'{name}.csv').open('w', encoding='utf-8-sig', newline='')
            handles.append(handle)
            writers[name] = csv.DictWriter(handle, fieldnames=fields)
            writers[name].writeheader()
        print(f'Output: {destination}', flush=True)
        for source_index, source in enumerate(sources, 1):
            track = PlateTracks(str(source), args.max_gap_seconds)
            preview = None
            source_info = {'source': str(source), 'processed_frames': 0, 'detections': 0, 'events': 0}
            report['sources'].append(source_info)
            started = perf_counter()
            stream = frames(source, args.every_n_frames, args.max_frames)
            try:
                for frame, index, seconds, preview_fps in stream:
                    if not warm:
                        # Warm-up both models once, independent of whether the
                        # first input contains a plate. Exclude it from timing.
                        warm_started = perf_counter()
                        detector.predict(frame)
                        reader.read(np.full((32, 128, 3), 220, dtype=np.uint8), 'one')
                        if args.device == 'cuda':
                            reader.torch.cuda.synchronize()
                        warm = True
                        started += perf_counter() - warm_started
                    begin = perf_counter()
                    timestamp = round(seconds * 1000, 3)
                    try:
                        rows = recognize_frame(frame, detector, reader, str(source), index, timestamp, args.layout, not args.no_rectify)
                    except Exception as exc:
                        writers['frames'].writerow(dict(source=str(source), frame_index=index, timestamp_ms_approx=timestamp,
                            detections='', ocr_errors='', status='error', compute_ms=round((perf_counter() - begin) * 1000, 3), error=str(exc)))
                        raise
                    compute_ms = (perf_counter() - begin) * 1000
                    events = track.update(rows, seconds)
                    writers['detections'].writerows(rows)
                    writers['events'].writerows(events)
                    errors = sum(r['status'] != 'ok' for r in rows)
                    writers['frames'].writerow(dict(source=str(source), frame_index=index, timestamp_ms_approx=timestamp,
                        detections=len(rows), ocr_errors=errors, status='ocr_error' if errors else ('ok' if rows else 'no_plate'),
                        compute_ms=round(compute_ms, 3), error=''))
                    for key, count in [('processed_frames', 1), ('detections', len(rows)), ('events', len(events))]:
                        source_info[key] += count
                        report[key] += count
                    report['ocr_errors'] += errors
                    if not args.no_preview:
                        annotated = draw_results(frame, rows)
                        if preview_fps is None:
                            name = f'{source_index:03d}_{source.stem}.jpg'
                            Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)).save(destination / name, quality=93)
                        else:
                            if preview is None:
                                # Pad odd dimensions for codec compatibility.
                                output_size = (frame.shape[1] + frame.shape[1] % 2, frame.shape[0] + frame.shape[0] % 2)
                                preview = cv2.VideoWriter(str(destination / f'{source_index:03d}_preview.mp4'),
                                    cv2.VideoWriter_fourcc(*'mp4v'), preview_fps, output_size)
                                if not preview.isOpened():
                                    raise ValueError('Video preview encoder unavailable; retry with --no-preview and a fresh output folder')
                            annotated = cv2.copyMakeBorder(annotated, 0, annotated.shape[0] % 2, 0, annotated.shape[1] % 2, cv2.BORDER_REPLICATE)
                            preview.write(annotated)
                    if preview_fps is None or source_info['processed_frames'] % 30 == 1:
                        print(f'{source.name} frame={index}: ' + ('; '.join(r['text'] or '[unreadable]' for r in rows) or '[no plate detected]'), flush=True)
            finally:
                stream.close()
                if preview is not None:
                    preview.release()
                remaining = track.finish()
                writers['events'].writerows(remaining)
                source_info['events'] += len(remaining)
                report['events'] += len(remaining)
                source_info['stream_wall_seconds'] = round(perf_counter() - started, 4)
        report['status'] = 'complete'
    except KeyboardInterrupt:
        report['status'], exit_code = 'interrupted', 130
    except Exception as exc:
        report['status'], report['error'], exit_code = 'failed', f'{type(exc).__name__}: {exc}', 1
        print(report['error'], file=sys.stderr)
    finally:
        for handle in handles:
            handle.close()
        (destination / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"{report['status']}: {report['processed_frames']} frames, {report['detections']} detections, {report['events']} events. Files: {destination}")
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
