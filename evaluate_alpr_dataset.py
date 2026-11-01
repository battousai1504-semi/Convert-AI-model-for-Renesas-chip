#!/usr/bin/env python3
"""Evaluate automatic ALPR image CSVs against independent full-plate labels.

No model inference runs here. Matching uses geometry only, never plate text.
"""
import argparse
import csv
import hashlib
import html
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def normalize(value):
    return ''.join(c for c in str(value).upper() if c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, char in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char != other)))
        previous = current
    return previous[-1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def valid_box(values, width, height):
    require(isinstance(values, (list, tuple)) and len(values) == 4, 'Expected four box coordinates')
    box = tuple(float(value) for value in values)
    require(all(math.isfinite(value) for value in box), 'Non-finite box coordinate')
    require(0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height, 'Invalid or out-of-image box')
    return box


def load_dataset(path):
    from PIL import Image, ImageOps

    path = Path(path).resolve()
    dataset = read_json(path)
    require(
        dataset.get('schema_version') == 1 and isinstance(dataset.get('images'), list) and dataset['images'],
        'Expected a nonempty schema_version=1 image dataset',
    )
    ids, files, digests = set(), set(), set()
    for image in dataset['images']:
        identity = image['id']
        require(isinstance(identity, str) and re.fullmatch(r'[A-Za-z0-9_-]+', identity), 'Invalid image ID')
        require(identity not in ids, 'Duplicate image ID')
        ids.add(identity)
        name = Path(image['file'])
        require(not name.is_absolute() and '..' not in name.parts, 'Use relative paths inside the dataset')
        source = (path.parent / name).resolve()
        source.relative_to(path.parent)
        require(source not in files, 'Duplicate image file')
        files.add(source)
        require(source.is_file() and sha256(source) == image['sha256'], f'Image hash mismatch: {identity}')
        require(image['sha256'] not in digests, 'Exact duplicate images cannot count as independent samples')
        digests.add(image['sha256'])
        width, height = image['width'], image['height']
        require(
            isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0,
            'Expected positive integer image dimensions',
        )
        with Image.open(source) as original:
            require(ImageOps.exif_transpose(original).size == (width, height), 'Image dimensions differ from manifest')
        require(image.get('split') in ('development', 'regression', 'test'), 'Declare each image split')
        require(isinstance(image.get('group_id'), str) and image['group_id'], 'Missing source group ID')
        require(isinstance(image.get('annotations'), list), 'Declare annotations, including [] for a negative image')
        plate_ids = set()
        for plate in image['annotations']:
            require(
                isinstance(plate['id'], str) and plate['id'] and plate['id'] not in plate_ids,
                'Duplicate or missing plate ID within an image',
            )
            plate_ids.add(plate['id'])
            require(
                isinstance(plate['text'], str) and plate['text'].isascii() and normalize(plate['text']),
                'Ground truth must contain nonempty ASCII plate text',
            )
            require(plate.get('layout') in ('one', 'two'), 'Expected ground-truth layout one/two')
            valid_box(plate['box_xyxy'], width, height)
    # A group (video/capture session) cannot be scattered over declared splits.
    group_splits = {}
    for image in dataset['images']:
        group_splits.setdefault(image['group_id'], set()).add(image['split'])
    require(all(len(splits) == 1 for splits in group_splits.values()), 'Source group leaks across splits')
    return dataset


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def match_boxes(truth, predictions, threshold=0.5):
    """Maximum-cardinality geometry-only matching; descending IoU neighbors, stable ties.

    Does not claim globally maximum summed IoU or implement COCO AP matching.
    """
    require(math.isfinite(threshold) and 0 < threshold <= 1, 'IoU threshold must be in (0,1]')
    neighbors = []
    for box in truth:
        choices = [(iou(box, pred), index) for index, pred in enumerate(predictions)]
        neighbors.append(
            [index for overlap, index in sorted(choices, key=lambda x: (-x[0], x[1])) if overlap >= threshold]
        )
    owner = {}

    def augment(gt, visited):
        for pred in neighbors[gt]:
            if pred in visited:
                continue
            visited.add(pred)
            if pred not in owner or augment(owner[pred], visited):
                owner[pred] = gt
                return True
        return False

    for gt in range(len(truth)):
        augment(gt, set())
    return {gt: pred for pred, gt in owner.items()}


def load_run(folder, dataset, dataset_path, expected_backend):
    folder, dataset_path = Path(folder).resolve(), Path(dataset_path).resolve()
    report = read_json(folder / 'summary.json')
    require(report['status'] == 'complete', f'{expected_backend}: pipeline run did not finish')
    if expected_backend == 'dxm1':
        require(
            report.get('ocr_encoder_backend') == 'DX-M1' and report.get('ocr_decoder_device') == 'cpu',
            'DX-M1 encoder + CPU decoder provenance missing',
        )
        require(report.get('deepx_encoder', {}).get('dxnn_sha256'), 'Missing DXNN hash')
    else:
        require(
            report.get('ocr_device') == 'cpu' and report.get('ocr_encoder_backend') != 'DX-M1',
            'Expected original PyTorch CPU baseline',
        )
    paths = {(dataset_path.parent / image['file']).resolve(): image for image in dataset['images']}
    frames, detections = {}, {image['id']: [] for image in dataset['images']}
    for row in read_csv(folder / 'frames.csv'):
        source = Path(row['source']).resolve()
        require(source in paths and int(row['frame_index']) == 0, 'Unexpected image/frame in frames.csv')
        identity = paths[source]['id']
        require(identity not in frames, 'Duplicate frame row')
        elapsed = float(row['compute_ms'])
        require(math.isfinite(elapsed) and elapsed >= 0, 'Invalid frame compute time')
        frames[identity] = row
    require(set(frames) == set(detections), 'Missing frame rows: incomplete evaluation cannot be scored')
    for row in read_csv(folder / 'detections.csv'):
        source = Path(row['source']).resolve()
        require(source in paths and int(row['frame_index']) == 0, 'Unexpected image/frame in detections.csv')
        image = paths[source]
        require(row['status'] in ('ok', 'error'), 'Unexpected detection status')
        box = valid_box([row[key] for key in ('x1', 'y1', 'x2', 'y2')], image['width'], image['height'])
        detections[image['id']].append(dict(row, box=box))
    require(
        report['processed_frames'] == len(frames) and report['detections'] == sum(map(len, detections.values())),
        'Pipeline summary contradicts CSV counts',
    )
    for identity, frame in frames.items():
        require(frame['status'] in ('ok', 'no_plate', 'ocr_error', 'error'), 'Unknown frame status')
        if frame['status'] != 'error':
            require(int(frame['detections']) == len(detections[identity]), 'Frame detection count contradicts CSV')
        else:
            require(not detections[identity], 'Failed frame has unexpected predictions')
    return {
        'report': report,
        'frames': frames,
        'detections': detections,
        'source_hashes': {name: sha256(folder / name) for name in ('summary.json', 'frames.csv', 'detections.csv')},
    }


def score_images(images, predictions, threshold):
    rows = []
    for image in images:
        truth = image['annotations']
        predicted = predictions[image['id']]
        matching = match_boxes([p['box_xyxy'] for p in truth], [p['box'] for p in predicted], threshold)
        for gt, plate in enumerate(truth):
            pred = predicted[matching[gt]] if gt in matching else None
            expected = normalize(plate['text'])
            observed = normalize(pred.get('text', '')) if pred and pred['status'] == 'ok' else ''
            category = (
                'detector_miss'
                if pred is None
                else 'ocr_error'
                if pred['status'] != 'ok'
                else 'correct'
                if observed == expected
                else 'ocr_wrong'
            )
            rows.append({
                'image_id': image['id'],
                'plate_id': plate['id'],
                'split': image['split'],
                'group_id': image['group_id'],
                'layout_gt': plate['layout'],
                'expected_text': expected,
                'prediction': observed,
                'raw_prediction': pred.get('text', '') if pred else '',
                'category': category,
                'detected': int(pred is not None),
                'exact': int(category == 'correct'),
                'edit_distance': edit_distance(expected, observed),
                'iou': iou(plate['box_xyxy'], pred['box']) if pred else 0.0,
                'prediction_index': matching.get(gt, ''),
                'gt_box': plate['box_xyxy'],
                'pred_box': pred['box'] if pred else [],
                'error': pred.get('error', '') if pred else '',
            })
        for index, pred in enumerate(predicted):
            if index not in matching.values():
                rows.append({
                    'image_id': image['id'],
                    'plate_id': '',
                    'split': image['split'],
                    'group_id': image['group_id'],
                    'layout_gt': '',
                    'expected_text': '',
                    'prediction': normalize(pred.get('text', '')) if pred['status'] == 'ok' else '',
                    'raw_prediction': pred.get('text', ''),
                    'category': 'false_positive',
                    'detected': 0,
                    'exact': 0,
                    'edit_distance': '',
                    'iou': 0.0,
                    'prediction_index': index,
                    'gt_box': [],
                    'pred_box': pred['box'],
                    'error': pred.get('error', ''),
                })
    return rows


def metrics(rows, images):
    gt = [row for row in rows if row['plate_id']]
    matched = [row for row in gt if row['detected']]
    correct = sum(row['exact'] for row in gt)
    fp = sum(row['category'] == 'false_positive' for row in rows)
    count, total = len(matched), len(matched) + fp
    characters = sum(len(row['expected_text']) for row in gt)
    matched_chars = sum(len(row['expected_text']) for row in matched)
    return {
        'images': len(images),
        'source_groups': len({image['group_id'] for image in images}),
        'ground_truth_plates': len(gt),
        'predicted_boxes': total,
        'detection_tp': count,
        'detection_fp': fp,
        'detection_fn': len(gt) - count,
        'detection_precision': count / total if total else None,
        'detection_recall': count / len(gt) if gt else None,
        'exact_plates': correct,
        'end_to_end_exact_recall': correct / len(gt) if gt else None,
        'end_to_end_exact_precision': correct / total if total else None,
        'ocr_exact_on_matched_boxes': correct / count if count else None,
        'ocr_cer_on_matched_boxes': sum(row['edit_distance'] for row in matched) / matched_chars
        if matched_chars
        else None,
        'gt_cer_including_missed_plates': sum(row['edit_distance'] for row in gt) / characters if characters else None,
        'error_counts': dict(Counter(row['category'] for row in rows)),
    }


def summarize(dataset, run, threshold):
    rows = score_images(dataset['images'], run['detections'], threshold)
    times = sorted(float(frame['compute_ms']) for frame in run['frames'].values())
    return rows, {
        'overall': metrics(rows, dataset['images']),
        'by_split': {
            split: metrics(
                [row for row in rows if row['split'] == split], [im for im in dataset['images'] if im['split'] == split]
            )
            for split in sorted({im['split'] for im in dataset['images']})
        },
        'gt_by_layout': {
            layout: {
                'plates': sum(row['layout_gt'] == layout for row in rows),
                'correct': sum(row['exact'] for row in rows if row['layout_gt'] == layout),
            }
            for layout in ('one', 'two')
        },
        'frame_errors': sum(frame['status'] == 'error' for frame in run['frames'].values()),
        'frame_compute_ms': {
            'n': len(times),
            'mean': sum(times) / len(times),
            'p50_nearest_rank': times[math.ceil(0.5 * len(times)) - 1],
            'p95_nearest_rank': times[math.ceil(0.95 * len(times)) - 1],
            'note': 'Existing per-frame compute only; excludes decode/I/O/load/warmup. Not end-to-end FPS.',
        },
        'source_report_sha256': run['source_hashes'],
    }


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['image_id', 'category'])
        writer.writeheader()
        writer.writerows(rows)


def write_review(destination, dataset, dataset_path, runs):
    """Annotated image evidence; ground truth is green, predictions are red."""
    from PIL import Image, ImageDraw, ImageOps

    review = destination / 'review'
    review.mkdir()
    sections = [
        '<!doctype html><meta charset="utf-8"><title>ALPR evaluation</title>',
        '<h1>Ground truth (green), predictions (red)</h1>',
        '<p>Review labels independently. Error categories and exact texts are in CSV.</p>',
    ]
    for image in dataset['images']:
        sections.append('<h2>' + html.escape(image['id']) + '</h2>')
        for backend, run in runs.items():
            with Image.open(Path(dataset_path).parent / image['file']) as original:
                canvas = ImageOps.exif_transpose(original).convert('RGB')
            draw = ImageDraw.Draw(canvas)
            for plate in image['annotations']:
                box = plate['box_xyxy']
                draw.rectangle(box, outline='lime', width=3)
                draw.text((box[0], max(0, box[1] - 13)), 'GT: ' + normalize(plate['text']), fill='lime')
            for row in run['detections'][image['id']]:
                draw.rectangle(row['box'], outline='red', width=2)
                draw.text((row['box'][0], row['box'][3] + 2), 'P: ' + normalize(row.get('text', '')), fill='red')
            name = image['id'] + '_' + backend + '.jpg'
            canvas.save(review / name)
            caption = html.escape(backend + ': ' + ', '.join(p.get('text', '') for p in run['detections'][image['id']]))
            sections.append(f'<figure><img width="480" src="review/{name}"><figcaption>{caption}</figcaption></figure>')
    (destination / 'review.html').write_text('\n'.join(sections), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--cpu-run', required=True)
    parser.add_argument('--dxm1-run', required=True, help='ALPR output folder, NOT compiled encoder artifacts')
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    require(math.isfinite(args.iou) and 0 < args.iou <= 1, 'IoU threshold must be in (0,1]')
    dataset_path = Path(args.dataset).resolve()
    dataset = load_dataset(dataset_path)
    runs = {
        name: load_run(folder, dataset, dataset_path, name)
        for name, folder in [('cpu', args.cpu_run), ('dxm1', args.dxm1_run)]
    }
    require(
        runs['cpu']['report']['detector_sha256'] == runs['dxm1']['report']['detector_sha256'],
        'CPU and DX-M1 runs used different detectors',
    )
    require(
        runs['cpu']['report']['checkpoint'] == runs['dxm1']['report']['checkpoint'], 'CPU and DX-M1 checkpoints differ'
    )
    for key in ('confidence', 'layout', 'no_rectify'):
        require(
            runs['cpu']['report']['arguments'][key] == runs['dxm1']['report']['arguments'][key],
            f'CPU and DX-M1 preprocessing/detection setting differs: {key}',
        )
    destination = Path(args.output or 'results/vn_eval_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_name': dataset['name'],
        'dataset_sha256': sha256(dataset_path),
        'evaluation_script_sha256': sha256(__file__),
        'dataset_scope': dataset['scope'],
        'label_policy': dataset.get('label_policy'),
        'iou_threshold': args.iou,
        'metric_policy': 'Geometry-only maximum-cardinality matching at IoU threshold. ASCII uppercase alphanumerics; no lookalike correction. Failed OCR treated as empty text. No mAP or event/tracking metric.',
        'dxnn_sha256': runs['dxm1']['report']['deepx_encoder']['dxnn_sha256'],
        'backends': {},
    }
    gt_rows = {}
    for backend, run in runs.items():
        rows, summary = summarize(dataset, run, args.iou)
        gt_rows[backend] = {(row['image_id'], row['plate_id']): row for row in rows if row['plate_id']}
        report['backends'][backend] = summary
        write_csv(destination / (backend + '_plates.csv'), rows)
        write_csv(destination / (backend + '_errors.csv'), [row for row in rows if row['category'] != 'correct'])
        score = summary['overall']
        print(
            f"{backend}: correct={score['exact_plates']}/{score['ground_truth_plates']}; "
            f"det TP/FP/FN={score['detection_tp']}/{score['detection_fp']}/{score['detection_fn']}"
        )
    paired = []
    for key in gt_rows['cpu']:
        a, b = gt_rows['cpu'][key], gt_rows['dxm1'][key]
        paired.append({
            'image_id': key[0],
            'plate_id': key[1],
            'expected_text': a['expected_text'],
            'cpu_text': a['prediction'],
            'dxm1_text': b['prediction'],
            'cpu_category': a['category'],
            'dxm1_category': b['category'],
            'both_detected': int(a['detected'] and b['detected']),
            'same_nonempty_successful_text': int(
                a['category'] not in ('detector_miss', 'ocr_error')
                and b['category'] not in ('detector_miss', 'ocr_error')
                and bool(a['prediction'])
                and a['prediction'] == b['prediction']
            ),
        })
    write_csv(destination / 'cpu_dxm1_comparison.csv', paired)
    paired_count = sum(row['both_detected'] for row in paired)
    report['conversion_comparison'] = {
        'paired_geometric_matches': paired_count,
        'same_nonempty_successful_text': sum(row['same_nonempty_successful_text'] for row in paired),
        'note': 'Agreement is not ground-truth accuracy; missing/error predictions never count as agreement.',
    }
    write_review(destination, dataset, dataset_path, runs)
    (destination / 'summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8'
    )
    print('REPORT:', destination)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
