#!/usr/bin/env python3
"""Collect a compact, cross-checked DX-M1 smoke checkpoint; stdlib only.

Reads reports and CSVs; performs no inference, reset, update or driver operation.
Agreement is conversion fidelity, not labeled test-set accuracy.
"""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_RUN = 'artifacts/dxm1_full_encoder_mm_20261101_134233_21170'
DEFAULT_OCR = 'results/dxm1_full_ocr_20261101_134253_446327'
DEFAULT_PHOTO = 'results/vn_dxm1_smoke_20261101_135749_21703'
DEFAULT_VIDEO = 'results/vn_dxm1_smoke_20261101_140152_21831'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def collect(repo, run, ocr, photo, video):
    repo = Path(repo).resolve()
    run, ocr, photo, video = [repo / p for p in (run, ocr, photo, video)]
    sources = {}

    def record(path):
        sources[str(path.resolve().relative_to(repo))] = digest(path)

    manifest_path = run / 'matmul_contract.json'
    manifest = read_json(manifest_path)
    require(manifest['status'] == 'compiled', 'Encoder run is not compiled')
    model = (run / manifest['dxnn']).resolve()
    # Reject an external artifact reference before opening/hashing it.
    model.relative_to(repo)
    model_hash = digest(model)
    require(model_hash == manifest['dxnn_sha256'], 'Compiled model hash differs from manifest')
    placement = manifest['placement']
    require(placement.get('projection_nodes') and
            list(placement['partitions'].values()) == ['NPU'],
            'This milestone requires exactly one NPU partition and no CPU partition')
    for filename, key in [('original_reference.onnx', 'original_sha256'),
                          ('parseq_rawpatch_matmul.onnx', 'matmul_sha256')]:
        require(digest(run / filename) == manifest[key], f'{filename} hash differs')
    record(manifest_path)

    full = read_json(ocr / 'summary.json')
    full_rows = read_rows(ocr / 'comparison.csv')
    require(full['dxnn_sha256'] == model_hash, 'Full OCR used another model')
    require(full['original_sha256'] == manifest['original_sha256'], 'Full OCR reference differs')
    require(full['images'] == 6 and full['text_matches'] == 6 and full['errors'] == 0
            and full['all_texts_match'] is True and len(full_rows) == 6,
            'Expected six successful full OCR matches')
    require(all(row['status'] == 'match' and row['pytorch_text'] == row['dxm1_hybrid_text']
                for row in full_rows), 'Full OCR CSV contradicts its summary')
    record(ocr / 'summary.json')
    record(ocr / 'comparison.csv')
    observations = {}
    for name, root, count, text, calls in [('photo', photo, 1, '30A33918', 2),
                                         ('video', video, 30, '59V184625', 61)]:
        smoke = read_json(root / 'smoke_comparison.json')
        require(smoke['detection_text_and_box_agreement'] is True and
                smoke['dxm1_ocr_errors'] == 0, f'{name}: smoke comparison failed')
        require(smoke['successful_encoder_calls_including_warmup'] == calls,
                f'{name}: unexpected encoder call count')
        paired_rows = []
        reports = []
        for backend in ('cpu', 'dxm1'):
            folder = root / backend
            report = read_json(folder / 'summary.json')
            rows = read_rows(folder / 'detections.csv')
            events = read_rows(folder / 'events.csv')
            frames = read_rows(folder / 'frames.csv')
            require(report['status'] == 'complete' and report['processed_frames'] == count
                    and report['detections'] == count and report['events'] == 1
                    and report['ocr_errors'] == 0, f'{name}/{backend}: unexpected report counts')
            require(len(rows) == count and len(events) == 1 and len(frames) == count,
                    f'{name}/{backend}: CSV counts differ from summary')
            require(all(row['status'] == 'ok' and row['text'] == text for row in rows),
                    f'{name}/{backend}: unexpected or failed plate read')
            require(Counter(smoke[f'{backend}_detection_texts']) == Counter(row['text'] for row in rows),
                    f'{name}/{backend}: smoke text list differs from CSV')
            if backend == 'dxm1':
                require(report['ocr_encoder_backend'] == 'DX-M1' and
                        report['ocr_decoder_device'] == 'cpu', f'{name}: wrong backend')
                dx = report['deepx_encoder']
                require(dx['dxnn_sha256'] == model_hash and
                        dx['original_encoder_sha256'] == manifest['original_sha256'],
                        f'{name}: encoder model/reference hash differs')
                require(dx['successful_encoder_calls'] == calls and dx['failed_encoder_calls'] == 0,
                        f'{name}: encoder counters differ')
            reports.append(report)
            fields = ('source', 'frame_index', 'x1', 'y1', 'x2', 'y2', 'text', 'status')
            paired_rows.append(Counter(tuple(row[key] for key in fields) for row in rows))
            for filename in ('summary.json', 'detections.csv', 'events.csv', 'frames.csv'):
                record(folder / filename)
        require(paired_rows[0] == paired_rows[1], f'{name}: CPU/DX-M1 boxes/text differ')
        if name == 'photo':
            require(smoke['expected_single_image_text'] == text and
                    smoke['expected_text_seen_cpu'] is True and smoke['expected_text_seen_dxm1'] is True,
                    'Photo expected-text check is missing')
        record(root / 'smoke_comparison.json')
        observations[name] = {
            'processed_frames': count, 'detections_each_backend': count, 'events_each_backend': 1,
            'text_counts_each_backend': {text: count}, 'box_and_text_agreement': True,
            'dxm1_ocr_errors': 0, 'successful_encoder_calls_including_warmup': calls,
            'sampling': 'one full-scene image' if name == 'photo' else 'sampled video frames, not independent plates',
        }
    return {
        'schema_version': 1, 'collected_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Cross-checked small smoke checkpoint, not a representative accuracy or speed benchmark',
        'dxnn_sha256': model_hash, 'original_encoder_sha256': manifest['original_sha256'],
        'placement': placement,
        'architecture': {'detector': 'ORT CPU', 'preprocessing': 'CPU',
                         'encoder': 'DX-M1', 'decoder': 'PyTorch CPU', 'tracking_csv': 'CPU'},
        'full_ocr': {key: full[key] for key in ('images', 'text_matches', 'errors', 'all_texts_match')},
        'ocr_runtime_versions': {key: full.get(key) for key in ('torch', 'numpy', 'onnxruntime', 'dx_engine')},
        'observations': observations, 'source_report_sha256': sources,
        'ground_truth_test_accuracy': None, 'end_to_end_fps': None,
        'limitations': ['six demo crops were also calibration inputs',
                        'one VN photo and one short sampled VN video',
                        'event count is not a unique-vehicle benchmark',
                        'decoder and detector remain on CPU; Renesas unverified'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    for name, default in [('run', DEFAULT_RUN), ('ocr', DEFAULT_OCR), ('photo', DEFAULT_PHOTO), ('video', DEFAULT_VIDEO)]:
        parser.add_argument('--' + name, default=default)
    parser.add_argument('--output', default='docs/evidence/dxm1_20261101.json')
    args = parser.parse_args()
    repo = args.repo.resolve()
    result = collect(repo, args.run, args.ocr, args.photo, args.video)
    result['collection_host'] = {'kernel': platform.release(), 'architecture': platform.machine()}
    for key, command in [('git_head_before_checkpoint', ['git', 'rev-parse', 'HEAD']),
                         ('git_branch', ['git', 'branch', '--show-current']),
                         ('dxrt_status_at_collection', ['dxrt-cli', '-s'])]:
        completed = subprocess.run(command, cwd=repo, text=True, capture_output=True, check=True, timeout=20)
        result[key] = completed.stdout.strip()
    destination = repo / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create avoids silently replacing an older proof file.
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print('Cross-checked evidence:', destination)


if __name__ == '__main__':
    main()
