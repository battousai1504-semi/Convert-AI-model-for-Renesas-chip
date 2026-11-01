#!/usr/bin/env python3
"""Download the pinned five-image VN development starter and prepare labels."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from PIL import Image, ImageDraw, ImageOps

from evaluate_alpr_dataset import load_dataset, normalize, require


def download_verified(url, expected, target):
    if target.exists():
        data = target.read_bytes()
    else:
        for attempt in range(3):
            try:
                with urlopen(url, timeout=30) as response:
                    data = response.read()
                break
            except OSError:
                if attempt == 2:
                    raise
    require(hashlib.sha256(data).hexdigest() == expected, f'Checksum differs; not overwriting: {target}')
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
    return data


def prepare(recipe_path, destination, calibration_manifest=None):
    recipe_path, destination = Path(recipe_path), Path(destination)
    recipe = json.loads(recipe_path.read_text(encoding='utf-8'))
    if calibration_manifest is not None:
        calibration = json.loads(Path(calibration_manifest).read_text(encoding='utf-8'))['calibration']
        require(calibration and all(row.get('sha256') for row in calibration), 'Calibration hashes are missing')
        hashes = {row['sha256'] for row in calibration}
        require(
            not any(item['sha256'] in hashes for item in recipe['images']),
            'Starter image overlaps the encoder calibration data',
        )
        print('Exact-hash overlap with this encoder calibration: none', flush=True)
    dataset = dict(recipe, recipe_sha256=hashlib.sha256(recipe_path.read_bytes()).hexdigest(), images=[])
    for item in recipe['images']:
        require(Path(item['file']).name == item['file'], 'Recipe filename must be a basename')
        target = destination / 'images' / item['file']
        print('Prepare:', item['id'], flush=True)
        download_verified(item['url'], item['sha256'], target)
        label = item['annotation_provenance']
        download_verified(
            label['box_label_url'], label['box_label_sha256'], destination / 'upstream_labels' / (item['id'] + '.txt')
        )
        with Image.open(target) as original:
            image = ImageOps.exif_transpose(original)
            require(image.size == (item['width'], item['height']), 'Unexpected image dimensions')
        dataset['images'].append(dict(item, file='images/' + item['file']))
    manifest = destination / 'dataset.json'
    serialized = json.dumps(dataset, ensure_ascii=False, indent=2) + '\n'
    if manifest.exists():
        require(
            manifest.read_text(encoding='utf-8') == serialized,
            'Dataset manifest differs; use a new output folder or explicitly review your label changes',
        )
    else:
        with manifest.open('x', encoding='utf-8') as stream:
            stream.write(serialized)
    load_dataset(manifest)
    sheet = Image.new('RGB', (1280, 320 * ((len(dataset['images']) + 1) // 2)), '#dddddd')
    for index, item in enumerate(dataset['images']):
        with Image.open(destination / item['file']) as original:
            original = ImageOps.exif_transpose(original).convert('RGB')
        cell = Image.new('RGB', (640, 320), 'white')
        preview = original.copy()
        preview.thumbnail((300, 280))
        cell.paste(preview, (0, 35))
        plate = item['annotations'][0]
        crop = original.crop(tuple(round(v) for v in plate['box_xyxy']))
        crop.thumbnail((320, 240))
        cell.paste(crop, (310, 55))
        ImageDraw.Draw(cell).text((5, 5), item['id'] + '  GT: ' + normalize(plate['text']), fill='black')
        sheet.paste(cell, ((index % 2) * 640, (index // 2) * 320))
    sheet.save(destination / 'label_review.jpg')
    print('Dataset:', manifest.resolve())
    print('Label review:', (destination / 'label_review.jpg').resolve())
    print('5 public development images. This is not a representative held-out test set.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', default=str(Path(__file__).resolve().parent / 'evaluation/vn_starter_v1.json'))
    parser.add_argument('--output', default='inputs/vn_eval_v1')
    parser.add_argument(
        '--calibration-manifest', help='Optional matmul_contract.json for exact image-hash overlap check'
    )
    args = parser.parse_args(argv)
    prepare(args.recipe, args.output, args.calibration_manifest)


if __name__ == '__main__':
    main()
