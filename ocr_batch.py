#!/usr/bin/env python3
r"""PARSeq cropped-text baseline, v1.0 (Python 3.10+).

Place this file beside read.py in the baudm/parseq checkout. No new packages
are required beyond the environment already used successfully with read.py.

PowerShell examples:
  .\.venv\Scripts\python.exe ocr_batch.py --input inputs --device both
  .\.venv\Scripts\python.exe ocr_batch.py --input inputs --device both --labels labels.csv

Optional labels.csv (UTF-8; image path relative to --input directory):
  image,text
  plate_01.png,THE_VERIFIED_TEXT

Scope: one already-cropped text line per image, batch size 1. This script
does NOT locate plates, split two-line plates, process video, or use DEEPX.
Do not put whole-car images or unsplit two-line plates in this baseline set.
Raw predictions are retained. Normalization only uppercases and removes
non-ASCII-alphanumeric separators; it never substitutes lookalike characters.
No ground-truth text is inferred from an image name or a model prediction.

Each sample is measured once after warm-up. model_ms measures model(tensor)
only. pipeline_ms includes opening/decoding the image, preprocessing, transfer
to the device, model execution, softmax and tokenizer decoding. CUDA is
synchronized at stage boundaries. Model loading, warm-up, console output and
CSV writes are excluded. These are single-image crop timings, NOT video FPS.
Small datasets are suitable for smoke tests, not production accuracy claims.
"""

import argparse
import csv
import gc
import json
import math
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter

VERSION = "1.0"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
FIELDS = [
    "image", "device", "prediction", "normalized_prediction", "expected_text",
    "normalized_expected", "exact_match", "edit_distance", "preprocess_ms",
    "model_ms", "decode_ms", "pipeline_ms", "status", "error",
]


def normalize(text):
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def image_key(value):
    return str(value).replace("\\", "/").removeprefix("./")


def load_labels(path):
    labels = {}
    if path is None:
        return labels
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"image", "text"}.issubset(reader.fieldnames or []):
            raise ValueError("labels.csv must have image,text columns")
        for line, item in enumerate(reader, 2):
            key = image_key(item.get("image") or "").strip()
            text = (item.get("text") or "").strip()
            if not key or not normalize(text):
                raise ValueError(f"Invalid or empty label at CSV line {line}")
            if key in labels:
                raise ValueError(f"Duplicate label: {key}")
            labels[key] = text
    return labels


def discover(source):
    source = Path(source).resolve()
    if source.is_file():
        images = [source] if source.suffix.lower() in EXTENSIONS else []
        root = source.parent
    elif source.is_dir():
        root = source
        images = sorted(p for p in source.rglob("*")
                        if p.is_file() and p.suffix.lower() in EXTENSIONS)
    else:
        raise ValueError(f"Input does not exist: {source}")
    if not images:
        raise ValueError(f"No supported images found: {source}")
    return root, images


def make_row(key, device, expected=None, prediction="", timings=None, error=""):
    row = dict.fromkeys(FIELDS, "")
    row.update(image=key, device=device, prediction=prediction,
               normalized_prediction=normalize(prediction),
               status="error" if error else "ok", error=error)
    if timings:
        row.update({name: round(value, 4) for name, value in timings.items()})
    if expected is not None:
        target = normalize(expected)
        row.update(expected_text=expected, normalized_expected=target,
                   exact_match=int(not error and normalize(prediction) == target),
                   edit_distance=edit_distance(normalize(prediction), target))
    return row


def describe(values):
    values = [value for value in values if isinstance(value, (int, float))]
    if not values:
        return {"mean": None, "median": None, "p95_nearest_rank": None}
    ordered = sorted(values)
    return {"mean": round(mean(values), 4), "median": round(median(values), 4),
            "p95_nearest_rank": round(ordered[math.ceil(.95 * len(ordered)) - 1], 4)}


def summarize(rows):
    success = [r for r in rows if r["status"] == "ok"]
    labeled = [r for r in rows if r["expected_text"] != ""]
    characters = sum(len(r["normalized_expected"]) for r in labeled)
    return {
        "images": len(rows), "successful": len(success),
        "errors": len(rows) - len(success), "labeled_images": len(labeled),
        "exact_plate_accuracy": (sum(r["exact_match"] for r in labeled) / len(labeled)
                                 if labeled else None),
        "character_error_rate": (sum(r["edit_distance"] for r in labeled) / characters
                                 if characters else None),
        "model_ms": describe([r["model_ms"] for r in success]),
        "pipeline_ms": describe([r["pipeline_ms"] for r in success]),
        "metric_policy": "Uppercase ASCII alphanumerics; labeled read failures count as errors.",
    }


def infer_one(path, model, transform, device, torch):
    from PIL import Image, ImageOps

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    sync()
    start = perf_counter()
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
    sync()
    prepared = perf_counter()
    logits = model(tensor)
    sync()
    inferred = perf_counter()
    predictions, _ = model.tokenizer.decode(logits.softmax(-1))
    sync()
    decoded = perf_counter()
    return predictions[0], {
        "preprocess_ms": (prepared - start) * 1000,
        "model_ms": (inferred - prepared) * 1000,
        "decode_ms": (decoded - inferred) * 1000,
        "pipeline_ms": (decoded - start) * 1000,
    }


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="inputs", help="Image file or cropped-image folder")
    parser.add_argument("--device", choices=["cpu", "cuda", "both"], default="both")
    parser.add_argument("--output", help="Output folder (default: new timestamped folder in results)")
    parser.add_argument("--labels", help="Optional verified image,text CSV")
    parser.add_argument("--checkpoint", default="pretrained=parseq")
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    try:
        root, images = discover(args.input)
        labels = load_labels(args.labels)
        keys = {image_key(p.relative_to(root)) for p in images}
        if labels:
            missing, extra = keys - labels.keys(), labels.keys() - keys
            if missing or extra:
                raise ValueError(f"Label/image mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    import torch
    from strhub.data.module import SceneTextDataModule
    from strhub.models.utils import load_from_checkpoint

    devices = ["cpu", "cuda"] if args.device == "both" else [args.device]
    if "cuda" in devices and not torch.cuda.is_available():
        parser.error("CUDA unavailable. Use --device cpu, or fix the PyTorch/driver environment.")
    destination = Path(args.output or (Path("results") /
                       datetime.now().strftime("baseline_%Y%m%d_%H%M%S_%f"))).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {destination}")
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
                     cwd=Path(__file__).resolve().parent, stderr=subprocess.DEVNULL,
                     text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    report = {
        "script_version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint, "repo_commit": revision, "input_root": str(root),
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "cpu_threads": torch.get_num_threads(),
        "warmup_passes_per_device": args.warmup, "batch_size": 1,
        "scope": "One cropped text line per image. No detection, row splitting, video or DEEPX.",
        "timing": "Single pass per image after warm-up; synchronized CUDA stages; model loading and output writes excluded.",
        "results": {},
    }
    if "cuda" in devices:
        report["gpu"] = torch.cuda.get_device_name(0)
    all_rows = {}
    failed = False
    with torch.inference_mode():
        for device in devices:
            print(f"\nLoading {args.checkpoint} on {device} ...")
            model = load_from_checkpoint(args.checkpoint).eval().to(device)
            transform = SceneTextDataModule.get_transform(model.hparams.img_size)
            # Warm up on the first usable image, without dropping failed inputs from the report.
            if args.warmup:
                for candidate in images:
                    try:
                        for _ in range(args.warmup):
                            infer_one(candidate, model, transform, device, torch)
                        break
                    except (OSError, ValueError):
                        continue
            rows = []
            for index, path in enumerate(images, 1):
                key = image_key(path.relative_to(root))
                try:
                    text, timings = infer_one(path, model, transform, device, torch)
                    row = make_row(key, device, labels.get(key), text, timings)
                    print(f"[{device} {index}/{len(images)}] {key}: {text} | {timings['pipeline_ms']:.2f} ms")
                except Exception as exc:
                    if device == "cuda" and isinstance(exc, RuntimeError):
                        # A CUDA failure may poison the context; report it rather than hiding a fallback.
                        raise RuntimeError(f"CUDA inference failed for {key}: {exc}") from exc
                    row = make_row(key, device, labels.get(key), error=f"{type(exc).__name__}: {exc}")
                    failed = True
                    print(f"[{device} {index}/{len(images)}] {key}: ERROR {row['error']}")
                rows.append(row)
            write_csv(destination / f"ocr_{device}.csv", rows)
            report["results"][device] = summarize(rows)
            all_rows[device] = rows
            print(json.dumps(report["results"][device], ensure_ascii=False, indent=2))
            del model, transform
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    if len(devices) == 2:
        paired = [(a, b) for a, b in zip(all_rows["cpu"], all_rows["cuda"])
                  if a["status"] == b["status"] == "ok"]
        report["cpu_gpu_comparison"] = {
            "paired_successful_images": len(paired),
            "raw_prediction_disagreements": [a["image"] for a, b in paired
                                               if a["prediction"] != b["prediction"]],
            "note": "CPU/GPU agreement is not ground-truth accuracy.",
        }
    (destination / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(f"\nSaved CSV and summary.json to: {destination}")
    if not labels:
        print("No verified labels supplied: accuracy and CER are null, not estimated.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
