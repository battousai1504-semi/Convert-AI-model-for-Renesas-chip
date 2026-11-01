#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

vnPython="$PWD/.venv-linux/bin/python"
vnRun="${2:-artifacts/dxm1_full_encoder_mm_20261101_134233_21170}"
vnSource="${1:-}"
vnExpected=""
for vnFile in "$vnPython" alpr_auto.py alpr_auto_dxm1.py dxm1_parseq_backend.py \
    "$vnRun/matmul_contract.json"; do
    if [ ! -f "$vnFile" ]; then
        echo "STOP: Missing $vnFile"
        exit 1
    fi
done

if [ -z "$vnSource" ]; then
    vnSource="inputs/vn_dxm1_smoke/3.jpg"
    vnExpected="30A33918"
    python3 - "$vnSource" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import urlopen

target = Path(sys.argv[1])
url = "https://raw.githubusercontent.com/trungdinh22/License-Plate-Recognition/0b08d1ad9d3da5b0569243a77f068138678f516b/test_image/3.jpg"
expected = "8345ddb7c0124ad6e823f5ea77544af44bd4fe7636f358035ad53c10d2c1afbd"
if target.exists():
    data = target.read_bytes()
else:
    print("Downloading public VN smoke image:", url, flush=True)
    last_error = None
    for attempt in range(3):
        try:
            with urlopen(url, timeout=30) as response:
                data = response.read()
            break
        except Exception as error:
            last_error = error
    else:
        raise RuntimeError(f"Image download failed: {last_error}")
if hashlib.sha256(data).hexdigest() != expected:
    raise ValueError("Image checksum differs; existing data will not be overwritten")
target.parent.mkdir(parents=True, exist_ok=True)
if not target.exists():
    with target.open("xb") as handle:
        handle.write(data)
(target.parent / "sample_source.json").write_text(json.dumps({
    "url": url, "sha256": expected,
    "expected_text": "30A33918", "label_source": "visually checked on this single public image",
    "scope": "single-image smoke test, not a representative VN benchmark or redistribution license"
}, indent=2), encoding="utf-8")
print("VN input:", target.resolve(), flush=True)
PY
fi

vnOutput="results/vn_dxm1_smoke_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$vnOutput"
echo "CPU baseline output: $vnOutput/cpu"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$vnPython" alpr_auto.py \
    --source "$vnSource" --device cpu --max-frames 30 --output "$vnOutput/cpu"

echo "DX-M1 output: $vnOutput/dxm1"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$vnPython" alpr_auto_dxm1.py \
    --dx-run "$vnRun" --source "$vnSource" --device cpu \
    --max-frames 30 --output "$vnOutput/dxm1"

python3 - "$vnOutput" "$vnExpected" <<'PY'
import csv
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
expected = sys.argv[2]
def read(name):
    with (root / name / "detections.csv").open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))
cpu, npu = read("cpu"), read("dxm1")
fields = ["source", "frame_index", "x1", "y1", "x2", "y2", "text", "status"]
signature = lambda rows: Counter(tuple(row[key] for key in fields) for row in rows)
agreement = signature(cpu) == signature(npu)
backend = json.loads((root / "dxm1/summary.json").read_text(encoding="utf-8"))
if backend.get("ocr_encoder_backend") != "DX-M1":
    raise RuntimeError("DX-M1 backend provenance is missing from the report")
errors = [row for row in npu if row["status"] != "ok"]
summary = {
    "cpu_detection_texts": [row["text"] for row in cpu],
    "dxm1_detection_texts": [row["text"] for row in npu],
    "detection_text_and_box_agreement": agreement,
    "dxm1_ocr_errors": len(errors),
    "expected_single_image_text": expected or None,
    "expected_text_seen_cpu": any(row["text"] == expected and row["status"] == "ok" for row in cpu) if expected else None,
    "expected_text_seen_dxm1": any(row["text"] == expected and row["status"] == "ok" for row in npu) if expected else None,
    "successful_encoder_calls_including_warmup": backend["deepx_encoder"]["successful_encoder_calls"],
    "scope": "Small smoke test. Agreement is not general OCR accuracy; timestamps/FPS retain baseline limitations.",
}
(root / "smoke_comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("All files:", root.resolve())
if not agreement or errors or (expected and not summary["expected_text_seen_dxm1"]):
    raise SystemExit(2)
PY
