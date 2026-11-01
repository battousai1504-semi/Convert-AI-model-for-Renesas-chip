#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

vnSource="${ONEKIWI_ENCODER_ONNX:-artifacts/encoder_20261031_225454_044655/parseq_encoder.onnx}"
vnPython="$PWD/.venv-linux/bin/python"
vnImages="${1:-demo_images}"
for vnFile in "$vnSource" "$vnPython" probe_parseq_dxm1.py \
    probe_dxm1_matmul_head.py compare_dxm1_matmul_ocr.py; do
    if [ ! -f "$vnFile" ]; then
        echo "STOP: Missing $vnFile"
        exit 1
    fi
done
vnRun="artifacts/dxm1_full_encoder_mm_$(date +%Y%m%d_%H%M%S)_$$"
echo "FULL encoder run: $vnRun"

sudo docker run --rm -i \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$PWD,target=/work" \
  --workdir /work \
  --env PYTHONUNBUFFERED=1 \
  --env OMP_NUM_THREADS=4 \
  --env MKL_NUM_THREADS=4 \
  --env XDG_CACHE_HOME=/tmp/onekiwi-cache \
  --env MPLCONFIGDIR=/tmp/onekiwi-matplotlib \
  onekiwi/dxcom:2.4.1-cpu \
  python3 probe_dxm1_matmul_head.py compile \
    --source "$vnSource" \
    --input "$vnImages" \
    --output "$vnRun"

"$vnPython" compare_dxm1_matmul_ocr.py \
  --run "$vnRun" \
  --input "$vnImages"
