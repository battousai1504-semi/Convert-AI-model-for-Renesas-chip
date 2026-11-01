#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

vnSource="artifacts/dxm1_embedding_probe_20261101_131111_19767/encoder_normalized_reference.onnx"
vnPython="$PWD/.venv-linux/bin/python"
for vnFile in "$vnSource" "$vnPython" probe_parseq_dxm1.py probe_dxm1_matmul_head.py; do
    if [ ! -f "$vnFile" ]; then
        echo "STOP: Missing $vnFile"
        exit 1
    fi
done
vnRun="artifacts/dxm1_matmul_head_probe_$(date +%Y%m%d_%H%M%S)_$$"
echo "Raw-patch MatMul probe: $vnRun"

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
    --input demo_images \
    --output "$vnRun"

"$vnPython" probe_dxm1_matmul_head.py compare \
  --run "$vnRun" \
  --input demo_images
