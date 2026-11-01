#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

vnAudit="${1:-results/dxm1_io_audit_20261101_125834_387917}"
vnPython="$PWD/.venv-linux/bin/python"
for vnFile in "$vnPython" probe_parseq_dxm1.py audit_dxm1_io.py \
    artifacts/dxm1_rgb_contract_v1/encoder_normalized_reference.onnx; do
    if [ ! -f "$vnFile" ]; then
        echo "STOP: Missing $vnFile"
        exit 1
    fi
done

if "$vnPython" audit_dxm1_layout.py --audit "$vnAudit"; then
    :
else
    vnStatus=$?
    if [ "$vnStatus" -eq 10 ]; then
        echo "Send the layout output before another compilation. No runtime data was changed."
        exit 0
    fi
    exit "$vnStatus"
fi

vnRun="artifacts/dxm1_embedding_probe_$(date +%Y%m%d_%H%M%S)_$$"
echo "Embedding-prefix run: $vnRun"

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
  python3 probe_dxm1_embedding.py \
    --source artifacts/dxm1_rgb_contract_v1/encoder_normalized_reference.onnx \
    --input demo_images \
    --output "$vnRun"

"$vnPython" audit_dxm1_io.py \
  --model "$vnRun/compiled/parseq_encoder_rgb.dxnn" \
  --reference-run "$vnRun" \
  --input demo_images

echo "Send the layout lines, cut-node/operator summary, and ONNX/Python/C++ cosine lines."
