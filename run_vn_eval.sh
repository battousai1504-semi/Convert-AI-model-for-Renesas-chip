#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

vnPython="$PWD/.venv-linux/bin/python"
vnRun="${1:-artifacts/dxm1_full_encoder_mm_20261101_134233_21170}"
for vnFile in "$vnPython" alpr_auto.py alpr_auto_dxm1.py \
    "$vnRun/matmul_contract.json"; do
    if [ ! -f "$vnFile" ]; then
        echo "STOP: Missing $vnFile"
        exit 1
    fi
done
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
"$vnPython" -m unittest discover -s tests -p test_alpr_dataset_eval.py -v
"$vnPython" prepare_vn_eval.py --calibration-manifest "$vnRun/matmul_contract.json"
vnOutput="results/vn_eval_$(date +%Y%m%d_%H%M%S)_$$"
echo "Evaluation output: $vnOutput"
"$vnPython" alpr_auto.py --source inputs/vn_eval_v1/images \
    --device cpu --no-preview --output "$vnOutput/cpu"
"$vnPython" alpr_auto_dxm1.py --dx-run "$vnRun" --source inputs/vn_eval_v1/images \
    --device cpu --no-preview --output "$vnOutput/dxm1"
"$vnPython" evaluate_alpr_dataset.py \
    --dataset inputs/vn_eval_v1/dataset.json \
    --cpu-run "$vnOutput/cpu" --dxm1-run "$vnOutput/dxm1" \
    --output "$vnOutput/evaluation"
echo "Summary: $vnOutput/evaluation/summary.json"
echo "Review:  $vnOutput/evaluation/review.html"
