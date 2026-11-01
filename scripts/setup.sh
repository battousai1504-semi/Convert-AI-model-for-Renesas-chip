#!/usr/bin/env bash
set -euo pipefail

device="${1:-cpu}"
case "$device" in
  cpu) requirement_file="requirements/pc-cpu.txt" ;;
  cuda) requirement_file="requirements/pc-cu121.txt" ;;
  *) echo "Usage: bash scripts/setup.sh [cpu|cuda]" >&2; exit 2 ;;
esac

if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
uv python install 3.10
if [[ ! -x .venv-linux/bin/python ]]; then
  uv venv --python 3.10 --seed .venv-linux
fi
venv_python="$repo_root/.venv-linux/bin/python"
"$venv_python" -c "import sys; assert sys.version_info[:2] == (3, 10), 'Use a fresh Python 3.10 environment'; print(sys.executable)"
"$venv_python" -m pip install -r "$requirement_file" -e .
"$venv_python" -m pip check
"$venv_python" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
echo "Ready. Example: .venv-linux/bin/python read.py pretrained=parseq --images demo_images/coco-1166773.jpg --device $device"
