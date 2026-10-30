#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/.cache/huggingface"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
PYTHON=.venv/bin/python
if [[ ! -x "$PYTHON" ]]; then
  echo '请先按 README 创建 .venv 并安装 requirements-lock.txt' >&2
  exit 1
fi
"$PYTHON" scripts/prepare_demo.py
if [[ ! -f models/vjepa2-vitl/model.safetensors ]]; then
  "$PYTHON" scripts/download_model.py --source "${MODEL_SOURCE:-auto}"
fi
"$PYTHON" vjepa2_site_monitor.py data/raw/excavator.webm \
  --model models/vjepa2-vitl --fit pad --device auto --batch 1 --annotated-video \
  --out outputs/excavator --overwrite
"$PYTHON" scripts/validate_method.py --device auto
"$PYTHON" scripts/write_validation_report.py
bash scripts/run_real_hazard.sh
