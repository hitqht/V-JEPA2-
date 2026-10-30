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
"$PYTHON" scripts/prepare_real_hazard.py
if [[ ! -f models/vjepa2-vitl/model.safetensors ]]; then
  "$PYTHON" scripts/download_model.py --source "${MODEL_SOURCE:-auto}"
fi
"$PYTHON" vjepa2_site_monitor.py data/derived/big_blue_calibration.mp4 \
  --model models/vjepa2-vitl --fit pad --device auto --out outputs/big_blue_calibration --overwrite
for sample in big_blue_negative big_blue_positive; do
  "$PYTHON" vjepa2_site_monitor.py "data/derived/$sample.mp4" \
    --model models/vjepa2-vitl --fit pad --device auto \
    --baseline-csv outputs/big_blue_calibration/scores.csv \
    --annotated-video --out "outputs/$sample" --overwrite
done
"$PYTHON" scripts/validate_real_hazard.py --device auto
"$PYTHON" scripts/render_hazard_review.py
