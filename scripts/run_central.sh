#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_central.sh
#   EPOCHS=20 BATCH=64 LR=0.01 SEED=42 ./scripts/run_central.sh
#
# Optional perf knobs:
#   NUM_WORKERS=4 PIN_MEMORY=1 ./scripts/run_central.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Prefer venv python if exists
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

DATA_DIR="${DATA_DIR:-./data}"
OUT_DIR="${OUT_DIR:-./outputs}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-64}"
LR="${LR:-0.01}"
SEED="${SEED:-42}"

# new flags for run_central.py
NUM_WORKERS="${NUM_WORKERS:-0}"
PIN_MEMORY="${PIN_MEMORY:-0}"   # 1이면 enable

mkdir -p "$DATA_DIR" "$OUT_DIR"

echo "[run_central] python=$PYTHON_BIN data_dir=$DATA_DIR out_dir=$OUT_DIR epochs=$EPOCHS batch=$BATCH lr=$LR seed=$SEED num_workers=$NUM_WORKERS pin_memory=$PIN_MEMORY"

PIN_ARG=()
if [[ "$PIN_MEMORY" == "1" ]]; then
  PIN_ARG+=(--pin_memory)
fi

"$PYTHON_BIN" run_central.py \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --lr "$LR" \
  --seed "$SEED" \
  --num_workers "$NUM_WORKERS" \
  "${PIN_ARG[@]}"

echo "[run_central] Done."
echo "  - Model:   $OUT_DIR/central_vendor.pt"
echo "  - Metrics: $OUT_DIR/vendor_metrics.csv"
