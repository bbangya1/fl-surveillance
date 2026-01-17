#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_central.sh
#   EPOCHS=20 BATCH=64 LR=0.01 SEED=42 ./scripts/run_central.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${DATA_DIR:-./data}"
OUT_DIR="${OUT_DIR:-./outputs}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-64}"
LR="${LR:-0.01}"
SEED="${SEED:-42}"

mkdir -p "$DATA_DIR" "$OUT_DIR"

echo "[run_central] python=$PYTHON_BIN data_dir=$DATA_DIR out_dir=$OUT_DIR epochs=$EPOCHS batch=$BATCH lr=$LR seed=$SEED"

"$PYTHON_BIN" run_central.py \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --lr "$LR" \
  --seed "$SEED"

echo "[run_central] Done. Output: $OUT_DIR/central_vendor.pt"
