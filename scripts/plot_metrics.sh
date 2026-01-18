#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/plot_metrics.sh
#   OUT_DIR=./outputs ./scripts/plot_metrics.sh
#   TITLE_PREFIX="FL Surveillance" ./scripts/plot_metrics.sh
#
# Optional explicit CSVs:
#   FEDAVG_CSV=./outputs/metrics.csv VENDOR_CSV=./outputs/vendor_metrics.csv LOCAL_CSV=./outputs/local_metrics.csv ./scripts/plot_metrics.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Prefer venv python if exists
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

OUT_DIR="${OUT_DIR:-./outputs}"
FEDAVG_CSV="${FEDAVG_CSV:-$OUT_DIR/metrics.csv}"
VENDOR_CSV="${VENDOR_CSV:-$OUT_DIR/vendor_metrics.csv}"
LOCAL_CSV="${LOCAL_CSV:-$OUT_DIR/local_metrics.csv}"
TITLE_PREFIX="${TITLE_PREFIX:-FL Surveillance}"

mkdir -p "$OUT_DIR"

echo "[plot_metrics] python=$PYTHON_BIN"
echo "[plot_metrics] out_dir=$OUT_DIR"
echo "[plot_metrics] fedavg_csv=$FEDAVG_CSV"
echo "[plot_metrics] vendor_csv=$VENDOR_CSV"
echo "[plot_metrics] local_csv=$LOCAL_CSV"
echo "[plot_metrics] title_prefix=$TITLE_PREFIX"
echo ""

"$PYTHON_BIN" plot_metrics.py \
  --out_dir "$OUT_DIR" \
  --fedavg_csv "$FEDAVG_CSV" \
  --vendor_csv "$VENDOR_CSV" \
  --local_csv "$LOCAL_CSV" \
  --title_prefix "$TITLE_PREFIX"

echo ""
echo "[plot_metrics] Done. Generated (if inputs exist):"
echo "  - $OUT_DIR/metrics.png (+ metrics_loss/metrics_acc/metrics_macro_f1)"
echo "  - $OUT_DIR/compare_loss.png"
echo "  - $OUT_DIR/compare_acc.png"
echo "  - $OUT_DIR/compare_macro_f1.png"
