#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${1:-1}"

python src/train.py --variant baseline --epochs "$EPOCHS" --use-subset --device cpu

echo ""
echo "Baseline finished."
echo "After the class implements edge/noise TODOs in src/transforms_custom.py, run:"
echo "python src/train.py --variant edge --epochs $EPOCHS --use-subset --device cpu"
echo "python src/train.py --variant noise --epochs $EPOCHS --use-subset --device cpu"
