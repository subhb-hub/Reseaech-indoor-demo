#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${1:-1}"

python train_fashion.py --variant baseline --epochs "$EPOCHS" --device cpu

echo ""
echo "Baseline finished."
echo "After the class implements edge/noise TODOs in train_fashion.py, run:"
echo "python train_fashion.py --variant edge --epochs $EPOCHS --device cpu"
echo "python train_fashion.py --variant noise --epochs $EPOCHS --device cpu"
