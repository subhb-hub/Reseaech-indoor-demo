#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${1:-2}"

python src/train.py --model official_like_cnn --variant baseline --epochs "$EPOCHS" --use-subset --device cpu
python src/train.py --model official_like_cnn --variant edge --epochs "$EPOCHS" --use-subset --device cpu
python src/train.py --model official_like_cnn --variant noise --epochs "$EPOCHS" --use-subset --device cpu

echo ""
echo "Finished official_like_cnn baseline / edge / noise experiments."
