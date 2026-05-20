#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${1:-1}"

python train_fashion.py --variant baseline --epochs "$EPOCHS" --device cpu
python train_fashion.py --variant edge --epochs "$EPOCHS" --device cpu
python train_fashion.py --variant noise --epochs "$EPOCHS" --device cpu
