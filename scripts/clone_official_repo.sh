#!/usr/bin/env bash
set -euo pipefail

OFFICIAL_DIR="${1:-official_fashion_mnist}"
TARGET_DIR="third_party/fashion-mnist"

if [ -d "$OFFICIAL_DIR/.git" ]; then
  echo "$OFFICIAL_DIR already exists."
  echo "Run this if you want to update it later:"
  echo "  git -C $OFFICIAL_DIR pull"
else
  git clone https://github.com/zalandoresearch/fashion-mnist.git "$OFFICIAL_DIR"
fi

mkdir -p "$TARGET_DIR"
mkdir -p "$TARGET_DIR/benchmark"
cp "$OFFICIAL_DIR/README.md" "$TARGET_DIR/"

if [ -f "$OFFICIAL_DIR/README.zh-CN.md" ]; then
  cp "$OFFICIAL_DIR/README.zh-CN.md" "$TARGET_DIR/"
else
  echo "README.zh-CN.md was not found in the official repo; skipping optional file."
fi

cp "$OFFICIAL_DIR/LICENSE" "$TARGET_DIR/"
cp "$OFFICIAL_DIR/utils/mnist_reader.py" "$TARGET_DIR/"
cp "$OFFICIAL_DIR/benchmark/convnet.py" "$TARGET_DIR/benchmark/"

python scripts/check_official_clone.py
