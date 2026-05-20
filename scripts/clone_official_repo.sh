#!/usr/bin/env bash
set -euo pipefail

mkdir -p external

if [ -d external/fashion-mnist/.git ]; then
  echo "external/fashion-mnist already exists."
  echo "Run this if you want to update it later:"
  echo "  git -C external/fashion-mnist pull"
else
  git clone https://github.com/zalandoresearch/fashion-mnist.git external/fashion-mnist
fi

python scripts/check_official_clone.py
