#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <CREMAD_ROOT> <OUTPUT_DIR> [GPU_IDS]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$ROOT_DIR/rcc_guard.py" train \
  --dataset cremad \
  --data-root "$1" \
  --output "$2" \
  --seeds 0 1 2 \
  --gpu-ids "${3:-0}"
