#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <CREMAD_ROOT> <CHECKPOINT> <OUTPUT_DIR> [GPU_IDS]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$ROOT_DIR/rcc_guard.py" eval \
  --dataset cremad \
  --data-root "$1" \
  --checkpoint "$2" \
  --output "$3" \
  --gpu-ids "${4:-0}"
