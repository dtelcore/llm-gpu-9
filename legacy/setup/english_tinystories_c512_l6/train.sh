#!/bin/bash
# Fresh c512 TinyStories train, or a resume of THIS checkpoint only.
# Usage:
#   setup/english_tinystories_c512_l6/train.sh            # first 50000 steps
#   setup/english_tinystories_c512_l6/train.sh 50000      # another fresh chunk length
#   setup/english_tinystories_c512_l6/train.sh --resume   # add 50000 to this run
#   setup/english_tinystories_c512_l6/train.sh --resume 100000
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

CKPT="output/checkpoints/english_tinystories_c512_l6"
CONFIG="legacy/setup/english_tinystories_c512_l6_config.json"
RESUME=0
if [[ "${1:-}" == "--resume" ]]; then
  RESUME=1
  shift
fi
STEPS="${1:-50000}"

python legacy/setup/english_tinystories_c512_l6/check_data.py

if [[ "$RESUME" -eq 0 && -f "$CKPT/weights.npz" ]]; then
  echo "Refusing a fresh start: $CKPT/weights.npz already exists." >&2
  echo "Continue with: setup/english_tinystories_c512_l6/train.sh --resume" >&2
  exit 1
fi
if [[ "$RESUME" -eq 1 && ! -f "$CKPT/weights.npz" ]]; then
  echo "Nothing to resume at $CKPT" >&2
  exit 1
fi

ARGS=(
  ./venv/bin/python auto_train.py
  --config "$CONFIG"
  --checkpoint "$CKPT"
  --steps "$STEPS"
  --seed 42
  --no-prompt
  --log-every 25
  --checkpoint-every 10000
)
if [[ "$RESUME" -eq 1 ]]; then
  ARGS+=(--resume)
fi
exec "${ARGS[@]}"
