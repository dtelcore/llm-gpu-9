#!/bin/bash
# Fresh v0.1.1 TinyStories train, or a resume of THIS checkpoint only.
# The cosine length is --steps. Do not resume a finished budget to make it longer.
# Usage:
#   setup/english_tinystories_c1024_l6/train.sh            # 4000 steps from step 0
#   setup/english_tinystories_c1024_l6/train.sh 8000       # a different fresh length
#   setup/english_tinystories_c1024_l6/train.sh --resume   # add 4000 to this run
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CKPT="output/checkpoints/english_tinystories_c1024_l6"
CONFIG="setup/english_tinystories_c1024_l6_config.json"
RESUME=0
if [[ "${1:-}" == "--resume" ]]; then
  RESUME=1
  shift
fi
STEPS="${1:-4000}"

python setup/english_tinystories_c1024_l6/check_data.py

if [[ "$RESUME" -eq 0 && -f "$CKPT/weights.npz" ]]; then
  echo "Refusing a fresh start: $CKPT/weights.npz already exists." >&2
  echo "Continue with: setup/english_tinystories_c1024_l6/train.sh --resume" >&2
  echo "A longer cosine is a new checkpoint and a new --steps, not a resume." >&2
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
  --checkpoint-every 1000
)
if [[ "$RESUME" -eq 1 ]]; then
  ARGS+=(--resume)
fi
exec "${ARGS[@]}"
