#!/bin/bash
# Story sample from the v0.1.1 checkpoint. Stops on <|endofstory|>.
# Usage:
#   setup/english_tinystories_c1024_l6/generate.sh
#   setup/english_tinystories_c1024_l6/generate.sh "Once upon a time there was a little girl named Lily who found a"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CKPT="output/checkpoints/english_tinystories_c1024_l6"
PROMPT="${1:-Once upon a time there was a brave little mouse named}"

if [[ ! -f "$CKPT/weights.npz" ]]; then
  echo "No weights at $CKPT" >&2
  exit 1
fi

exec ./venv/bin/python generate.py \
  --checkpoint "$CKPT" \
  --prompt "$PROMPT" \
  --max-new-tokens 160 \
  --temperature 0.8 \
  --seed 42 \
  --no-prompt
