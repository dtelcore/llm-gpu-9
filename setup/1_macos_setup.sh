#!/bin/zsh
# macOS setup for the M3 Air MLX GPT port (Python 3.11 or 3.12).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if command -v python3.11 >/dev/null 2>&1; then
  PY=python3.11
elif command -v python3.12 >/dev/null 2>&1; then
  PY=python3.12
else
  echo "Need Python 3.11 or 3.12 (Homebrew: brew install python@3.11)" >&2
  exit 1
fi

if [[ -x venv/bin/python ]]; then
  echo "Using existing ./venv"
else
  "$PY" -m venv venv
fi
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python setup/2_test_workspace.py
echo "OK: venv ready. Activate with: source venv/bin/activate"
