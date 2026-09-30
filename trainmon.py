#!/usr/bin/env python3
"""Host-only training monitor (same series as the desktop plotters).

Reads output/logs/*.log. Does not load Metal. Safe beside unguided_trainer.py.
Do not start App.py with a checkpoint while a train is running.

Usage:
    python trainmon.py
    python trainmon.py --port 7862
"""

from __future__ import annotations

import argparse

from app.trainmon import create_app
from paths import OUTPUT_LOGS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web training monitor (log read only)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--log-dir", type=str, default=str(OUTPUT_LOGS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(log_dir=args.log_dir)
    print("=" * 70)
    print("TRAINMON -- training logs only (no Metal)")
    print(f"Logs: {args.log_dir}")
    print(f"Open http://{args.host}:{args.port}")
    print("=" * 70)
    app.run(host=args.host, port=int(args.port), debug=False, threaded=False, use_reloader=False)


if __name__ == "__main__":
    main()
