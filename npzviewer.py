"""Flask inspector for .npz / .npy / .npx weights (web npzviewer).

Prefer ``python App.py`` to bind the viewer to the same checkpoint as chat.

Usage:
    python npzviewer.py
    python npzviewer.py --open output/checkpoints/chat_facts_v6/weights.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.npzviewer import create_app
from paths import DEFAULT_CHAT_URL, OUTPUT_CHECKPOINTS, PROJECT_ROOT
from tools.npz_inspect import resolve_weight_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web npzviewer for NumPy weight files")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address (default loopback)")
    parser.add_argument("--port", type=int, default=7861, help="Port (default 7861)")
    parser.add_argument("--open", type=str, default="", help="Weight file to select on first load")
    parser.add_argument(
        "--chat-url", type=str, default=DEFAULT_CHAT_URL,
        help="Chat UI base URL for the Chat nav link (default 7860)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(PROJECT_ROOT),
        help="Sandbox root (default: project root)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    initial = args.open.strip()
    if initial:
        resolve_weight_path(initial, root)
    app = create_app(root=root, initial=initial, chat_url=args.chat_url)
    print("=" * 70)
    print("NPZVIEWER -- host mmap of .npz / .npy / .npx (no Metal load)")
    print(f"Checkpoints: {OUTPUT_CHECKPOINTS}")
    if initial:
        print(f"Open: {initial}")
    print(f"Open http://{args.host}:{args.port}")
    print("=" * 70)
    app.run(host=args.host, port=int(args.port), debug=False, threaded=False, use_reloader=False)


if __name__ == "__main__":
    main()
