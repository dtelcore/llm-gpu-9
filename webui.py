"""Flask chat UI around ChatSession (same path as interactive.py).

Prefer ``python App.py`` for the model selector + viewer in one process.

Usage:
    python webui.py --checkpoint output/checkpoints/chat_facts_v6 --chat --facts data/chat_facts_v6.jsonl
"""

from __future__ import annotations

import argparse

from app.webui import create_app
from paths import DEFAULT_VIEWER_URL
from training.chat_session import ChatSession, add_session_args

__all__ = ["create_app", "main", "parse_args"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flask chat UI for the interactive session")
    add_session_args(parser)
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address (default loopback)")
    parser.add_argument("--port", type=int, default=7860, help="Port (default 7860)")
    parser.add_argument(
        "--viewer-url", type=str, default=DEFAULT_VIEWER_URL,
        help="npzviewer base URL for the Weights nav link (default 7861)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args._entry = "webui"
    session = ChatSession.from_args(args, configure_logging=True)
    app = create_app(session, viewer_url=args.viewer_url)
    st = session.status()
    print("=" * 70)
    print(f"WEBUI -- checkpoint: {st['checkpoint']}  model: {st['model']}")
    print(f"Open http://{args.host}:{args.port}  (one checkpoint, 2 GB; quit interactive.py first)")
    print("=" * 70)
    app.run(host=args.host, port=int(args.port), debug=False, threaded=False, use_reloader=False)


if __name__ == "__main__":
    main()
