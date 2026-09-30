"""Central Apple MLX app: model selector + chat + npzviewer + train monitor.

One process, one Metal checkpoint (2 GB). Viewer mmaps the same weights.
Train tab reads logs only. Do not run this together with webui.py or
interactive.py. Do not Load a checkpoint while unguided_trainer.py holds Metal
— use python trainmon.py (7862) instead.

Usage:
    python App.py
    python App.py --checkpoint output/checkpoints/chat_facts_v6 --chat
"""

from __future__ import annotations

import argparse
from argparse import Namespace

from app import create_app, load_chat_session
from app.models import infer_facts, resolve_model
from paths import DATA_DIR, OUTPUT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apple MLX app (chat + npzviewer + model selector)")
    parser.add_argument("--checkpoint", type=str, default="", help="Load this checkpoint on startup")
    parser.add_argument("--chat", action="store_true", help="Multi-turn chat (default on)")
    parser.add_argument("--no-chat", action="store_true", help="Single-prompt generate")
    parser.add_argument("--system", type=str, default=None, help="Optional chat system prefix")
    parser.add_argument("--facts", type=str, default="", help="Cabinet JSONL (inferred from checkpoint if omitted)")
    parser.add_argument(
        "--learned", type=str, default=str(OUTPUT_ROOT / "cabinet_learned.jsonl"),
        help="Wikipedia overlay JSONL",
    )
    parser.add_argument("--no-search", action="store_true", help="Skip Wikipedia")
    parser.add_argument(
        "--cabinet-retrain-log",
        type=str,
        default=str(OUTPUT_ROOT / "cabinet_retrain.jsonl"),
        help="JSONL path for generate_target_mismatch events",
    )
    parser.add_argument(
        "--cabinet-diagnostics-log",
        type=str,
        default=str(OUTPUT_ROOT / "cabinet_diagnostics.jsonl"),
        help="JSONL path for per-turn router/generation diagnostics",
    )
    parser.add_argument("--router", dest="router", action="store_true")
    parser.add_argument("--no-router", dest="router", action="store_false")
    parser.set_defaults(router=None, chat=True)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def _session_args(args: argparse.Namespace) -> Namespace:
    facts = args.facts
    if args.checkpoint and not facts:
        facts = infer_facts(args.checkpoint)
    if not facts:
        facts = str(DATA_DIR / "chat_facts.jsonl")
    return Namespace(
        checkpoint=args.checkpoint,
        seed=args.seed,
        chat=bool(args.chat) and not args.no_chat,
        no_chat=args.no_chat,
        system=args.system,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        stop=None,
        router=args.router,
        facts=facts,
        learned=args.learned,
        no_search=args.no_search,
        cabinet_retrain_log=getattr(args, "cabinet_retrain_log", str(OUTPUT_ROOT / "cabinet_retrain.jsonl")),
        cabinet_diagnostics_log=getattr(args, "cabinet_diagnostics_log", str(OUTPUT_ROOT / "cabinet_diagnostics.jsonl")),
        no_kv_cache=False,
        cuda_graph=False,
        verbose=False,
        trace_logits=False,
        trace_tokens=False,
        trace_neurons=False,
        trace_vectorization=False,
        trace_every=None,
        host=args.host,
        port=args.port,
        _entry="app",
    )


def main() -> None:
    args = parse_args()
    ns = _session_args(args)
    session = None
    selected = None
    if ns.checkpoint:
        selected = resolve_model(ns.checkpoint)
        if selected is not None:
            ns.checkpoint = str(selected["checkpoint"])
            if not args.facts:
                ns.facts = str(selected["facts"])
        session = load_chat_session(ns)
        st = session.status()
        print("=" * 70)
        print(f"APP -- checkpoint: {st['checkpoint']}  model: {st['model']}")
        print(f"Open http://{args.host}:{args.port}  (one checkpoint, 2 GB)")
        print("=" * 70)
    else:
        print("=" * 70)
        print("APP -- no checkpoint loaded (selector only; viewer mmap is safe)")
        print(f"Open http://{args.host}:{args.port}  then Load a model")
        print("=" * 70)
    app = create_app(session=session, selected=selected, args=ns)
    app.run(host=args.host, port=int(args.port), debug=False, threaded=False, use_reloader=False)


if __name__ == "__main__":
    main()
