"""
interactive.py

Interactive REPL for sampling from a trained checkpoint. Each prompt you
type continues from the model; trace flags apply to every generation
you run in the session.

Usage:
    python interactive.py --checkpoint output/checkpoints/run1
    python interactive.py --checkpoint output/checkpoints/chat_5m --chat
    python interactive.py --checkpoint output/checkpoints/run1 --trace-tokens --trace-logits --trace-every 1

Session commands:
    :temp <value>       set sampling temperature
    :tokens <n>         set max new tokens per turn
    :topk <n>|none      set top-k
    :topp <n>|none      set top-p
    :clear              reset chat history (chat mode)
    :system <text>      set / replace the system prefix (chat mode)
    :trace on|off       toggle all tracing for subsequent turns
    :quit / :exit       leave the REPL
"""

from __future__ import annotations

import argparse
from typing import List

import numpy as np

import cli_common
from logging_config import logger, setup_generate_run_logging
from model.gpt import GPTModel
from paths import ensure_output_dirs
from training.checkpoint import load_checkpoint
from training.chat_format import (
    ASSISTANT_ROLE,
    CHAT_STOP_STRINGS,
    DEFAULT_CHAT_SYSTEM,
    DEFAULT_CHAT_TEMPERATURE,
    DEFAULT_CHAT_TOP_K,
    DEFAULT_CHAT_TOP_P,
    USER_ROLE,
    build_chat_prompt_ids,
    is_chat_model_name,
    sanitize_assistant_reply,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive REPL for a trained checkpoint")
    cli_common.add_checkpoint_arg(parser)
    cli_common.add_seed_arg(parser)
    parser.add_argument("--chat", action="store_true", help="Multi-turn User/Assistant history")
    parser.add_argument(
        "--no-chat", action="store_true",
        help="Force single-prompt REPL even if the checkpoint name looks like chat_5m",
    )
    parser.add_argument(
        "--system", type=str, default=None,
        help="Optional system prefix for --chat (default: a short simple-assistant line)",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Initial sampling temperature")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Initial tokens generated per turn")
    parser.add_argument("--top-k", type=int, default=None, help="Only sample from top K tokens")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling threshold (e.g. 0.9)")
    parser.add_argument(
        "--stop", action="append", default=None,
        help="Stop string (repeatable). Chat mode defaults to User: role markers.",
    )
    cli_common.add_generate_decode_args(parser)
    cli_common.add_trace_args(parser)
    return parser.parse_args()


def _resolve_chat_mode(args: argparse.Namespace, model_name: str) -> bool:
    if getattr(args, "no_chat", False):
        return False
    if getattr(args, "chat", False):
        return True
    return is_chat_model_name(model_name)


def run_repl(args: argparse.Namespace, *, configure_logging: bool = True) -> None:
    """Runs the interactive generation REPL for the checkpoint in `args.checkpoint`.
    Reusable by other CLIs (e.g. train.py --generate) that build their own args."""
    ensure_output_dirs()
    log_path = None
    if configure_logging:
        log_path = setup_generate_run_logging(args.checkpoint)
        source = getattr(args, "_entry", None) or "interactive"
        logger.info(
            "interactive generation | source=%s | checkpoint=%s | log=%s",
            source, args.checkpoint, log_path,
        )

    gpt_config, params, tokenizer, _, _ = load_checkpoint(args.checkpoint)
    model = GPTModel(gpt_config, params)
    tracer = cli_common.build_tracer(args, default_trace_every=1)
    rng = np.random.default_rng(args.seed)

    chat_mode = _resolve_chat_mode(args, getattr(gpt_config, "name", "") or "")
    if chat_mode:
        temperature = args.temperature if args.temperature is not None else DEFAULT_CHAT_TEMPERATURE
        top_k = args.top_k if args.top_k is not None else DEFAULT_CHAT_TOP_K
        top_p = args.top_p if args.top_p is not None else DEFAULT_CHAT_TOP_P
        system = args.system if args.system is not None else DEFAULT_CHAT_SYSTEM
        stop_strings = list(args.stop) if args.stop else list(CHAT_STOP_STRINGS)
    else:
        temperature = args.temperature if args.temperature is not None else 0.8
        top_k = getattr(args, "top_k", None)
        top_p = getattr(args, "top_p", None)
        system = args.system
        stop_strings = list(args.stop) if args.stop else None
    max_new_tokens = args.max_new_tokens
    trace_enabled = tracer.any_enabled
    history: List[tuple] = []
    use_kv_cache = not getattr(args, "no_kv_cache", False)
    use_cuda_graph = bool(getattr(args, "cuda_graph", False))

    print("=" * 70)
    mode_label = "chat" if chat_mode else "prompt"
    print(f"INTERACTIVE GENERATION -- checkpoint: {args.checkpoint} ({mode_label})")
    print(f"Model: {gpt_config.name} | vocab={gpt_config.vocab_size} | max_len={gpt_config.max_len}")
    if log_path is not None:
        print(f"Log: {log_path}")
    cmds = ":temp N  :tokens N  :topk N  :topp N  :trace on|off  :quit"
    if chat_mode:
        cmds = ":clear  :system TEXT  " + cmds
        print(f"Chat format: User: … Assistant: …  (history truncated from the front)")
        if system:
            print(f"System: {system}")
    print(f"Type a prompt and press Enter. Commands: {cmds}")
    print("=" * 70)

    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not prompt:
            continue
        if prompt in (":quit", ":exit"):
            break
        if prompt.startswith(":temp "):
            temperature = float(prompt.split(maxsplit=1)[1])
            print(f"[temperature -> {temperature}]")
            continue
        if prompt.startswith(":tokens "):
            max_new_tokens = int(prompt.split(maxsplit=1)[1])
            print(f"[max_new_tokens -> {max_new_tokens}]")
            continue
        if prompt.startswith(":topk "):
            raw = prompt.split(maxsplit=1)[1].strip().lower()
            top_k = None if raw in ("none", "off", "0") else int(raw)
            print(f"[top_k -> {top_k}]")
            continue
        if prompt.startswith(":topp "):
            raw = prompt.split(maxsplit=1)[1].strip().lower()
            top_p = None if raw in ("none", "off", "1", "1.0") else float(raw)
            print(f"[top_p -> {top_p}]")
            continue
        if prompt.startswith(":trace "):
            trace_enabled = prompt.split(maxsplit=1)[1].strip().lower() == "on"
            print(f"[tracing -> {'on' if trace_enabled else 'off'}]")
            continue
        if chat_mode and prompt == ":clear":
            history = []
            print("[history cleared]")
            continue
        if chat_mode and prompt.startswith(":system"):
            rest = prompt[len(":system"):].strip()
            system = rest or None
            print(f"[system -> {system!r}]")
            continue

        if chat_mode:
            budget = max(1, int(gpt_config.max_len) - int(max_new_tokens))
            prompt_ids, prompt_text = build_chat_prompt_ids(
                tokenizer,
                history,
                prompt,
                system=system,
                max_prompt_tokens=budget,
            )
        else:
            prompt_text = prompt
            prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            print("[No recognized characters in prompt for this vocabulary; try different text]")
            continue

        active_tracer = tracer if trace_enabled else None
        if active_tracer is not None:
            active_tracer.dump_tokens(prompt_ids, tokenizer, label="prompt")

        need_tokenizer = active_tracer is not None or bool(stop_strings)
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            tracer=active_tracer,
            tokenizer=tokenizer if need_tokenizer else None,
            rng=rng,
            use_kv_cache=use_kv_cache,
            use_cuda_graph=use_cuda_graph,
            stop_strings=stop_strings,
        )
        new_ids = generated_ids[len(prompt_ids):]
        reply = tokenizer.decode(new_ids)
        if chat_mode:
            reply = sanitize_assistant_reply(reply, stop_strings)
        else:
            reply = reply.strip()
        full_text = tokenizer.decode(generated_ids)
        logger.info("prompt=%r generated_text:\n%s", prompt_text, full_text)
        if chat_mode:
            print(reply)
            history.append((USER_ROLE, prompt))
            history.append((ASSISTANT_ROLE, reply))
        else:
            print(full_text)


def main() -> None:
    args = parse_args()
    run_repl(args)


if __name__ == "__main__":
    main()
