"""Chat + tool-calling inference loop for gpu9_max512 checkpoints.

The model is trained to emit plain-text tool calls in this exact format:

    Tool: calc | Query: 17 + 4 | Result: 21 | Answer: 17 + 4 is 21.
    Tool: search | Query: capital of France | Result: ... | Answer: ...

This loop closes the loop for real:
  1. Generate an Assistant turn (chat history, User:/Assistant: roles).
  2. If the turn has a Query but no (or a wrong) Result, execute the tool
     locally (safe calc, or Wikipedia websearch) and re-query the model with
     the true Result appended so the final Answer is grounded.
  3. Otherwise the trained Result stands and the Answer is shown as-is.

Calc is offline (tools/calc.py, no eval). Websearch uses the Wikipedia REST
API with a short timeout and degrades to a clear offline message, so the
assistant works with or without network.

Usage:
    python tools/assistant_tools.py --checkpoint output/checkpoints/gpu9_max512 --chat
    python tools/assistant_tools.py --checkpoint output/checkpoints/gpu9_max512 --ask "What is 17 + 4?"
    python tools/assistant_tools.py --checkpoint output/checkpoints/gpu9_max512 --ask "What city is the capital of France?"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from model.gpt import GPTModel  # noqa: E402
from training.chat_format import (  # noqa: E402
    ASSISTANT_ROLE,
    CHAT_STOP_STRINGS,
    DEFAULT_CHAT_SYSTEM,
    USER_ROLE,
    build_chat_prompt_ids,
    sanitize_assistant_reply,
)
from training.checkpoint import load_checkpoint as _load  # noqa: E402
from tools.calc import try_calc  # noqa: E402

TOOL_RE = re.compile(
    r"Tool:\s*(calc|search)\s*\|\s*Query:\s*(?P<query>[^|]+?)\s*"
    r"(?:\|\s*Result:\s*(?P<result>[^|]*?))?\s*(?:\|\s*Answer:\s*(?P<answer>.*))?$",
    re.IGNORECASE | re.DOTALL,
)

SEARCH_TIMEOUT_S = 8


def websearch(query: str) -> str:
    """One-shot Wikipedia summary search. Offline-safe, read-only HTTP."""
    q = " ".join((query or "").split())
    if not q:
        return "empty query."
    params = urllib.parse.urlencode({
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": "1",
        "explaintext": "1",
        "exsentences": "2",
        "titles": q,
        "redirects": "1",
    })
    url = f"https://en.wikipedia.org/w/api.php?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "llm-gpu-9-assistant/0.1"})
        with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        pages = ((payload.get("query") or {}).get("pages") or {}).values()
        for page in pages:
            extract = (page.get("extract") or "").strip()
            if extract and "missing" not in page:
                return " ".join(extract.split())
        return f"No Wikipedia article found for {q!r}."
    except Exception as exc:  # offline or blocked: stay a chat assistant, say so
        return f"Websearch unavailable ({exc}). Answering from trained knowledge."


def run_tool(kind: str, query: str) -> str:
    kind = (kind or "").strip().lower()
    if kind == "calc":
        value = try_calc(query)
        return value if value is not None else "not an arithmetic expression."
    if kind == "search":
        return websearch(query)
    return "unknown tool."


def parse_tool_call(reply: str) -> Optional[Tuple[str, str, Optional[str], Optional[str]]]:
    """Return (kind, query, result, answer) or None when no tool call present."""
    match = TOOL_RE.search(reply or "")
    if not match:
        return None
    kind = match.group(1).lower()
    query = (match.group("query") or "").strip()
    result = match.group("result")
    result = result.strip() if result is not None else None
    answer = match.group("answer")
    answer = answer.strip() if answer is not None else None
    if not query:
        return None
    return kind, query, result, answer


def generate_turn(model: GPTModel, tokenizer, history, user_text: str,
                  system: Optional[str], max_new_tokens: int,
                  temperature: float, top_k, top_p, rng) -> Tuple[str, str]:
    """Generate one Assistant reply for user_text given history. Returns (reply, prompt_text)."""
    budget = max(1, int(model.config.max_len) - int(max_new_tokens))
    prompt_ids, prompt_text = build_chat_prompt_ids(
        tokenizer, history, user_text, system=system, max_prompt_tokens=budget,
    )
    if not prompt_ids:
        return "", prompt_text
    gen_ids = model.generate(
        prompt_ids,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=top_k,
        top_p=top_p,
        rng=rng,
        tokenizer=tokenizer,
        stop_strings=list(CHAT_STOP_STRINGS),
    )
    reply = sanitize_assistant_reply(tokenizer.decode(gen_ids[len(prompt_ids):]),
                                     list(CHAT_STOP_STRINGS))
    return reply, prompt_text


def answer_with_tools(model: GPTModel, tokenizer, history, user_text: str,
                      system: Optional[str], max_new_tokens: int,
                      temperature: float, top_k, top_p, rng,
                      verbose: bool = False) -> str:
    """Chat turn with a live tool loop (max one tool execution per turn)."""
    reply, _ = generate_turn(model, tokenizer, history, user_text, system,
                             max_new_tokens, temperature, top_k, top_p, rng)
    parsed = parse_tool_call(reply)
    if parsed is None:
        return reply
    kind, query, result, _ = parsed
    if result:  # model already supplied Result + Answer from training
        if verbose:
            print(f"[tool] {kind} query={query!r} (trained result kept)")
        return reply
    live = run_tool(kind, query)
    if verbose:
        print(f"[tool] {kind} query={query!r} result={live!r}")
    # Second pass: model writes the Answer grounded on the true Result.
    follow_up = (
        f"{user_text} [Tool {kind} result for {query!r}: {live}] "
        f"Answer using that result."
    )
    reply2, _ = generate_turn(model, tokenizer, history, follow_up, system,
                              max_new_tokens, temperature, top_k, top_p, rng)
    if parse_tool_call(reply2) is None:
        reply2 = (
            f"Tool: {kind} | Query: {query} | Result: {live} | Answer: {reply2}"
        )
    return reply2


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat + tool-calling assistant")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ask", type=str, default=None, help="Single question (non-interactive)")
    parser.add_argument("--system", type=str, default=DEFAULT_CHAT_SYSTEM)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose-tools", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    gpt_config, params, tokenizer, _, _ = _load(args.checkpoint)
    model = GPTModel(gpt_config, params)
    rng = np.random.default_rng(args.seed)
    history = []

    def _turn(user_text: str) -> str:
        reply = answer_with_tools(
            model, tokenizer, history, user_text,
            args.system, args.max_new_tokens, args.temperature,
            args.top_k, args.top_p, rng, verbose=args.verbose_tools,
        )
        history.append((USER_ROLE, user_text))
        history.append((ASSISTANT_ROLE, reply))
        return reply

    if args.ask:
        print(_turn(args.ask))
        return 0

    print("Assistant ready. Type a question (:quit to exit).")
    while True:
        try:
            text = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not text or text in (":quit", ":exit"):
            break
        print("Assistant:", _turn(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
