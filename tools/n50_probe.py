"""50-prompt acceptance probe for gpu9_max512 checkpoints.

Mirrors run_n50_template.py criteria (restart / name-held / dump markers) and
adds tool-format checks for the instruction prompts:

    2  comparison  (brave mouse, Lily)
    43 held-out    (first-sentence prefixes of data/tinystories/text/valid.txt)
    5  instruction (brave mouse, Who are you, capital France, capital Atlantis, 17 + 4)

Pass guidance (TinyStories-scale chat model, temp 0.8, 160 new tokens):
  - second-story restart rate  < 25%  (story stays one story)
  - named-character held       > 50%  (of prompts that name someone)
  - cabinet-dump markers       = 0
  - calc prompt "17 + 4" mentions 21, capital France mentions Paris

Writes output/n50_<checkpoint-stem>.md and prints PASS/REVIEW per criterion.

Usage:
    python tools/n50_probe.py --checkpoint output/checkpoints/gpu9_max512
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.gpt import GPTModel  # noqa: E402
from training.checkpoint import load_checkpoint  # noqa: E402
from training.unguided.prober import (  # noqa: E402
    TINYSTORIES_GENERATE_TEMP,
    TINYSTORIES_MAX_NEW_TOKENS,
    generate_english_continuation,
    looks_like_cabinet_dump,
)

VALID = ROOT / "data/tinystories/text/valid.txt"
SEED = 42
N_HELDOUT = 43

OOD = (
    "Tell me a short story about a brave mouse.",
    "Who are you?",
    "What city is the capital of France?",
    "What is the capital of Atlantis?",
    "What is 17 + 4?",
)
MOUSE = "Once upon a time there was a brave little mouse named"
LILY = "Once upon a time there was a little girl named Lily who found a"
NAMED = re.compile(r"\bnamed\s+([A-Z][a-z]+)")


def story_prefix(story: str) -> str:
    dot = story.find(". ")
    first = story[: dot + 1] if dot != -1 else story
    parts = first.split()
    words = story.split()
    if len(parts) > 18 or len(parts) < 6:
        return " ".join(words[:12])
    return first


def heldout_prompts() -> list[str]:
    stories: list[str] = []
    with VALID.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line.lower().startswith("once upon a time"):
                stories.append(line)
    rng = random.Random(SEED)
    rng.shuffle(stories)
    picked: list[str] = []
    seen = {MOUSE.casefold(), LILY.casefold()}
    for story in stories:
        prefix = story_prefix(story)
        key = prefix.casefold()
        if key in seen:
            continue
        seen.add(key)
        picked.append(prefix)
        if len(picked) >= N_HELDOUT:
            break
    if len(picked) < N_HELDOUT:
        raise RuntimeError(f"only {len(picked)} held-out prefixes")
    return picked


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="50-prompt acceptance probe")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    ckpt = Path(args.checkpoint)

    prompts: list[tuple[str, str, int]] = [
        ("compare", MOUSE, SEED),
        ("compare", LILY, SEED),
    ]
    for i, prompt in enumerate(heldout_prompts()):
        prompts.append(("heldout", prompt, SEED + 1 + i))
    for i, prompt in enumerate(OOD):
        prompts.append(("ood", prompt, SEED + 1000 + i))
    assert len(prompts) == 50, f"expected 50 prompts, got {len(prompts)}"

    gpt_config, params, tokenizer, _, _ = load_checkpoint(str(ckpt))
    model = GPTModel(gpt_config, params)

    rows = []
    for i, (kind, prompt, seed) in enumerate(prompts, start=1):
        text = generate_english_continuation(
            model, tokenizer, prompt, seed,
            max_new_tokens=TINYSTORIES_MAX_NEW_TOKENS,
            temperature=TINYSTORIES_GENERATE_TEMP,
        )
        names = NAMED.findall(prompt)
        name = names[-1] if names else ""
        low = text.casefold()
        rows.append({
            "i": i, "kind": kind, "seed": seed, "prompt": prompt, "text": text,
            "name": name,
            "name_held": bool(name) and name.casefold() in low,
            "restart": "once upon a time" in low,
            "dump": looks_like_cabinet_dump(text),
        })
        print(f"{i:02d}/50 {kind} chars={len(text)} restart={rows[-1]['restart']}", flush=True)

    named = [r for r in rows if r["name"]]
    held = [r for r in rows if r["kind"] == "heldout"]
    n_restart = sum(1 for r in rows if r["restart"])
    n_restart_held = sum(1 for r in held if r["restart"])
    n_held = sum(1 for r in named if r["name_held"])
    n_dump = sum(1 for r in rows if r["dump"])
    calc_ok = any("21" in r["text"] for r in rows if r["prompt"] == "What is 17 + 4?")
    paris_ok = any("paris" in r["text"].casefold()
                   for r in rows if r["prompt"] == "What city is the capital of France?")

    checks = [
        ("restart<25%", n_restart / 50 < 0.25, f"{n_restart}/50 (held-out {n_restart_held}/{len(held)})"),
        ("named-held>50%", (n_held / len(named) > 0.5) if named else False,
         f"{n_held}/{len(named)}"),
        ("dumps=0", n_dump == 0, f"{n_dump}/50"),
        ("calc 17+4=21", calc_ok, str(calc_ok)),
        ("capital=Paris", paris_ok, str(paris_ok)),
    ]
    verdict = "PASS" if all(ok for _, ok, _ in checks) else "REVIEW"

    import json as _json
    state = _json.loads((ckpt / "state.json").read_text(encoding="utf-8"))
    metrics = _json.loads((ckpt / "metrics.json").read_text(encoding="utf-8"))
    lines = [
        f"# N50 probe: {ckpt.name} -> {verdict}",
        "",
        f"- step: {state.get('step')}  train loss: {metrics.get('loss')}  ppl: {metrics.get('ppl')}",
        f"- temp: {TINYSTORIES_GENERATE_TEMP}  max_new: {TINYSTORIES_MAX_NEW_TOKENS}",
        "",
        "## Checks",
        "",
    ]
    for label, ok, detail in checks:
        lines.append(f"- [{'x' if ok else ' '}] {label}: {detail}")
    lines.append("")
    for r in rows:
        flags = []
        if r["name"]:
            flags.append(f"name={r['name']} held={r['name_held']}")
        if r["restart"]:
            flags.append("restart")
        if r["dump"]:
            flags.append("cabinet_dump")
        flag = f" ({', '.join(flags)})" if flags else ""
        lines += [f"## {r['i']:02d}. {r['kind']}{flag}", "", f"Prompt: {r['prompt']}", "", r["text"], ""]

    out = Path(args.out) if args.out else (ROOT / "output" / f"n50_{ckpt.name}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nVerdict: {verdict}")
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    print(f"wrote {out}", flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
