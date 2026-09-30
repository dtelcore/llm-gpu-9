"""50-prompt generation check for the finished step-8000 C=512 checkpoint.

Held-out openings are the first sentence of validation stories. The mouse
and Lily lines match earlier samples. Five prompts are instruction-shaped.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.checkpoint import load_checkpoint
from model.gpt import GPTModel
from training.unguided.prober import (
    TINYSTORIES_GENERATE_TEMP,
    TINYSTORIES_MAX_NEW_TOKENS,
    generate_english_continuation,
    looks_like_cabinet_dump,
)

CHECKPOINT = ROOT / "output/checkpoints/english_tinystories_c512_l6_8000_smoke_b2_accum2"
VALID = ROOT / "data/tinystories/text/valid.txt"
OUT = Path(__file__).resolve().parent / "generate_n50.md"
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
    words = story.split()
    dot = story.find(". ")
    first = story[: dot + 1] if dot != -1 else story
    parts = first.split()
    if len(parts) > 18:
        return " ".join(words[:12])
    if len(parts) < 6:
        return " ".join(words[:12])
    return first


def heldout_prompts() -> list[str]:
    stories: list[str] = []
    with VALID.open(encoding="utf-8") as handle:
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


def main() -> None:
    prompts: list[tuple[str, str, int]] = [
        ("compare", MOUSE, SEED),
        ("compare", LILY, SEED),
    ]
    for i, prompt in enumerate(heldout_prompts()):
        prompts.append(("heldout", prompt, SEED + 1 + i))
    for i, prompt in enumerate(OOD):
        prompts.append(("ood", prompt, SEED + 1000 + i))
    if len(prompts) != 50:
        raise RuntimeError(f"expected 50 prompts, got {len(prompts)}")

    gpt_config, params, tokenizer, _, _ = load_checkpoint(str(CHECKPOINT))
    model = GPTModel(gpt_config, params)
    rows = []
    for i, (kind, prompt, seed) in enumerate(prompts, start=1):
        text = generate_english_continuation(
            model,
            tokenizer,
            prompt,
            seed,
            max_new_tokens=TINYSTORIES_MAX_NEW_TOKENS,
            temperature=TINYSTORIES_GENERATE_TEMP,
        )
        names = NAMED.findall(prompt)
        name = names[-1] if names else ""
        low = text.casefold()
        rows.append(
            {
                "i": i,
                "kind": kind,
                "seed": seed,
                "prompt": prompt,
                "text": text,
                "name": name,
                "name_held": bool(name) and name.casefold() in low,
                "restart": "once upon a time" in low,
                "dump": looks_like_cabinet_dump(text),
            }
        )
        print(f"{i:02d}/50 {kind} chars={len(text)} restart={rows[-1]['restart']}", flush=True)

    named = [r for r in rows if r["name"]]
    held = [r for r in rows if r["kind"] == "heldout"]
    ood = [r for r in rows if r["kind"] == "ood"]
    state = json.loads((CHECKPOINT / "state.json").read_text(encoding="utf-8"))
    metrics = json.loads((CHECKPOINT / "metrics.json").read_text(encoding="utf-8"))

    lines = [
        "# 50-prompt generation, step 8000",
        "",
        f"- checkpoint: `{CHECKPOINT.relative_to(ROOT)}`",
        f"- step: {state.get('step')}  epoch: {state.get('epoch')}",
        f"- train loss at save: {metrics.get('loss')}  ppl: {metrics.get('ppl')}",
        f"- temperature: {TINYSTORIES_GENERATE_TEMP}  max_new_tokens: {TINYSTORIES_MAX_NEW_TOKENS}",
        f"- prompts: 2 comparison, {len(held)} held-out validation openings, {len(ood)} instruction-shaped",
        "",
        "## Counts",
        "",
        f"- second story (`Once upon a time` inside the continuation): "
        f"{sum(1 for r in rows if r['restart'])}/50",
        f"- held-out second story: {sum(1 for r in held if r['restart'])}/{len(held)}",
        f"- named character still in the continuation: "
        f"{sum(1 for r in named if r['name_held'])}/{len(named)}",
        f"- cabinet-dump marker: {sum(1 for r in rows if r['dump'])}/50",
        "",
    ]
    for r in rows:
        flags = []
        if r["name"]:
            flags.append(f"name={r['name']} held={r['name_held']}")
        if r["restart"]:
            flags.append("restart")
        if r["dump"]:
            flags.append("cabinet_dump")
        flag = f" ({', '.join(flags)})" if flags else ""
        lines.append(f"## {r['i']:02d}. {r['kind']}{flag}")
        lines.append("")
        lines.append(f"Prompt: {r['prompt']}")
        lines.append("")
        lines.append(r["text"])
        lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
