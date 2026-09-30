"""
training/quality.py

Heuristic generation-quality scores (spelling, punctuation, grammar, semantics)
and sequential inter-quarter comparison / best promotion.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from logging_config import logger
from paths import DATA_DIR, list_quarter_dirs, run_root_for_checkpoint
from training.checkpoint import load_checkpoint, promote_best
from training.chat_format import (
    CHAT_STOP_STRINGS,
    DEFAULT_CHAT_PROMPT,
    is_chat_model_name,
)

DEFAULT_QUALITY_PROMPT = "once upon a"
DEFAULT_CHAT_QUALITY_PROMPT = DEFAULT_CHAT_PROMPT
DEFAULT_QUALITY_MAX_NEW_TOKENS = 256
DEFAULT_QUALITY_TEMPERATURE = 0.6
DEFAULT_QUALITY_TOP_K = 10
DEFAULT_QUALITY_TOP_P = 0.9

CHAT_INSTRUCTION_PROBES: Tuple[Tuple[str, str], ...] = (
    ("story", "User: Tell me a short story about a cat. Assistant:"),
    ("yesno", "User: Can cats fly? Answer yes or no. Assistant:"),
    ("onesentence", "User: Summarize this in one sentence: The cat sat on the mat. Assistant:"),
    ("dialogue", "User: Hi. Assistant: Hello! User: What is your name? Assistant:"),
)

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENT_END_RE = re.compile(r"[.!?]+")
_WORDLIST_CACHE: Optional[set] = None


@dataclass
class QualityScores:
    spelling: float
    punctuation: float
    grammar: float
    semantics: float
    aggregate: float
    turn_format: Optional[float] = None
    instruction_follow: Optional[float] = None
    coherence: Optional[float] = None

    def as_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items() if v is not None}


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _load_wordlist() -> Optional[set]:
    global _WORDLIST_CACHE
    if _WORDLIST_CACHE is not None:
        return _WORDLIST_CACHE or None
    candidates = [
        DATA_DIR / "wordlist.txt",
        DATA_DIR / "words.txt",
        DATA_DIR / "english_words.txt",
    ]
    for path in candidates:
        if path.exists():
            words = set()
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    w = line.strip().lower()
                    if w and w.isalpha():
                        words.add(w)
            _WORDLIST_CACHE = words
            return words
    _WORDLIST_CACHE = set()
    return None


def _tokenize_words(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def score_spelling(text: str) -> float:
    """Higher = fewer garbage/repeated-char tokens; optional wordlist boost."""
    words = _tokenize_words(text)
    if not words:
        return 0.0

    garbage = 0
    for w in words:
        lower = w.lower()
        if len(lower) >= 4 and len(set(lower)) == 1:
            garbage += 1
            continue
        # long runs of the same char (e.g. "helllllo")
        if re.search(r"(.)\1{3,}", lower):
            garbage += 1

    clean_ratio = 1.0 - (garbage / len(words))

    wordlist = _load_wordlist()
    if wordlist:
        known = sum(1 for w in words if w.lower() in wordlist)
        known_ratio = known / len(words)
        return _clamp01(0.4 * clean_ratio + 0.6 * known_ratio)

    # Without a wordlist: prefer alphabetic tokens over digit/symbol soup.
    alpha_chars = sum(1 for c in text if c.isalpha())
    total = max(1, len(text.strip()))
    alpha_ratio = alpha_chars / total
    return _clamp01(0.7 * clean_ratio + 0.3 * alpha_ratio)


def score_punctuation(text: str) -> float:
    """Sentence terminators, spacing around punctuation, quote/paren balance."""
    stripped = text.strip()
    if not stripped:
        return 0.0

    has_terminator = 1.0 if _SENT_END_RE.search(stripped) else 0.0
    # Prefer ending with terminator for multi-word output
    ends_ok = 1.0 if stripped[-1] in ".!?" else (0.5 if len(stripped.split()) < 4 else 0.0)

    # Bad spacing: "word ," or " ." or double spaces around punct
    bad_space = len(re.findall(r"\s+[,.!?;:]", stripped)) + len(re.findall(r"[,.!?;:]{2,}", stripped))
    space_score = _clamp01(1.0 - bad_space / max(1, len(stripped) // 20))

    # Balance quotes / parens
    balance = 1.0
    for open_c, close_c in (('"', '"'), ("'", "'"), ("(", ")"), ("[", "]")):
        if open_c == close_c:
            if stripped.count(open_c) % 2 != 0:
                balance -= 0.15
        else:
            if stripped.count(open_c) != stripped.count(close_c):
                balance -= 0.15
    balance = _clamp01(balance)

    return _clamp01(0.35 * has_terminator + 0.25 * ends_ok + 0.25 * space_score + 0.15 * balance)


def score_grammar(text: str) -> float:
    """Capitalization after terminators, extreme repetition, broken spacing."""
    stripped = text.strip()
    if not stripped:
        return 0.0

    # Capitalization after sentence end
    caps_ok = 0
    caps_total = 0
    for m in _SENT_END_RE.finditer(stripped):
        rest = stripped[m.end() :].lstrip()
        if not rest:
            continue
        caps_total += 1
        if rest[0].isupper():
            caps_ok += 1
    caps_score = (caps_ok / caps_total) if caps_total else (1.0 if stripped[0].isupper() else 0.5)

    words = _tokenize_words(stripped)
    if words:
        # Extreme consecutive word repetition
        reps = 0
        for i in range(1, len(words)):
            if words[i].lower() == words[i - 1].lower():
                reps += 1
        rep_score = _clamp01(1.0 - reps / len(words))
    else:
        rep_score = 0.0

    # Broken spacing (multiple spaces, space at start of "word")
    multi_space = len(re.findall(r"  +", stripped))
    space_score = _clamp01(1.0 - multi_space / max(1, len(stripped) // 30))

    return _clamp01(0.4 * caps_score + 0.35 * rep_score + 0.25 * space_score)


def score_semantics(text: str, prompt: str = "") -> float:
    """Prompt-token overlap, unique-token ratio, non-gibberish entropy."""
    stripped = text.strip()
    if not stripped:
        return 0.0

    # Continuation = text after prompt if prompt is a prefix
    continuation = stripped
    if prompt and stripped.lower().startswith(prompt.lower()):
        continuation = stripped[len(prompt) :].lstrip()

    cont_words = [w.lower() for w in _tokenize_words(continuation)]
    prompt_words = [w.lower() for w in _tokenize_words(prompt)]

    if cont_words:
        unique_ratio = len(set(cont_words)) / len(cont_words)
    else:
        unique_ratio = 0.0

    # Soft prompt stickiness: share of prompt content words that reappear
    if prompt_words and cont_words:
        prompt_set = set(prompt_words)
        overlap = sum(1 for w in cont_words if w in prompt_set) / len(cont_words)
        # Prefer some topical glue but not parroting the whole prompt
        stickiness = 1.0 - abs(overlap - 0.15) / 0.85
        stickiness = _clamp01(stickiness)
    else:
        stickiness = 0.5

    # Character entropy (gibberish tends toward extreme low or high for short noise)
    chars = continuation.lower() if continuation else stripped.lower()
    if chars:
        freq: Dict[str, int] = {}
        for c in chars:
            freq[c] = freq.get(c, 0) + 1
        ent = 0.0
        n = len(chars)
        for count in freq.values():
            p = count / n
            ent -= p * math.log2(p)
        # English-ish text often ~3–4.5 bits; map into 0–1 softly
        entropy_score = _clamp01((ent - 1.5) / 3.0)
    else:
        entropy_score = 0.0

    return _clamp01(0.35 * unique_ratio + 0.30 * stickiness + 0.35 * entropy_score)


def _continuation(text: str, prompt: str = "") -> str:
    stripped = text.strip()
    if prompt and stripped.lower().startswith(prompt.lower()):
        return stripped[len(prompt) :].lstrip()
    return stripped


def score_turn_format(text: str, prompt: str = "") -> float:
    """Role-marker hygiene: assistant continuation without a new User turn."""
    cont = _continuation(text, prompt)
    if not cont.strip():
        return 0.0
    extra_user = 1.0 if re.search(r"(?:^|\s)User:", cont) else 0.0
    starts_user = 1.0 if cont.lstrip().startswith("User:") else 0.0
    repeats_assistant = 1.0 if cont.lstrip().startswith("Assistant:") else 0.0
    n_words = len(_tokenize_words(cont))
    has_content = 1.0 if n_words >= 3 else (0.4 if n_words else 0.0)
    return _clamp01(
        0.45 * has_content
        + 0.35 * (1.0 - extra_user)
        + 0.10 * (1.0 - starts_user)
        + 0.10 * (1.0 - repeats_assistant)
    )


def _instruction_kind(prompt: str) -> str:
    p = prompt.lower()
    if "yes or no" in p or "yes/no" in p:
        return "yesno"
    if "summarize" in p or "one sentence" in p:
        return "onesentence"
    if "what is your name" in p or p.count("user:") >= 2:
        return "dialogue"
    if "story" in p or "tell me" in p:
        return "story"
    return "generic"


def score_instruction_follow(text: str, prompt: str = "") -> float:
    """Prompt-conditioned checks for short instruction / dialogue probes."""
    cont = _continuation(text, prompt)
    words = _tokenize_words(cont)
    if not words:
        return 0.0
    kind = _instruction_kind(prompt)
    extra_user_ok = 0.0 if re.search(r"(?:^|\s)User:", cont) else 1.0
    if kind == "yesno":
        hit = 1.0 if re.search(r"\b(yes|no)\b", cont.lower()) else 0.0
        short = 1.0 if len(words) <= 40 else 0.4
        return _clamp01(0.7 * hit + 0.2 * short + 0.1 * extra_user_ok)
    if kind == "onesentence":
        n_end = len(_SENT_END_RE.findall(cont))
        sent_score = 1.0 if n_end <= 1 else (0.6 if n_end == 2 else 0.2)
        length_ok = 1.0 if 3 <= len(words) <= 40 else 0.4
        return _clamp01(0.5 * sent_score + 0.4 * length_ok + 0.1 * extra_user_ok)
    if kind == "story":
        stop = {"user", "assistant", "tell", "me", "a", "an", "the", "about", "short", "story"}
        prompt_kw = {w.lower() for w in _tokenize_words(prompt) if w.lower() not in stop}
        if prompt_kw:
            overlap = sum(1 for w in words if w.lower() in prompt_kw) / len(prompt_kw)
        else:
            overlap = 0.5
        long_enough = _clamp01(len(words) / 20.0)
        storyish = 1.0 if any(
            w.lower() in {"once", "then", "was", "had", "little", "one"} for w in words
        ) else 0.5
        return _clamp01(0.35 * _clamp01(overlap) + 0.3 * long_enough + 0.25 * storyish + 0.1 * extra_user_ok)
    if kind == "dialogue":
        length_ok = 1.0 if 2 <= len(words) <= 60 else 0.4
        return _clamp01(0.55 * extra_user_ok + 0.45 * length_ok)
    length_ok = _clamp01(len(words) / 8.0)
    return _clamp01(0.5 * extra_user_ok + 0.5 * length_ok)


def resolve_quality_mode(args, config: Optional[Dict] = None) -> str:
    explicit = getattr(args, "quality_mode", None) if args is not None else None
    if explicit in ("story", "chat", "both"):
        return explicit
    name = ""
    if isinstance(config, dict):
        model = config.get("model") or {}
        if isinstance(model, dict):
            name = str(model.get("name", ""))
    return "chat" if is_chat_model_name(name) else "story"


def apply_chat_quality_defaults(args, config: Optional[Dict] = None) -> str:
    """Fill quality_mode / chat probe prompt when the run is chat_5m."""
    mode = resolve_quality_mode(args, config)
    if args is not None:
        args.quality_mode = mode
        if mode in ("chat", "both"):
            if getattr(args, "generate_probe_prompt", None) in (None, "", DEFAULT_QUALITY_PROMPT):
                args.generate_probe_prompt = DEFAULT_CHAT_QUALITY_PROMPT
            quality_prompt = getattr(args, "quality_prompt", None)
            if quality_prompt in (None, "", DEFAULT_QUALITY_PROMPT):
                args.quality_prompt = args.generate_probe_prompt
            if getattr(args, "quality_stop_strings", None) is None:
                args.quality_stop_strings = list(CHAT_STOP_STRINGS)
        elif getattr(args, "quality_stop_strings", None) is None:
            args.quality_stop_strings = None
    return mode


def _quality_weight_map(mode: str, weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    if mode == "chat":
        w = {
            "spelling": 1.0,
            "punctuation": 1.0,
            "grammar": 1.0,
            "semantics": 0.25,
            "turn_format": 1.5,
            "instruction_follow": 1.5,
            "coherence": 1.0,
        }
    elif mode == "both":
        w = {
            "spelling": 1.0,
            "punctuation": 1.0,
            "grammar": 1.0,
            "semantics": 1.0,
            "turn_format": 1.0,
            "instruction_follow": 1.0,
            "coherence": 1.0,
        }
    else:
        w = {
            "spelling": 1.0,
            "punctuation": 1.0,
            "grammar": 1.0,
            "semantics": 1.0,
        }
    if weights:
        w.update({k: float(v) for k, v in weights.items() if k in w})
    return w


def _weighted_quality_aggregate(
    values: Dict[str, Optional[float]],
    mode: str,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    w = _quality_weight_map(mode, weights)
    total_w = 0.0
    weighted = 0.0
    for key, weight in w.items():
        val = values.get(key)
        if val is None:
            continue
        weighted += weight * float(val)
        total_w += weight
    return _clamp01(weighted / (total_w or 1.0))


def score_generation(
    text: str,
    prompt: str = "",
    *,
    weights: Optional[Dict[str, float]] = None,
    mode: str = "story",
) -> QualityScores:
    """Score text on spelling/punctuation/grammar/semantics; aggregate weighted mean.

    ``mode='chat'`` adds turn_format / instruction_follow / coherence and
    de-emphasizes fairy-tale semantics. ``mode='both'`` keeps story scores and
    adds the chat fields.
    """
    spelling = score_spelling(text)
    punctuation = score_punctuation(text)
    grammar = score_grammar(text)
    semantics = score_semantics(text, prompt=prompt)
    coherence = (spelling + punctuation + grammar) / 3.0

    chat_mode = mode in ("chat", "both")
    turn_format = score_turn_format(text, prompt=prompt) if chat_mode else None
    instruction_follow = score_instruction_follow(text, prompt=prompt) if chat_mode else None
    aggregate = _weighted_quality_aggregate(
        {
            "spelling": spelling,
            "punctuation": punctuation,
            "grammar": grammar,
            "semantics": semantics,
            "coherence": coherence if chat_mode else None,
            "turn_format": turn_format,
            "instruction_follow": instruction_follow,
        },
        mode,
        weights,
    )

    return QualityScores(
        spelling=_clamp01(spelling),
        punctuation=_clamp01(punctuation),
        grammar=_clamp01(grammar),
        semantics=_clamp01(semantics),
        aggregate=_clamp01(aggregate),
        turn_format=None if turn_format is None else _clamp01(turn_format),
        instruction_follow=None if instruction_follow is None else _clamp01(instruction_follow),
        coherence=None if not chat_mode else _clamp01(coherence),
    )


def _delta_label(curr: float, prev: Optional[float]) -> str:
    if prev is None:
        return "baseline"
    diff = curr - prev
    if abs(diff) < 0.02:
        return "stable"
    return "improving" if diff > 0 else "regressing"


def compare_quarters(
    run_dir: str,
    *,
    prompt: str = DEFAULT_QUALITY_PROMPT,
    max_new_tokens: int = DEFAULT_QUALITY_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_QUALITY_TEMPERATURE,
    top_k: Optional[int] = DEFAULT_QUALITY_TOP_K,
    top_p: Optional[float] = DEFAULT_QUALITY_TOP_P,
    seed: int = 42,
    weights: Optional[Dict[str, float]] = None,
    interactive_promote: bool = True,
    set_best: Optional[str] = None,
    mode: str = "story",
    stop_strings: Optional[Sequence[str]] = None,
) -> List[Dict]:
    """Sequential generate+score across quarter_* checkpoints; optionally promote best."""
    from model.gpt import GPTModel

    root = run_root_for_checkpoint(run_dir)
    quarters = list_quarter_dirs(root)
    if not quarters:
        print(f"No quarter_* checkpoints found under {root}")
        logger.warning("compare_quarters: no quarters under %s", root)
        return []

    results: List[Dict] = []
    prev_agg: Optional[float] = None
    chat_mode = mode in ("chat", "both")
    stops = list(stop_strings) if stop_strings else (list(CHAT_STOP_STRINGS) if chat_mode else None)

    print("=" * 70)
    print(f"QUALITY TRIAL — sequential generation across quarters in {root}")
    print(
        f"mode={mode} | prompt={prompt!r} | tokens={max_new_tokens} | "
        f"temp={temperature} | top_k={top_k} | top_p={top_p}"
    )
    print("=" * 70)

    for qdir in quarters:
        gpt_config, params, tokenizer, _, state = load_checkpoint(str(qdir))
        model = GPTModel(gpt_config, params)
        step = int(state.get("step", 0))

        prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            print(f"[{qdir.name}] prompt encodes empty; skipping")
            continue

        rng = np.random.default_rng(seed)
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rng=rng,
            tokenizer=tokenizer if stops else None,
            stop_strings=stops,
        )
        text = tokenizer.decode(generated_ids)
        scores = score_generation(text, prompt=prompt, weights=weights, mode=mode)
        if chat_mode:
            probe_follow: List[float] = []
            probe_tokens = min(80, max_new_tokens)
            for _kind, probe_prompt in CHAT_INSTRUCTION_PROBES:
                pids = tokenizer.encode(probe_prompt)
                if not pids:
                    continue
                prng = np.random.default_rng(seed)
                gids = model.generate(
                    pids,
                    max_new_tokens=probe_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    rng=prng,
                    tokenizer=tokenizer if stops else None,
                    stop_strings=stops,
                )
                ptext = tokenizer.decode(gids)
                probe_follow.append(
                    score_instruction_follow(ptext, prompt=probe_prompt)
                )
            if probe_follow:
                scores.instruction_follow = float(sum(probe_follow) / len(probe_follow))
                scores.aggregate = _weighted_quality_aggregate(
                    scores.as_dict(), mode, weights,
                )
        trend = _delta_label(scores.aggregate, prev_agg)
        prev_agg = scores.aggregate

        row = {
            "name": qdir.name,
            "path": str(qdir),
            "step": step,
            "text": text,
            "scores": scores.as_dict(),
            "trend": trend,
        }
        results.append(row)

        print("\n" + "-" * 70)
        print(f"{qdir.name} | step={step:,} | aggregate={scores.aggregate:.3f} ({trend})")
        extra = ""
        if scores.turn_format is not None:
            extra = (
                f"  turn_format={scores.turn_format:.3f}  "
                f"instruction_follow={scores.instruction_follow:.3f}  "
                f"coherence={scores.coherence:.3f}"
            )
        print(
            f"  spelling={scores.spelling:.3f}  punctuation={scores.punctuation:.3f}  "
            f"grammar={scores.grammar:.3f}  semantics={scores.semantics:.3f}"
        )
        if extra:
            print(extra)
        print(text[:500] + ("…" if len(text) > 500 else ""))
        logger.info(
            "[quality] quarter=%s step=%s aggregate=%.4f spelling=%.4f punctuation=%.4f "
            "grammar=%.4f semantics=%.4f trend=%s",
            qdir.name, step, scores.aggregate, scores.spelling, scores.punctuation,
            scores.grammar, scores.semantics, trend,
        )

    if not results:
        return results

    if set_best:
        chosen = _resolve_set_best(root, set_best, results)
        if chosen:
            _do_promote(root, chosen, results)
        return results

    if interactive_promote:
        _prompt_promote(root, results)

    return results


def _resolve_set_best(root: Path, set_best: str, results: Sequence[Dict]) -> Optional[Path]:
    name = set_best.strip()
    by_name = {r["name"]: Path(r["path"]) for r in results}
    if name in by_name:
        return by_name[name]
    candidate = Path(name)
    if not candidate.is_absolute():
        candidate = root / name
    if (candidate / "config.json").exists():
        return candidate
    print(f"--set-best '{set_best}' not found among quarters; skipping promote")
    return None


def _do_promote(root: Path, source: Path, results: Sequence[Dict]) -> None:
    match = next((r for r in results if Path(r["path"]) == source or r["name"] == source.name), None)
    meta = {
        "step": match["step"] if match else None,
        "scores": match["scores"] if match else None,
        "trend": match["trend"] if match else None,
    }
    best = promote_best(root, source, meta=meta)
    print(f"\nPromoted '{source.name}' -> {best}")


def _prompt_promote(root: Path, results: Sequence[Dict]) -> None:
    print("\n" + "=" * 70)
    print("Promote one quarter as best/? (or Enter to skip)")
    for i, r in enumerate(results, 1):
        s = r["scores"]
        print(f"  {i}. {r['name']} (step={r['step']:,}, agg={s['aggregate']:.3f}, {r['trend']})")
    try:
        choice = input("Select number to promote [default=skip]: ").strip()
    except EOFError:
        return
    if not choice:
        print("Skipped best promotion.")
        return
    if choice.isdigit() and 1 <= int(choice) <= len(results):
        _do_promote(root, Path(results[int(choice) - 1]["path"]), results)
        return
    print(f"'{choice}' not recognized; skipped.")


def parse_quality_weights(raw: Optional[str]) -> Optional[Dict[str, float]]:
    """Parse 'spelling=1,punctuation=1,grammar=1,semantics=1' into a weight dict."""
    if not raw:
        return None
    out: Dict[str, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid quality weight fragment: {part!r}")
        key, val = part.split("=", 1)
        out[key.strip()] = float(val.strip())
    return out
