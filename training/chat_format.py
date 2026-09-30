"""Role-marked chat formatting shared by corpus tools, REPL, and quality probes.

Training data is one document per line, space-joined turns:

    User: Tell me a story about a cat. Assistant: Once upon a time ...
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
USER_PREFIX = "User: "
ASSISTANT_PREFIX = "Assistant: "
CHAT_STOP_STRINGS = (" User:", "User:")
DEFAULT_CHAT_PROMPT = "User: Tell me a short story about a cat. Assistant:"
DEFAULT_CHAT_SYSTEM = "You are a helpful assistant that speaks simply."
DEFAULT_CHAT_TEMPERATURE = 0.7
DEFAULT_CHAT_TOP_K = 32
DEFAULT_CHAT_TOP_P = 0.9

Turn = Tuple[str, str]

# Cut a completed User: turn leak, then trailing incomplete markers left by
# stop-string token trim (" User", " Use", " Us", " U").
_FULL_USER_LEAK = re.compile(r"(?:^|\s)User:")
_TRAILING_PARTIAL_USER = re.compile(r"(?:\s+User:?|\s+Use|\s+Us|\s+U)$")


def is_chat_model_name(name: Optional[str]) -> bool:
    text = (name or "").lower()
    return "chat_5m" in text or "chat 5m" in text


def format_turn(role: str, text: str) -> str:
    body = " ".join((text or "").split())
    if role == ASSISTANT_ROLE:
        return f"{ASSISTANT_PREFIX}{body}".rstrip()
    return f"{USER_PREFIX}{body}".rstrip()


def format_conversation(
    turns: Sequence[Turn],
    *,
    system: Optional[str] = None,
    pending_user: Optional[str] = None,
    open_assistant: bool = True,
) -> str:
    """Build a single-line prompt matching the training chat format."""
    parts: List[str] = []
    system_text = " ".join((system or "").split())
    if system_text:
        parts.append(system_text)
    for role, text in turns:
        parts.append(format_turn(role, text))
    if pending_user is not None:
        parts.append(format_turn(USER_ROLE, pending_user))
    if open_assistant:
        parts.append(ASSISTANT_PREFIX.rstrip())
    return " ".join(p for p in parts if p)


def _drop_oldest_complete_turn(history: List[Turn]) -> List[Turn]:
    """Drop one User+Assistant pair from the front; never leave a leading Assistant:."""
    if not history:
        return history
    if history[0][0] == ASSISTANT_ROLE:
        return history[1:]
    if len(history) >= 2 and history[1][0] == ASSISTANT_ROLE:
        return history[2:]
    return history[1:]


def sanitize_assistant_reply(
    text: str,
    stop_strings: Optional[Sequence[str]] = None,
) -> str:
    """Strip whitespace and leaked / partial ``User:`` prefixes from a generated turn.

    Stop-string trim can leave `` User``, `` Use``, or a trailing ``User:`` fragment
    that would otherwise be stored in history and nudge the next turn into
    self-prompting.
    """
    if not text:
        return ""
    cleaned = " ".join(text.replace("\n", " ").split())
    leak = _FULL_USER_LEAK.search(cleaned)
    if leak:
        cleaned = cleaned[: leak.start()].rstrip()
    stops = [s for s in (stop_strings or CHAT_STOP_STRINGS) if s]
    for stop in stops:
        idx = cleaned.find(stop)
        if idx >= 0:
            cleaned = cleaned[:idx].rstrip()
    while True:
        stripped = _TRAILING_PARTIAL_USER.sub("", cleaned).rstrip()
        if stripped == cleaned:
            break
        cleaned = stripped
    return cleaned.strip()


def build_chat_prompt_ids(
    tokenizer,
    turns: Sequence[Turn],
    user_text: str,
    *,
    system: Optional[str] = None,
    max_prompt_tokens: int,
) -> Tuple[List[int], str]:
    """Encode a chat prompt, dropping oldest *complete* turns until it fits.

    Always keeps the system prefix and the current ``User: … Assistant:`` opener.
    Never tail-slices the encoded ids (that orphans ``Assistant:`` fragments).
    If the remaining structured prompt is still too long, shrink the current
    user utterance from the front, then drop the system prefix only as a last
    resort.
    """
    budget = max(1, int(max_prompt_tokens))
    history: List[Turn] = list(turns)
    pending = " ".join((user_text or "").split())

    def _encode(hist: Sequence[Turn], user: str, sys: Optional[str]) -> Tuple[List[int], str]:
        text = format_conversation(
            hist, system=sys, pending_user=user, open_assistant=True,
        )
        return list(tokenizer.encode(text)), text

    ids, text = _encode(history, pending, system)
    while len(ids) > budget and history:
        history = _drop_oldest_complete_turn(history)
        ids, text = _encode(history, pending, system)

    if len(ids) <= budget:
        return ids, text

    words = pending.split()
    while len(words) > 1 and len(ids) > budget:
        words = words[1:]
        ids, text = _encode([], " ".join(words), system)

    if len(ids) <= budget:
        return ids, text

    if system:
        ids, text = _encode([], " ".join(words) if words else pending, None)
    if len(ids) > budget:
        ids, text = _encode([], "", None)
    return ids, text


def trim_generated_stop_strings(
    ids: List[int],
    prompt_len: int,
    tokenizer,
    stop_strings: Optional[Sequence[str]],
) -> Tuple[List[int], bool]:
    """Drop completing tokens if the generated suffix contains a stop string.

    Returns ``(ids, hit)``. No-op when tokenizer or stop strings are missing.
    """
    if not stop_strings or tokenizer is None:
        return ids, False
    stops = [s for s in stop_strings if s]
    if not stops:
        return ids, False
    prompt_len = max(0, int(prompt_len))
    generated = ids[prompt_len:]
    if not generated:
        return ids, False
    text = tokenizer.decode(generated)
    hit = next((s for s in stops if s in text), None)
    if hit is None:
        return ids, False
    while generated and hit in tokenizer.decode(generated):
        generated = generated[:-1]
    return ids[:prompt_len] + generated, True
