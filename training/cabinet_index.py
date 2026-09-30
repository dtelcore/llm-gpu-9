"""Normalized exact lookup for the chat fact cabinet.

The index is a dictionary of unique User questions. Hits return the
*trained* ``User: … Assistant:`` generate prompt, not the REPL typing.
No fuzzy or substring matching.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, Iterator, List, Optional, Union

from training.chat_format import ASSISTANT_PREFIX, USER_PREFIX, format_conversation
from paths import DATA_DIR

DEFAULT_FACTS_PATH = DATA_DIR / "chat_facts.jsonl"
DEFAULT_ALIASES_PATH = DATA_DIR / "cabinet_aliases.json"
DEFAULT_QUARANTINE_PATH = DATA_DIR / "cabinet_quarantine.json"
RELATED_LIMIT = 5
_ARTICLES = ("the", "a", "an")

_CAPITAL_USER = re.compile(r"^what is the capital of (.+)$")
_CAPITAL_ASST = re.compile(r"^the capital of (.+) is (.+)$")
_WHERE_CITY_ASST = re.compile(r"^(.+) is the capital of (.+)$")
_INVENTOR_USER = re.compile(r"^who invented (.+)$")
_INVENTOR_ASST = re.compile(r"^(.+) is credited with inventing (.+)$")
_ATOMIC_USER = re.compile(r"^what is the atomic number of (.+)$")
_ATOMIC_ASST = re.compile(r"^the atomic number of (.+) is (\d+)$")
_BIRTH_USER = re.compile(r"^when was (.+) born$")
_TELL_USER = re.compile(r"^tell me about (.+)$")
_POSSESS_USER = re.compile(r"^what is (.+)'s capital$")
_COUNTRY_OF_USER = re.compile(r"^what country is (.+) in$")
_WHERE_USER = re.compile(r"^where is (.+)$")


def _bare_entity(text: str) -> str:
    s = " ".join((text or "").split()).strip(" .")
    return s.casefold()


def extract_entities(user: str, assistant: str) -> FrozenSet[str]:
    """Closed-world slots from known trained templates. Empty if unrecognized."""
    u = normalize_question(user)
    a = _bare_entity(assistant)
    found = set()

    m = _CAPITAL_USER.match(u)
    if m:
        found.add(_bare_entity(m.group(1)))
    m = _CAPITAL_ASST.match(a)
    if m:
        found.add(_bare_entity(m.group(1)))
        found.add(_bare_entity(m.group(2)))
    m = _WHERE_CITY_ASST.match(a)
    if m and "capital of" in a:
        found.add(_bare_entity(m.group(1)))
        found.add(_bare_entity(m.group(2)))

    m = _INVENTOR_USER.match(u)
    if m:
        found.add(_bare_entity(m.group(1)))
    m = _INVENTOR_ASST.match(a)
    if m:
        found.add(_bare_entity(m.group(1)))
        found.add(_bare_entity(m.group(2)))

    m = _ATOMIC_USER.match(u)
    if m:
        found.add(_bare_entity(m.group(1)))
    m = _ATOMIC_ASST.match(a)
    if m:
        found.add(_bare_entity(m.group(1)))
        found.add(_bare_entity(m.group(2)))

    m = _BIRTH_USER.match(u)
    if m:
        found.add(_bare_entity(m.group(1)))
    m = _TELL_USER.match(u)
    if m:
        found.add(_bare_entity(m.group(1)))
    m = _POSSESS_USER.match(u)
    if m:
        found.add(_bare_entity(m.group(1)))
    m = _COUNTRY_OF_USER.match(u)
    if m:
        found.add(_bare_entity(m.group(1)))
    m = _WHERE_USER.match(u)
    if m:
        found.add(_bare_entity(m.group(1)))

    return frozenset(e for e in found if e)


def template_keys_for(entity: str) -> List[str]:
    """Normalized questions that count as a known follow-up for one entity."""
    e = _bare_entity(entity)
    if not e:
        return []
    return [
        normalize_question(f"where is {e}"),
        normalize_question(f"what country is {e} in"),
        normalize_question(f"what is the capital of {e}"),
        normalize_question(f"what is {e}'s capital"),
        normalize_question(f"who invented {e}"),
        normalize_question(f"who is credited with inventing {e}"),
        normalize_question(f"what did {e} invent"),
        normalize_question(f"what is the atomic number of {e}"),
        normalize_question(f"which element has atomic number {e}"),
        normalize_question(f"tell me about {e}"),
        normalize_question(f"when was {e} born"),
        normalize_question(f"what is {e}"),
    ]


@dataclass(frozen=True)
class CabinetFact:
    """One unique cabinet Q&A plus the canonical generate prompt."""

    user: str
    assistant: str
    generate_prompt: str
    key: str
    source: str = "trained"


def normalize_question(text: str) -> str:
    """Canonical key for exact cabinet lookup.

    - Strip leading/trailing whitespace and collapse internal whitespace
    - Strip a leading ``User:`` if the user typed it
    - Drop a trailing ``Assistant: …`` paste if present
    - Drop an optional trailing ``?``
    - Casefold
    """
    s = " ".join((text or "").split())
    if not s:
        return ""
    lower = s.casefold()
    if lower.startswith("user:"):
        s = s.split(":", 1)[1].strip()
        s = " ".join(s.split())
        lower = s.casefold()
    marker = ASSISTANT_PREFIX.rstrip().casefold()
    idx = lower.find(marker)
    if idx >= 0:
        s = s[:idx].rstrip()
        s = " ".join(s.split())
    if s.endswith("?"):
        s = s[:-1].rstrip()
        s = " ".join(s.split())
    return s.casefold()


def generate_prompt_for(user: str) -> str:
    """Stored generate string: ``User: {trained question} Assistant:``."""
    return format_conversation([], pending_user=user, open_assistant=True)


def entity_article_variants(entity: str) -> List[str]:
    """Entity with and without a leading the/a/an."""
    e = " ".join((entity or "").split()).strip(" .")
    if not e:
        return []
    out: List[str] = []
    seen = set()
    low = e.casefold()
    bare = e
    for art in _ARTICLES:
        prefix = art + " "
        if low.startswith(prefix):
            bare = e[len(prefix):].lstrip()
            break
    candidates = [bare, e]
    for art in _ARTICLES:
        candidates.append(f"{art} {bare}")
    for cand in candidates:
        cand = " ".join(cand.split())
        if not cand:
            continue
        key = cand.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


_ARTICLE_SLOT_PATTERNS = (
    (re.compile(r"^who invented (.+)$"), "who invented {e}"),
    (re.compile(r"^who is credited with inventing (.+)$"), "who is credited with inventing {e}"),
    (re.compile(r"^what is the capital of (.+)$"), "what is the capital of {e}"),
)


def article_slot_candidates(text: str) -> List[str]:
    """Normalized keys for inventor/capital templates with/without a leading article.

    Only recognized slots. Unrelated questions such as ``What is unobtanium?``
    yield an empty list (no fuzzy rewrite).
    """
    key = normalize_question(text)
    if not key:
        return []
    out: List[str] = []
    seen = set()
    for rx, fmt in _ARTICLE_SLOT_PATTERNS:
        m = rx.match(key)
        if not m:
            continue
        for variant in entity_article_variants(m.group(1)):
            cand = normalize_question(fmt.format(e=variant))
            if cand and cand not in seen:
                seen.add(cand)
                out.append(cand)
        break
    return out


def load_quarantine_keys(path: Optional[Union[str, Path]] = None) -> FrozenSet[str]:
    """Normalized questions that must not enter the trained cabinet."""
    src = Path(path) if path is not None else DEFAULT_QUARANTINE_PATH
    if not src.is_file():
        return frozenset()
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return frozenset()
    if not isinstance(data, list):
        return frozenset()
    return frozenset(normalize_question(str(item)) for item in data if str(item).strip())


def drop_quarantined(
    pairs: Iterable[tuple],
    keys: Optional[FrozenSet[str]] = None,
) -> List[tuple]:
    banned = keys if keys is not None else load_quarantine_keys()
    if not banned:
        return list(pairs)
    return [(u, a) for u, a in pairs if normalize_question(u) not in banned]


_JSONL_DECODER = json.JSONDecoder()


def _records_from_jsonl_line(line: str) -> Iterator[dict]:
    """Yield JSON objects on a line. Glued records (missing newline) still parse."""
    i = 0
    n = len(line)
    while i < n:
        while i < n and line[i].isspace():
            i += 1
        if i >= n:
            return
        try:
            rec, end = _JSONL_DECODER.raw_decode(line, i)
        except json.JSONDecodeError:
            return
        if isinstance(rec, dict):
            yield rec
        i = end


def _pairs_from_jsonl(path: Path) -> Iterator[tuple]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            for rec in _records_from_jsonl_line(line):
                user = (rec.get("query") or {}).get("user") or rec.get("user")
                assistant = (rec.get("response") or {}).get("assistant") or rec.get("assistant")
                if user and assistant:
                    yield str(user).strip(), str(assistant).strip()


def _pairs_from_txt(path: Path) -> Iterator[tuple]:
    prefix_u = USER_PREFIX.rstrip()
    prefix_a = ASSISTANT_PREFIX.rstrip()
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            text = " ".join(raw.split())
            if prefix_u not in text or prefix_a not in text:
                continue
            after = text.split(USER_PREFIX, 1)[-1] if USER_PREFIX in text else text.split("User:", 1)[-1]
            if ASSISTANT_PREFIX not in after and "Assistant:" not in after:
                continue
            if ASSISTANT_PREFIX in after:
                user, assistant = after.split(ASSISTANT_PREFIX, 1)
            else:
                user, assistant = after.split("Assistant:", 1)
            user, assistant = user.strip(), assistant.strip()
            if user and assistant:
                yield user, assistant


class CabinetIndex:
    """Exact-normalized map of trained questions → CabinetFact."""

    def __init__(self, facts: Optional[Dict[str, CabinetFact]] = None):
        self._facts: Dict[str, CabinetFact] = dict(facts or {})
        self._aliases: Dict[str, CabinetFact] = {}
        self._file_aliases: Dict[str, str] = {}
        self._quarantine: FrozenSet[str] = frozenset()
        self._entity_facts: Dict[str, List[CabinetFact]] = defaultdict(list)
        self._fact_entities: Dict[str, FrozenSet[str]] = {}
        for fact in self._facts.values():
            if fact.source == "trained":
                self._index_entities(fact)

    def __len__(self) -> int:
        return len(self._facts)

    def set_quarantine(self, keys: Iterable[str]) -> None:
        self._quarantine = frozenset(normalize_question(k) for k in keys if k)

    def lookup(self, text: str, *, source: Optional[str] = None) -> Optional[CabinetFact]:
        key = normalize_question(text)
        if not key:
            return None
        fact = self._facts.get(key)
        if fact is not None and (source is None or fact.source == source):
            return fact
        if source == "learned":
            aliased = self._aliases.get(key)
            return aliased if aliased is not None and aliased.source == "learned" else None
        if source == "trained":
            resolved = self.resolve_trained(text)
            return resolved[0] if resolved else None
        return self._aliases.get(key)

    def resolve_trained(self, text: str) -> Optional[tuple]:
        """Collision-safe trained hit: (fact, match_type) or None.

        match_type is trained_exact, trained_alias, or trained_article.
        Learned rows that occupy the same typed key do not win.
        """
        key = normalize_question(text)
        if not key or key in self._quarantine:
            return None
        fact = self._facts.get(key)
        if fact is not None and fact.source == "trained":
            return fact, "trained_exact"

        target = self._file_aliases.get(key)
        if target and target not in self._quarantine:
            fact = self._facts.get(target)
            if fact is not None and fact.source == "trained":
                return fact, "trained_alias"

        aliased = self._aliases.get(key)
        if aliased is not None and aliased.source == "trained":
            return aliased, "trained_alias"

        hits: Dict[str, CabinetFact] = {}
        for cand in article_slot_candidates(text):
            if cand in self._quarantine:
                continue
            found = self._facts.get(cand)
            if found is not None and found.source == "trained":
                hits[found.key] = found
        if len(hits) == 1:
            return next(iter(hits.values())), "trained_article"
        return None

    def resolve_learned(self, text: str) -> Optional[tuple]:
        """Learned exact or in-memory topic alias. Trained keys are not returned."""
        key = normalize_question(text)
        if not key:
            return None
        fact = self._facts.get(key)
        if fact is not None and fact.source == "learned":
            return fact, "learned_exact"
        aliased = self._aliases.get(key)
        if aliased is not None and aliased.source == "learned":
            return aliased, "learned_topic"
        return None

    def add(self, user: str, assistant: str, *, source: str = "trained") -> Optional[CabinetFact]:
        key = normalize_question(user)
        if not key:
            return None
        if source == "trained" and key in self._quarantine:
            return None
        existing = self._facts.get(key)
        if existing is not None:
            if existing.assistant != assistant.strip():
                return None
            return existing
        fact = CabinetFact(
            user=user.strip(),
            assistant=assistant.strip(),
            generate_prompt=generate_prompt_for(user.strip()),
            key=key,
            source=source,
        )
        self._facts[key] = fact
        if source == "trained":
            self._index_entities(fact)
        return fact

    def add_alias(self, text: str, fact: CabinetFact) -> bool:
        """In-memory lookup alias. Not a second stored question."""
        key = normalize_question(text)
        if not key or key in self._facts or key in self._aliases:
            return False
        self._aliases[key] = fact
        return True

    def unique_facts(self) -> Iterator[CabinetFact]:
        yield from self._facts.values()

    def _index_entities(self, fact: CabinetFact) -> None:
        ents = extract_entities(fact.user, fact.assistant)
        self._fact_entities[fact.key] = ents
        for ent in ents:
            bucket = self._entity_facts[ent]
            if fact not in bucket:
                bucket.append(fact)

    def entities_of(self, fact: CabinetFact) -> FrozenSet[str]:
        return self._fact_entities.get(fact.key, frozenset())

    def known_entities(self) -> FrozenSet[str]:
        return frozenset(self._entity_facts)

    def entities_mentioned(self, text: str) -> FrozenSet[str]:
        """Entities from the closed set that appear as whole tokens in text."""
        key = normalize_question(text)
        if not key:
            return frozenset()
        tokens = f" {key} "
        found = set()
        for ent in self._entity_facts:
            pad = f" {ent} "
            if pad in tokens or key == ent:
                found.add(ent)
        return frozenset(found)

    def facts_for_entities(self, entities: Iterable[str]) -> List[CabinetFact]:
        seen: Dict[str, CabinetFact] = {}
        for ent in entities:
            for fact in self._entity_facts.get(_bare_entity(ent), []):
                if fact.source == "trained":
                    seen[fact.key] = fact
        return list(seen.values())

    def related_prompts(
        self,
        entities: Iterable[str],
        *,
        limit: int = RELATED_LIMIT,
        exclude_key: Optional[str] = None,
    ) -> List[str]:
        out: List[str] = []
        for fact in self.facts_for_entities(entities):
            if exclude_key and fact.key == exclude_key:
                continue
            out.append(fact.user)
            if len(out) >= limit:
                break
        return out

    def related_template_hits(self, text: str, extra_entities: Iterable[str] = ()) -> List[CabinetFact]:
        """Trained facts whose user is a known template for a mentioned/session entity."""
        resolved = self.resolve_trained(text)
        if resolved is not None:
            return [resolved[0]]
        typed = normalize_question(text)
        if not typed:
            return []
        ents = set(self.entities_mentioned(text))
        ents.update(_bare_entity(e) for e in extra_entities if e)
        cands = set(article_slot_candidates(text))
        cands.add(typed)
        seen: Dict[str, CabinetFact] = {}
        for ent in ents:
            for tmpl in template_keys_for(ent):
                if tmpl not in cands:
                    continue
                fact = self._facts.get(tmpl)
                if fact is not None and fact.source == "trained":
                    seen[fact.key] = fact
        return list(seen.values())

    def load_file_aliases(self, path: Optional[Union[str, Path]] = None) -> int:
        """Load explicit trained aliases. Missing or ambiguous targets are skipped."""
        src = Path(path) if path is not None else DEFAULT_ALIASES_PATH
        if not src.is_file():
            return 0
        try:
            raw = json.loads(src.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 0
        if not isinstance(raw, dict):
            return 0
        added = 0
        for src_q, tgt_q in raw.items():
            ns = normalize_question(str(src_q))
            nt = normalize_question(str(tgt_q))
            if not ns or not nt or nt in self._quarantine:
                continue
            fact = self._facts.get(nt)
            if fact is None or fact.source != "trained":
                continue
            existing = self._file_aliases.get(ns)
            if existing is not None and existing != nt:
                continue
            self._file_aliases[ns] = nt
            added += 1
        return added


def _pairs_from_path(path: Path) -> Iterator[tuple]:
    if path.suffix.lower() == ".jsonl":
        return _pairs_from_jsonl(path)
    return _pairs_from_txt(path)


def _jsonl_line(user: str, assistant: str) -> str:
    return json.dumps(
        {"query": {"user": user}, "response": {"assistant": assistant}},
        ensure_ascii=False,
    )


def load_cabinet(
    path: Union[str, Path],
    *,
    source: str = "trained",
    quarantine_path: Optional[Union[str, Path]] = None,
    aliases_path: Optional[Union[str, Path]] = None,
) -> CabinetIndex:
    """Load unique facts from JSONL (or native ``User:/Assistant:`` txt)."""
    src = Path(path)
    index = CabinetIndex()
    if source == "trained":
        index.set_quarantine(load_quarantine_keys(quarantine_path))
    if not src.is_file():
        raise FileNotFoundError(f"Cabinet facts not found: {src}")
    for user, assistant in _pairs_from_path(src):
        index.add(user, assistant, source=source)
    if source == "trained":
        index.load_file_aliases(aliases_path)
    return index


def merge_cabinet(
    index: CabinetIndex,
    path: Union[str, Path],
    *,
    source: str = "learned",
) -> int:
    """Load extra facts into an existing index. Returns how many keys were new."""
    src = Path(path)
    if not src.is_file():
        return 0
    before = len(index)
    for user, assistant in _pairs_from_path(src):
        index.add(user, assistant, source=source)
    return len(index) - before


def remember(
    index: CabinetIndex,
    path: Union[str, Path],
    user: str,
    assistant: str,
    *,
    source: str = "learned",
) -> Optional[CabinetFact]:
    """Index a new Q&A and append JSONL. Duplicates and conflicts do not write."""
    if not (user or "").strip() or not (assistant or "").strip():
        return None
    trained = index.resolve_trained(user)
    if trained is not None:
        return trained[0]
    existing = index.lookup(user)
    if existing is not None:
        return existing
    fact = index.add(user, assistant, source=source)
    if fact is None:
        return None
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size:
        with dest.open("rb+") as handle:
            handle.seek(-1, 2)
            if handle.read(1) != b"\n":
                handle.write(b"\n")
    with dest.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(_jsonl_line(fact.user, fact.assistant) + "\n")
    return fact
