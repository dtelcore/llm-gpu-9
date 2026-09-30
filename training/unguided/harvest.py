"""Harvest cabinet_retrain.jsonl and emit a bounded perturbed mix."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from paths import DATA_DIR
from tools.make_fact_mix import load_user_facts, write_outputs
from tools.wikidata_to_facts import linked_variants
from training.cabinet_index import (
    _pairs_from_jsonl,
    article_slot_candidates,
    drop_quarantined,
    entity_article_variants,
    normalize_question,
)

RETRAIN_CLASSES = frozenset({"BINDING_ENTITY_SWAP", "TARGET_MISMATCH"})

ANCHOR_QUESTIONS = (
    "What is the capital of France?",
    "Where is Paris?",
    "What did William Cubitt invent?",
    "Who invented penal treadmill?",
    "What is sequential layer streaming?",
    "How do I disable layer streaming?",
    "How do I force layer streaming?",
)

Pair = Tuple[str, str]


def _questionize(normalized_key: str) -> str:
    s = " ".join((normalized_key or "").split())
    if not s:
        return ""
    s = s[0].upper() + s[1:]
    if not s.endswith("?"):
        s += "?"
    return s


def closed_retrain_variants(user: str, assistant: str) -> List[Pair]:
    """Small closed set: inventor active/passive and article form. Not open-ended."""
    out: List[Pair] = []
    seen = {normalize_question(user)}

    def add(question: str, answer: str) -> None:
        key = normalize_question(question)
        if key and key not in seen:
            seen.add(key)
            out.append((question, answer))

    asst = assistant if str(assistant).endswith(".") else f"{assistant}."
    for cand in article_slot_candidates(user):
        add(_questionize(cand), asst)
    for extra_user, extra_asst in linked_variants(user, asst):
        add(extra_user, extra_asst)

    inv_u = re.match(r"(?i)^who invented (.+?)\??$", " ".join((user or "").split()))
    inv_a = re.match(r"^(.+) is credited with inventing (.+)$", asst.rstrip("."))
    if inv_u and inv_a:
        invention = inv_u.group(1).strip().rstrip("?")
        person = inv_a.group(1).strip()
        add(f"What did {person} invent?", asst)
        add(f"Who is credited with inventing {invention}?", asst)
        for variant in entity_article_variants(invention):
            add(f"Who invented {variant}?", asst)
    return out


def load_pair_map(paths: Sequence[Path]) -> dict[str, Pair]:
    mapping: dict[str, Pair] = {}
    for path in paths:
        if not path.is_file():
            continue
        pairs: Iterable[Pair]
        if path.suffix.lower() == ".jsonl":
            pairs = _pairs_from_jsonl(path)
        else:
            pairs = load_user_facts([path])
        for user, assistant in pairs:
            mapping[normalize_question(user)] = (user, assistant)
    return mapping


def read_jsonl_from_offset(path: Path, byte_offset: int) -> tuple[list[dict], int]:
    """Read complete JSONL rows starting at byte_offset. Returns (rows, new_offset)."""
    if not path.is_file():
        return [], 0
    size = path.stat().st_size
    if byte_offset > size:
        byte_offset = size
    rows: list[dict] = []
    with path.open("rb") as handle:
        handle.seek(max(0, int(byte_offset)))
        leftover = b""
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            leftover += chunk
            while True:
                nl = leftover.find(b"\n")
                if nl < 0:
                    break
                line, leftover = leftover[:nl], leftover[nl + 1 :]
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    rec = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    rows.append(rec)
        new_offset = handle.tell() - len(leftover)
    return rows, max(0, new_offset)


@dataclass
class HarvestResult:
    rows: list[dict] = field(default_factory=list)
    byte_offset: int = 0
    ready: bool = False


def harvest_retrain_log(
    path: Path,
    byte_offset: int = 0,
    *,
    min_queue_size: int = 5,
    max_queue_size: int = 40,
    classifications: Optional[frozenset[str]] = None,
    skip_keys: Optional[set[str]] = None,
) -> HarvestResult:
    """Parse new retrain rows.

    Offset advances when the queue is ready, or when every scanned row was
    skipped (consumed/poison). Below-min *new* keys keep the start offset.
    """
    allowed = classifications if classifications is not None else RETRAIN_CLASSES
    banned = skip_keys or set()
    raw, new_offset = read_jsonl_from_offset(path, byte_offset)
    seen: set[str] = set()
    out: list[dict] = []
    skipped = 0
    for rec in raw:
        klass = str(rec.get("classification") or "")
        if klass not in allowed:
            continue
        key = normalize_question(
            str(rec.get("canonical_question") or rec.get("typed_question") or "")
        )
        if not key or key in seen:
            continue
        if key in banned:
            skipped += 1
            continue
        seen.add(key)
        out.append(rec)
        if len(out) >= int(max_queue_size):
            break
    ready = len(out) >= int(min_queue_size)
    if ready or (not out and int(new_offset) != int(byte_offset)):
        committed = new_offset
    else:
        committed = byte_offset
    return HarvestResult(
        rows=out,
        byte_offset=committed,
        ready=ready,
    )


def pair_from_harvest(row: dict, pair_map: dict[str, Pair]) -> Optional[Pair]:
    question = str(row.get("canonical_question") or row.get("typed_question") or "").strip()
    expected = str(row.get("expected") or "").strip()
    if question and expected:
        return question, expected
    hit = pair_map.get(normalize_question(question))
    return hit


def build_bounded_mix(
    harvested: Sequence[dict],
    *,
    broad_jsonl: Path,
    user_facts: Path,
    out_txt: Path,
    out_jsonl: Path,
    retrain_weight: int = 4,
    anchor_weight: int = 3,
    broad_weight: int = 1,
    drop_quarantine: bool = True,
) -> dict:
    """Write a bounded mix: harvested+variants, anchors+user_facts, clean v7 rows."""
    pair_map = load_pair_map([broad_jsonl, user_facts])
    retrain: List[Pair] = []
    for row in harvested:
        pair = pair_from_harvest(row, pair_map)
        if pair is None:
            continue
        retrain.append(pair)
        retrain.extend(closed_retrain_variants(pair[0], pair[1]))
    if drop_quarantine:
        retrain = drop_quarantined(retrain)

    anchors: List[Pair] = []
    for question in ANCHOR_QUESTIONS:
        hit = pair_map.get(normalize_question(question))
        if hit is not None:
            anchors.append(hit)
    user_pairs = load_user_facts([user_facts]) if user_facts.is_file() else []
    if drop_quarantine:
        anchors = drop_quarantined(anchors)
        user_pairs = drop_quarantined(user_pairs)

    broad = list(_pairs_from_jsonl(broad_jsonl)) if broad_jsonl.is_file() else []
    if drop_quarantine:
        broad = drop_quarantined(broad)

    mixed: List[Pair] = []
    mixed.extend(retrain * max(1, int(retrain_weight)))
    mixed.extend(anchors * max(1, int(anchor_weight)))
    mixed.extend(user_pairs * max(1, int(anchor_weight)))
    mixed.extend(broad * max(1, int(broad_weight)))
    n = write_outputs(mixed, out_txt, out_jsonl)
    return {
        "n": n,
        "retrain": len(retrain),
        "anchors": len(anchors),
        "user_facts": len(user_pairs),
        "broad": len(broad),
        "txt": str(out_txt),
        "jsonl": str(out_jsonl),
    }


def write_fresh_recipe(base_recipe: Path, mix_jsonl: Path, dest: Path) -> Path:
    """Copy v7 architecture; point dataset at the new mix. Fresh BPE on train."""
    recipe = json.loads(base_recipe.read_text(encoding="utf-8"))
    dataset = recipe.setdefault("dataset", {})
    dataset["path"] = str(Path(mix_jsonl).as_posix())
    dataset["name"] = "chat_facts_unguided"
    dataset["combine"] = False
    recipe.setdefault("metadata", {})
    recipe["metadata"]["created"] = "unguided-bounded-mix"
    recipe["metadata"]["description"] = (
        "Bounded harvest mix. New BPE. Do not --resume chat_facts_v7 or any other checkpoint."
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
    return dest


def default_v7_jsonl() -> Path:
    return DATA_DIR / "chat_facts_v7.jsonl"


def default_user_facts() -> Path:
    return DATA_DIR / "user_facts.txt"
