#!/usr/bin/env python3
"""
tools/wikidata_to_facts.py

Pull clean atomic facts from Wikidata and write native User:/Assistant: lines
for make_fact_mix.py (loaded from data/facts/*.txt).

Does not concatenate data/train.txt or chat_train.txt. Keep LIMIT modest:
this cabinet memorizes tens-to-hundreds of repeated facts, not thousands of
thin ones. Same User: line with two different answers is dropped entirely.

Network: SPARQLWrapper if installed, else stdlib urllib. Be polite (--sleep).
Optional: `pip install SPARQLWrapper` (pulls rdflib). Not required.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from version import __version__
from tools.make_chat_trainset import wrap_native

ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = (
    f"Apple-MLX-llm-gpu-8/{__version__} "
    "(local research; https://github.com/dtelcore/Apple-MLX-llm-gpu-8)"
)

_QID = re.compile(r"^Q\d+$")
_LIMIT_RE = re.compile(r"LIMIT\s+\d+", re.IGNORECASE)

Pair = Tuple[str, str]
QueryFn = Callable[[str], List[dict]]

QUERIES: Dict[str, str] = {
    "capitals": """
    SELECT ?countryLabel ?capitalLabel WHERE {
      ?country wdt:P31 wd:Q6256 .
      ?country wdt:P36 ?capital .
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    }
    LIMIT 80
    """,
    "inventors": """
    SELECT ?inventionLabel ?inventorLabel WHERE {
      ?invention wdt:P61 ?inventor .
      ?invention wdt:P31/wdt:P279* wd:Q11019 .
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    }
    LIMIT 60
    """,
    "birth_years": """
    SELECT ?personLabel (YEAR(?birth) AS ?year) WHERE {
      ?person wdt:P31 wd:Q5 .
      ?person wdt:P569 ?birth .
      ?person wdt:P106 wd:Q36180 .
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    }
    LIMIT 50
    """,
    "elements": """
    SELECT ?elementLabel ?number WHERE {
      ?element wdt:P31 wd:Q11344 .
      ?element wdt:P1086 ?number .
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    }
    ORDER BY ?number
    LIMIT 30
    """,
}


def apply_limit(query: str, limit: Optional[int]) -> str:
    if limit is None:
        return query
    n = max(1, int(limit))
    if _LIMIT_RE.search(query):
        return _LIMIT_RE.sub(f"LIMIT {n}", query, count=1)
    return query.rstrip() + f"\nLIMIT {n}\n"


def cell_value(row: dict, key: str) -> str:
    cell = row.get(key) or {}
    if not isinstance(cell, dict):
        return ""
    return str(cell.get("value") or "").strip()


def _cell(row: dict, key: str) -> str:
    return cell_value(row, key)


def usable_label(text: str) -> bool:
    value = " ".join((text or "").split())
    if len(value) < 1:
        return False
    if _QID.match(value):
        return False
    if value.startswith("http://") or value.startswith("https://"):
        return False
    return True


def linked_variants(user: str, assistant: str) -> List[Pair]:
    """Extra trained questions that share the same capital/inventor/element slots."""
    u = " ".join((user or "").split())
    a = " ".join((assistant or "").split()).rstrip(".")
    out: List[Pair] = []
    cap_u = re.match(r"^What is the capital of (.+)\?$", u)
    cap_a = re.match(r"^The capital of (.+) is (.+)$", a)
    if cap_u and cap_a and cap_u.group(1) == cap_a.group(1):
        country, city = cap_a.group(1), cap_a.group(2)
        body = f"{city} is the capital of {country}."
        out.append((f"Where is {city}?", body))
        out.append((f"What country is {city} in?", body))
        out.append((f"What is {country}'s capital?", f"The capital of {country} is {city}."))
        return out
    inv_u = re.match(r"^Who invented (.+)\?$", u)
    inv_a = re.match(r"^(.+) is credited with inventing (.+)$", a)
    if inv_u and inv_a and inv_u.group(1) == inv_a.group(2):
        inv, person = inv_u.group(1), inv_a.group(1)
        out.append((f"Who is credited with inventing {inv}?", assistant if assistant.endswith(".") else assistant + "."))
        out.append((f"What did {person} invent?", assistant if assistant.endswith(".") else assistant + "."))
        return out
    el_u = re.match(r"^What is the atomic number of (.+)\?$", u)
    el_a = re.match(r"^The atomic number of (.+) is (\d+)$", a)
    if el_u and el_a and el_u.group(1) == el_a.group(1):
        num = el_a.group(2)
        asst = assistant if assistant.endswith(".") else assistant + "."
        out.append((f"Which element has atomic number {num}?", asst))
    return out


def expand_linked_pairs(pairs: Iterable[Pair]) -> List[Pair]:
    """Append linked variants; drop any new question that conflicts."""
    base = list(pairs)
    extra: List[Pair] = []
    for user, assistant in base:
        extra.extend(linked_variants(user, assistant))
    return drop_conflicts(dedupe_pairs(base + extra))


def verbalize(row: dict, kind: str) -> Optional[Pair]:
    """Turn one SPARQL binding into (user, assistant) or None."""
    if kind == "capitals":
        country = _cell(row, "countryLabel")
        capital = _cell(row, "capitalLabel")
        if not usable_label(country) or not usable_label(capital):
            return None
        return (
            f"What is the capital of {country}?",
            f"The capital of {country} is {capital}.",
        )
    if kind == "inventors":
        inv = _cell(row, "inventionLabel")
        person = _cell(row, "inventorLabel")
        if not usable_label(inv) or not usable_label(person):
            return None
        return (
            f"Who invented {inv}?",
            f"{person} is credited with inventing {inv}.",
        )
    if kind == "birth_years":
        person = _cell(row, "personLabel")
        year = _cell(row, "year")
        if not usable_label(person) or not year.isdigit():
            return None
        return (
            f"When was {person} born?",
            f"{person} was born in {year}.",
        )
    if kind == "elements":
        elem = _cell(row, "elementLabel")
        num = _cell(row, "number")
        if not usable_label(elem):
            return None
        try:
            number = str(int(float(num)))
        except (TypeError, ValueError):
            return None
        return (
            f"What is the atomic number of {elem}?",
            f"The atomic number of {elem} is {number}.",
        )
    return None


def dedupe_pairs(pairs: Iterable[Pair]) -> List[Pair]:
    seen = set()
    out: List[Pair] = []
    for user, assistant in pairs:
        key = (user.lower(), assistant.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append((user, assistant))
    return out


def drop_conflicts(pairs: Iterable[Pair]) -> List[Pair]:
    """Drop any User: question that has two or more distinct answers.

    Same question with different answers, then repeated N×, skews the fact
    mix (e.g. multi-capital regions). Do not pick a winner — drop entirely.
    """
    by_user: Dict[str, List[Pair]] = {}
    order: List[str] = []
    for user, assistant in pairs:
        key = " ".join((user or "").split()).lower()
        if key not in by_user:
            order.append(key)
            by_user[key] = []
        by_user[key].append((user, assistant))
    out: List[Pair] = []
    for key in order:
        group = by_user[key]
        answers = {" ".join((assistant or "").split()).lower() for _, assistant in group}
        if len(answers) != 1:
            continue
        out.append(group[0])
    return out


def run_query_urllib(query: str, *, timeout: float = 60.0) -> List[dict]:
    body = urllib.parse.urlencode({"query": query, "format": "json"}).encode("utf-8")
    req = urllib.request.Request(
        ENDPOINT,
        data=body,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return list((payload.get("results") or {}).get("bindings") or [])


def run_query(query: str) -> List[dict]:
    """Hit WDQS via SPARQLWrapper when installed, else stdlib urllib."""
    try:
        from SPARQLWrapper import JSON, SPARQLWrapper
    except ImportError:
        return run_query_urllib(query)
    sparql = SPARQLWrapper(ENDPOINT)
    sparql.addCustomHttpHeader("User-Agent", USER_AGENT)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        sparql.setTimeout(60)
    except Exception:
        pass
    results = sparql.query().convert()
    return list((results.get("results") or {}).get("bindings") or [])


def collect_facts(
    kinds: Sequence[str],
    *,
    query_fn: QueryFn = run_query,
    sleep_s: float = 1.0,
    limit: Optional[int] = None,
) -> List[Pair]:
    pairs: List[Pair] = []
    for i, kind in enumerate(kinds):
        if kind not in QUERIES:
            raise ValueError(f"Unknown query {kind!r}. Choose from: {sorted(QUERIES)}")
        rows = query_fn(apply_limit(QUERIES[kind], limit))
        for row in rows:
            pair = verbalize(row, kind)
            if pair:
                pairs.append(pair)
        if i + 1 < len(kinds) and sleep_s > 0:
            time.sleep(sleep_s)
    return expand_linked_pairs(drop_conflicts(dedupe_pairs(pairs)))


def write_facts(pairs: Sequence[Pair], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for user, assistant in pairs:
            handle.write(wrap_native(user, assistant) + "\n")
    return len(pairs)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wikidata SPARQL → native User:/Assistant: facts",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=["capitals", "elements"],
        choices=sorted(QUERIES),
        help="Which query groups to run",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "data" / "facts" / "wikidata_facts.txt"),
        help="Native chat lines (picked up by make_fact_mix.py)",
    )
    parser.add_argument("--sleep", type=float, default=1.0, help="Delay between query groups")
    parser.add_argument("--limit", type=int, default=None, help="Override SPARQL LIMIT for every group")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None, *, query_fn: QueryFn = run_query) -> int:
    args = parse_args(argv)
    try:
        pairs = collect_facts(
            list(args.queries),
            query_fn=query_fn,
            sleep_s=float(args.sleep),
            limit=args.limit,
        )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"Wikidata query failed: {exc}", file=sys.stderr)
        return 1
    if not pairs:
        print("No usable facts (empty or all Q-id labels). Nothing written.", file=sys.stderr)
        return 1
    n = write_facts(pairs, Path(args.output))
    print(f"Wrote {n:,} facts → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
