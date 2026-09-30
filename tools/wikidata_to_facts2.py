#!/usr/bin/env python3
"""
tools/wikidata_to_facts2.py  (wikidatafetch2)

Second Wikidata SPARQL pack for the chat-facts cabinet: technology, health,
and maths. Writes separate files under data/facts/:

  data/facts/tech_facts.txt
  data/facts/health_facts.txt
  data/facts/maths_facts.txt

Those names avoid colliding with wikidata_facts.txt, capitals_priority.txt,
and elements.txt from the first tool / local overrides.

Does not concatenate data/train.txt or data/chat_train.*. Keep LIMIT modest:
this cabinet memorizes tens-to-hundreds of repeated facts, not thousands.

Conflict rule: one answer per question. If the same User: line would get two
distinct Assistant: lines, drop that question entirely (do not pick a winner).

Health lines are encyclopedic facts only (vitamins, organs, pathogens, glands).
Never treatment, dosing, or "should you" advice.

Network: SPARQLWrapper if installed, else stdlib urllib. Be polite (--sleep).
Reuses helpers from tools.wikidata_to_facts.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
import unicodedata
import urllib.error
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.wikidata_to_facts import (
    apply_limit,
    cell_value,
    drop_conflicts,
    dedupe_pairs,
    run_query,
    usable_label,
    write_facts,
    QueryFn,
    Pair,
)

FACTS_DIR = ROOT / "data" / "facts"

DEFAULT_OUTPUTS: Dict[str, Path] = {
    "technology": FACTS_DIR / "tech_facts.txt",
    "health": FACTS_DIR / "health_facts.txt",
    "maths": FACTS_DIR / "maths_facts.txt",
}

# Distinct templates per kind so the pack is not one "What is the capital of X?" slot.
# Notable-item filter: wikibase:sitelinks + ORDER BY DESC, then a modest LIMIT.
QUERIES: Dict[str, str] = {
    "lang_designers": """
    # query: lang_designers
    SELECT DISTINCT ?langLabel ?designerLabel WHERE {
      ?lang wdt:P31 wd:Q9143 .
      ?lang wdt:P287 ?designer .
      ?lang wikibase:sitelinks ?links .
      FILTER(?links >= 40)
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    }
    ORDER BY DESC(?links)
    LIMIT 40
    """,
    "lang_years": """
    # query: lang_years
    SELECT DISTINCT ?langLabel (YEAR(?date) AS ?year) WHERE {
      ?lang wdt:P31 wd:Q9143 .
      ?lang wdt:P571 ?date .
      ?lang wikibase:sitelinks ?links .
      FILTER(?links >= 40)
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    }
    ORDER BY DESC(?links)
    LIMIT 40
    """,
    "protocols": """
    # query: protocols
    SELECT DISTINCT ?protocolLabel (YEAR(?date) AS ?year) WHERE {
      ?protocol wdt:P31 wd:Q15836568 .
      ?protocol wdt:P571 ?date .
      ?protocol wikibase:sitelinks ?links .
      FILTER(?links >= 20)
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    }
    ORDER BY DESC(?links)
    LIMIT 30
    """,
    "si_units": """
    # query: si_units
    SELECT DISTINCT ?quantityLabel ?unitLabel WHERE {
      VALUES ?kind { wd:Q223662 wd:Q208469 }
      ?quantity wdt:P8111 ?unit .
      ?unit wdt:P31 ?kind .
      ?quantity wikibase:sitelinks ?links .
      FILTER(?links >= 20)
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    }
    ORDER BY DESC(?links)
    LIMIT 30
    """,
    "vitamin_formulas": """
    # query: vitamin_formulas
    SELECT DISTINCT ?vitaminLabel ?formula WHERE {
      ?vitamin wdt:P279* wd:Q34956 .
      ?vitamin wdt:P274 ?formula .
      ?vitamin wikibase:sitelinks ?links .
      FILTER(?links >= 10)
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    }
    ORDER BY DESC(?links)
    LIMIT 25
    """,
    "organ_systems": """
    # query: organ_systems
    SELECT DISTINCT ?organLabel ?systemLabel WHERE {
      ?organ wdt:P279* wd:Q712378 .
      ?organ wdt:P361 ?system .
      ?system wdt:P279* wd:Q188193 .
      ?organ wikibase:sitelinks ?links .
      FILTER(?links >= 30)
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    }
    ORDER BY DESC(?links)
    LIMIT 40
    """,
    "disease_pathogens": """
    # query: disease_pathogens
    SELECT DISTINCT ?diseaseLabel ?pathogenLabel WHERE {
      ?disease wdt:P828 ?pathogen .
      ?pathogen wdt:P31 wd:Q16521 .
      ?disease wikibase:sitelinks ?links .
      FILTER(?links >= 30)
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    }
    ORDER BY DESC(?links)
    LIMIT 40
    """,
    "amino_acids": """
    # query: amino_acids
    SELECT DISTINCT ?aaLabel ?formula WHERE {
      ?aa wdt:P279* wd:Q8066 .
      ?aa wdt:P274 ?formula .
      ?aa wikibase:sitelinks ?links .
      FILTER(?links >= 20)
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    }
    ORDER BY DESC(?links)
    LIMIT 20
    """,
    "math_constants": """
    # query: math_constants
    SELECT DISTINCT ?constLabel ?value WHERE {
      ?const wdt:P31 wd:Q186509 .
      ?const wdt:P1181 ?value .
      ?const wikibase:sitelinks ?links .
      FILTER(?links >= 20)
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    }
    ORDER BY DESC(?links)
    LIMIT 20
    """,
    "theorem_names": """
    # query: theorem_names
    SELECT DISTINCT ?theoremLabel ?personLabel WHERE {
      ?theorem wdt:P31 wd:Q65943 .
      ?theorem wdt:P138 ?person .
      ?person wdt:P31 wd:Q5 .
      ?theorem wikibase:sitelinks ?links .
      FILTER(?links >= 25)
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    }
    ORDER BY DESC(?links)
    LIMIT 30
    """,
    "prime_numbers": """
    # query: prime_numbers
    SELECT DISTINCT ?value WHERE {
      ?n wdt:P31 wd:Q49008 .
      ?n wdt:P1181 ?value .
      FILTER(xsd:integer(?value) >= 2 && xsd:integer(?value) <= 97)
    }
    LIMIT 25
    """,
    "composite_numbers": """
    # query: composite_numbers
    SELECT DISTINCT ?value WHERE {
      ?n wdt:P31 wd:Q50707 .
      ?n wdt:P1181 ?value .
      FILTER(xsd:integer(?value) >= 4 && xsd:integer(?value) <= 50)
    }
    LIMIT 15
    """,
}

DOMAIN_KINDS: Dict[str, Tuple[str, ...]] = {
    "technology": ("lang_designers", "lang_years", "protocols", "si_units"),
    "health": ("vitamin_formulas", "organ_systems", "disease_pathogens", "amino_acids"),
    "maths": ("math_constants", "theorem_names", "prime_numbers", "composite_numbers"),
}

KIND_TO_DOMAIN: Dict[str, str] = {
    kind: domain
    for domain, kinds in DOMAIN_KINDS.items()
    for kind in kinds
}

# Wikidata has no clean "statement" property. Small high-school / early-undergrad
# seed so the maths pack is not only named-after and numeric-value slots.
THEOREM_STATEMENTS: List[Pair] = [
    (
        "State the Pythagorean theorem.",
        "In a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides.",
    ),
    (
        "State the quadratic formula.",
        "The solutions of ax^2 + bx + c = 0 are x = (-b ± sqrt(b^2 - 4ac)) / (2a) when a is not zero.",
    ),
    (
        "State the fundamental theorem of calculus.",
        "If F is an antiderivative of f, then the definite integral of f from a to b equals F(b) - F(a).",
    ),
    (
        "What is the sum of the interior angles of a triangle?",
        "The sum of the interior angles of a triangle is 180 degrees.",
    ),
    (
        "State the binomial square formula.",
        "(a + b)^2 equals a^2 + 2ab + b^2.",
    ),
    (
        "What is the derivative of sin(x) with respect to x?",
        "The derivative of sin(x) is cos(x).",
    ),
    (
        "What is the derivative of e^x with respect to x?",
        "The derivative of e^x is e^x.",
    ),
    (
        "State the law of sines.",
        "In any triangle, a/sin A = b/sin B = c/sin C.",
    ),
    (
        "State the law of cosines.",
        "In any triangle, c^2 = a^2 + b^2 - 2ab cos C.",
    ),
    (
        "What is the formula for the area of a circle?",
        "The area of a circle is pi r^2, where r is the radius.",
    ),
]

NUMBER_PROPERTY_SEEDS: List[Pair] = [
    ("Is 1 a prime number?", "No, 1 is not a prime number."),
    ("What is 0 factorial?", "0 factorial equals 1."),
    ("Is 2 the only even prime number?", "Yes, 2 is the only even prime number."),
]

# Encyclopedic anatomy counts Wikidata does not model as clean Q&A. Not advice.
HEALTH_SEEDS: List[Pair] = [
    (
        "How many chambers does the human heart have?",
        "The human heart has four chambers.",
    ),
    (
        "How many bones are in an adult human skeleton?",
        "An adult human skeleton has 206 bones.",
    ),
    (
        "How many chromosomes does a typical human cell have?",
        "A typical human cell has 46 chromosomes.",
    ),
    (
        "What gas do humans inhale that cells use for respiration?",
        "Humans inhale oxygen, which cells use for respiration.",
    ),
    (
        "What gas do humans exhale as a product of respiration?",
        "Humans exhale carbon dioxide as a product of respiration.",
    ),
]

_ADVICE_RE = re.compile(
    r"\b("
    r"take\s+\d|"
    r"dosage|dosing|"
    r"mg/?d(?:ay)?|"
    r"should\s+you|"
    r"how\s+much\s+to\s+take|"
    r"treatment\s+for|"
    r"how\s+to\s+treat|"
    r"cure\s+for|"
    r"prescri(?:be|ption)"
    r")\b",
    re.IGNORECASE,
)

_SCIENCE_CHARS = str.maketrans({
    "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5",
    "₆": "6", "₇": "7", "₈": "8", "₉": "9", "₀": "0",
    "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5",
    "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁰": "0",
    "β": "beta", "α": "alpha", "γ": "gamma",
    "π": "pi",
    "–": "-", "—": "-",
})


def fold_label(text: str) -> str:
    """ASCII-fold labels/formulas for the tiny chat-facts BPE."""
    raw = " ".join((text or "").split())
    mapped = raw.translate(_SCIENCE_CHARS)
    folded = unicodedata.normalize("NFKD", mapped).encode("ascii", "ignore").decode("ascii")
    folded = " ".join(folded.split())
    return folded or mapped or raw


def looks_like_health_advice(user: str, assistant: str) -> bool:
    """True for dosing / treatment wording. Encyclopedic anatomy stays False."""
    blob = f"{user} {assistant}"
    return bool(_ADVICE_RE.search(blob))


def format_year(raw: str) -> Optional[str]:
    text = (raw or "").strip()
    if text.isdigit() and 1 <= len(text) <= 4:
        year = int(text)
        if 1 <= year <= 2100:
            return str(year)
        return None
    try:
        year = int(float(text))
    except (TypeError, ValueError):
        return None
    if 1 <= year <= 2100:
        return str(year)
    return None


def format_numeric(raw: str) -> Optional[str]:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if abs(number - round(number)) < 1e-9 and abs(number) < 1e12:
        return str(int(round(number)))
    return f"{number:.6g}"


def _label(row: dict, key: str) -> str:
    return fold_label(cell_value(row, key))


def verbalize(row: dict, kind: str) -> Optional[Pair]:
    """Turn one SPARQL binding into (user, assistant) or None."""
    if kind == "lang_designers":
        lang = _label(row, "langLabel")
        designer = _label(row, "designerLabel")
        if not usable_label(lang) or not usable_label(designer):
            return None
        if "style sheet" in lang.lower():
            return None
        return (
            f"Who designed the {lang} programming language?",
            f"{designer} designed the {lang} programming language.",
        )
    if kind == "lang_years":
        lang = _label(row, "langLabel")
        year = format_year(cell_value(row, "year"))
        if not usable_label(lang) or year is None:
            return None
        if "style sheet" in lang.lower():
            return None
        return (
            f"In which year was the {lang} programming language first released?",
            f"The {lang} programming language was first released in {year}.",
        )
    if kind == "protocols":
        protocol = _label(row, "protocolLabel")
        year = format_year(cell_value(row, "year"))
        if not usable_label(protocol) or year is None:
            return None
        return (
            f"In which year was {protocol} introduced?",
            f"{protocol} was introduced in {year}.",
        )
    if kind == "si_units":
        quantity = _label(row, "quantityLabel")
        unit = _label(row, "unitLabel")
        if not usable_label(quantity) or not usable_label(unit):
            return None
        return (
            f"What is the SI unit of {quantity}?",
            f"The SI unit of {quantity} is the {unit}.",
        )
    if kind == "vitamin_formulas":
        vitamin = _label(row, "vitaminLabel")
        formula = _label(row, "formula")
        if not usable_label(vitamin) or not usable_label(formula):
            return None
        if looks_like_health_advice(vitamin, formula):
            return None
        return (
            f"What is the chemical formula of {vitamin}?",
            f"The chemical formula of {vitamin} is {formula}.",
        )
    if kind == "organ_systems":
        organ = _label(row, "organLabel")
        system = _label(row, "systemLabel")
        if not usable_label(organ) or not usable_label(system):
            return None
        if organ.lower() == system.lower():
            return None
        return (
            f"Which organ system does the {organ} belong to?",
            f"The {organ} belongs to the {system}.",
        )
    if kind == "disease_pathogens":
        disease = _label(row, "diseaseLabel")
        pathogen = _label(row, "pathogenLabel")
        if not usable_label(disease) or not usable_label(pathogen):
            return None
        if "pandemic" in disease.lower():
            return None
        user = f"What pathogen causes {disease}?"
        assistant = f"{pathogen} is the pathogen that causes {disease}."
        if looks_like_health_advice(user, assistant):
            return None
        return (user, assistant)
    if kind == "amino_acids":
        aa = _label(row, "aaLabel")
        formula = _label(row, "formula")
        if not usable_label(aa) or not usable_label(formula):
            return None
        return (
            f"What is the chemical formula of the amino acid {aa}?",
            f"The chemical formula of the amino acid {aa} is {formula}.",
        )
    if kind == "math_constants":
        const = _label(row, "constLabel")
        value = format_numeric(cell_value(row, "value"))
        if not usable_label(const) or value is None:
            return None
        if value in {"0", "1", "-1"}:
            return None
        return (
            f"What is the approximate numerical value of {const}?",
            f"The approximate value of {const} is {value}.",
        )
    if kind == "theorem_names":
        theorem = _label(row, "theoremLabel")
        person = _label(row, "personLabel")
        if not usable_label(theorem) or not usable_label(person):
            return None
        return (
            f"Who is {theorem} named after?",
            f"{theorem} is named after {person}.",
        )
    if kind == "prime_numbers":
        value = format_numeric(cell_value(row, "value"))
        if value is None or not value.isdigit():
            return None
        n = int(value)
        if n < 2:
            return None
        return (
            f"Is {n} a prime number?",
            f"Yes, {n} is a prime number.",
        )
    if kind == "composite_numbers":
        value = format_numeric(cell_value(row, "value"))
        if value is None or not value.isdigit():
            return None
        n = int(value)
        if n < 4:
            return None
        return (
            f"Is {n} a prime number?",
            f"No, {n} is a composite number.",
        )
    return None


def seed_pairs(domain: str) -> List[Pair]:
    if domain == "maths":
        return list(THEOREM_STATEMENTS) + list(NUMBER_PROPERTY_SEEDS)
    if domain == "health":
        return list(HEALTH_SEEDS)
    return []


def collect_kinds(
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
    return list(pairs)


def finalize_pairs(pairs: Iterable[Pair], *, domain: Optional[str] = None) -> List[Pair]:
    out = drop_conflicts(dedupe_pairs(pairs))
    if domain == "health":
        out = [p for p in out if not looks_like_health_advice(p[0], p[1])]
    return out


def collect_domain(
    domain: str,
    *,
    query_fn: QueryFn = run_query,
    sleep_s: float = 1.0,
    limit: Optional[int] = None,
) -> List[Pair]:
    if domain not in DOMAIN_KINDS:
        raise ValueError(f"Unknown domain {domain!r}. Choose from: {sorted(DOMAIN_KINDS)}")
    kinds = DOMAIN_KINDS[domain]
    pairs = collect_kinds(kinds, query_fn=query_fn, sleep_s=sleep_s, limit=limit)
    pairs.extend(seed_pairs(domain))
    return finalize_pairs(pairs, domain=domain)


def resolve_output_path(domain: str, args: argparse.Namespace) -> Path:
    explicit = {
        "technology": args.tech_output,
        "health": args.health_output,
        "maths": args.maths_output,
    }[domain]
    if explicit:
        return Path(explicit)
    return Path(args.output_dir) / DEFAULT_OUTPUTS[domain].name


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wikidata SPARQL → native User:/Assistant: facts "
            "(technology, health, maths packs)"
        ),
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(DOMAIN_KINDS),
        choices=sorted(DOMAIN_KINDS),
        help="Which fact packs to write (default: all three)",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=None,
        choices=sorted(QUERIES),
        help="Optional SPARQL kinds only (still grouped into domain output files)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(FACTS_DIR),
        help="Directory for tech_facts.txt / health_facts.txt / maths_facts.txt",
    )
    parser.add_argument(
        "--tech-output",
        type=str,
        default=None,
        help="Override path for the technology pack (default: <output-dir>/tech_facts.txt)",
    )
    parser.add_argument(
        "--health-output",
        type=str,
        default=None,
        help="Override path for the health pack (default: <output-dir>/health_facts.txt)",
    )
    parser.add_argument(
        "--maths-output",
        type=str,
        default=None,
        help="Override path for the maths pack (default: <output-dir>/maths_facts.txt)",
    )
    parser.add_argument("--sleep", type=float, default=1.0, help="Delay between SPARQL groups")
    parser.add_argument("--limit", type=int, default=None, help="Override SPARQL LIMIT for every group")
    return parser.parse_args(argv)


def kinds_for_run(args: argparse.Namespace) -> Dict[str, List[str]]:
    domains = list(args.domains)
    if args.queries:
        grouped: Dict[str, List[str]] = {d: [] for d in DOMAIN_KINDS}
        for kind in args.queries:
            grouped[KIND_TO_DOMAIN[kind]].append(kind)
        return {d: kinds for d, kinds in grouped.items() if kinds and d in domains}
    return {d: list(DOMAIN_KINDS[d]) for d in domains}


def main(argv: Optional[Sequence[str]] = None, *, query_fn: QueryFn = run_query) -> int:
    args = parse_args(argv)
    grouped = kinds_for_run(args)
    if not grouped:
        print("No query groups selected. Nothing written.", file=sys.stderr)
        return 1

    written = 0
    files = 0
    domain_list = list(grouped)
    try:
        for i, domain in enumerate(domain_list):
            pairs = collect_kinds(
                grouped[domain],
                query_fn=query_fn,
                sleep_s=float(args.sleep),
                limit=args.limit,
            )
            pairs.extend(seed_pairs(domain))
            pairs = finalize_pairs(pairs, domain=domain)
            if i + 1 < len(domain_list) and float(args.sleep) > 0:
                time.sleep(float(args.sleep))
            if not pairs:
                print(f"No usable {domain} facts. Skipping that pack.", file=sys.stderr)
                continue
            dest = resolve_output_path(domain, args)
            n = write_facts(pairs, dest)
            print(f"Wrote {n:,} {domain} facts → {dest}")
            written += n
            files += 1
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"Wikidata query failed: {exc}", file=sys.stderr)
        return 1
    if not written:
        print("No usable facts (empty, conflicts, or all Q-id labels). Nothing written.", file=sys.stderr)
        return 1
    print(f"Done: {written:,} facts across {files} pack(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
