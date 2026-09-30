"""Tests for Wikidata → native chat facts (no live SPARQL)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.make_fact_mix import load_user_facts
from tools.wikidata_to_facts import (
    apply_limit,
    collect_facts,
    drop_conflicts,
    usable_label,
    verbalize,
    write_facts,
    main as wiki_main,
)


def _bind(**kwargs):
    return {key: {"value": value} for key, value in kwargs.items()}


class WikidataFactsTests(unittest.TestCase):
    def test_verbalize_capitals_and_elements(self):
        cap = verbalize(_bind(countryLabel="France", capitalLabel="Paris"), "capitals")
        self.assertEqual(
            cap,
            ("What is the capital of France?", "The capital of France is Paris."),
        )
        elem = verbalize(_bind(elementLabel="Oxygen", number="8.0"), "elements")
        self.assertEqual(
            elem,
            ("What is the atomic number of Oxygen?", "The atomic number of Oxygen is 8."),
        )

    def test_drops_qid_and_url_labels(self):
        self.assertFalse(usable_label("Q42"))
        self.assertTrue(usable_label("C"))
        self.assertTrue(usable_label("e"))
        self.assertIsNone(
            verbalize(_bind(countryLabel="Q123", capitalLabel="Paris"), "capitals"),
        )
        self.assertIsNone(
            verbalize(_bind(elementLabel="https://example.com", number="1"), "elements"),
        )

    def test_birth_year_and_inventor(self):
        birth = verbalize(_bind(personLabel="Ada Lovelace", year="1815"), "birth_years")
        self.assertEqual(
            birth,
            ("When was Ada Lovelace born?", "Ada Lovelace was born in 1815."),
        )
        inv = verbalize(
            _bind(inventionLabel="the telephone", inventorLabel="Alexander Graham Bell"),
            "inventors",
        )
        self.assertIn("Alexander Graham Bell", inv[1])

    def test_apply_limit_and_collect_dedupes(self):
        q = apply_limit("SELECT ?x WHERE { ?x ?y ?z } LIMIT 80", 12)
        self.assertIn("LIMIT 12", q)

        rows = [
            _bind(countryLabel="France", capitalLabel="Paris"),
            _bind(countryLabel="France", capitalLabel="Paris"),
            _bind(countryLabel="Q9", capitalLabel="Nope"),
        ]

        def fake_query(_query: str):
            return list(rows)

        pairs = collect_facts(["capitals"], query_fn=fake_query, sleep_s=0)
        users = [p[0] for p in pairs]
        self.assertIn("What is the capital of France?", users)
        self.assertIn("Where is Paris?", users)
        self.assertIn("What country is Paris in?", users)
        self.assertIn("What is France's capital?", users)

    def test_collect_drops_conflicting_answers(self):
        rows = [
            _bind(countryLabel="Kashmir", capitalLabel="Srinagar"),
            _bind(countryLabel="Kashmir", capitalLabel="Jammu"),
            _bind(countryLabel="France", capitalLabel="Paris"),
        ]

        def fake_query(_query: str):
            return list(rows)

        pairs = collect_facts(["capitals"], query_fn=fake_query, sleep_s=0)
        users = [p[0] for p in pairs]
        self.assertIn("What is the capital of France?", users)
        self.assertNotIn("What is the capital of Kashmir?", users)
        self.assertEqual(
            drop_conflicts(
                [
                    ("What is the capital of Kashmir?", "Srinagar"),
                    ("What is the capital of Kashmir?", "Jammu"),
                ]
            ),
            [],
        )

    def test_write_is_native_and_fact_mix_loads_it(self):
        pairs = [
            ("What is the capital of France?", "The capital of France is Paris."),
            ("What is the atomic number of Oxygen?", "The atomic number of Oxygen is 8."),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "facts" / "wikidata_facts.txt"
            n = write_facts(pairs, dest)
            self.assertEqual(n, 2)
            text = dest.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("User: What is the capital of France?"))
            self.assertIn(" Assistant: The capital of France is Paris.", text)
            loaded = load_user_facts([dest])
            self.assertEqual(loaded, pairs)

    def test_cli_writes_via_injected_query_fn(self):
        def fake_query(_query: str):
            return [_bind(elementLabel="Hydrogen", number="1")]

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "wikidata_facts.txt"
            code = wiki_main(
                ["--queries", "elements", "--output", str(dest), "--sleep", "0"],
                query_fn=fake_query,
            )
            self.assertEqual(code, 0)
            self.assertIn("atomic number of Hydrogen", dest.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
