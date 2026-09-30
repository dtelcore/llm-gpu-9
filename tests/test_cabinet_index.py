"""Normalized exact lookup for the chat fact cabinet."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.cabinet_index import (
    CabinetIndex,
    article_slot_candidates,
    extract_entities,
    load_cabinet,
    merge_cabinet,
    normalize_question,
    remember,
)
from training.chat_format import USER_PREFIX


def _write_jsonl(path: Path, pairs) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for user, assistant in pairs:
            handle.write(
                json.dumps({"query": {"user": user}, "response": {"assistant": assistant}})
                + "\n"
            )


class NormalizeQuestionTests(unittest.TestCase):
    def test_strip_and_collapse_whitespace(self):
        self.assertEqual(
            normalize_question("  What   is the   capital of France?  "),
            normalize_question("What is the capital of France?"),
        )

    def test_casefold(self):
        self.assertEqual(
            normalize_question("WHAT IS THE CAPITAL OF FRANCE?"),
            normalize_question("what is the capital of france?"),
        )

    def test_optional_trailing_question_mark(self):
        self.assertEqual(
            normalize_question("What is the capital of France"),
            normalize_question("What is the capital of France?"),
        )

    def test_strip_leading_user_prefix(self):
        self.assertEqual(
            normalize_question("User: What is the capital of France?"),
            normalize_question("What is the capital of France?"),
        )
        self.assertEqual(
            normalize_question("user: What is the capital of France?"),
            normalize_question("What is the capital of France?"),
        )

    def test_strip_pasted_assistant_tail(self):
        self.assertEqual(
            normalize_question(
                "User: What is the capital of France? Assistant: The capital of France is Paris."
            ),
            normalize_question("What is the capital of France?"),
        )


class CabinetIndexTests(unittest.TestCase):
    def setUp(self):
        self.index = CabinetIndex()
        self.index.add("What is the capital of France?", "The capital of France is Paris.")
        self.index.add("Tell me about Neonics.", "Neonics were banned in the EU.")
        self.index.add("What is 0 factorial?", "0 factorial equals 1.")

    def test_france_hit_without_question_mark(self):
        hit = self.index.lookup("what is the capital of france")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.user, "What is the capital of France?")
        self.assertEqual(hit.generate_prompt, "User: What is the capital of France? Assistant:")

    def test_typed_user_prefix_still_hits(self):
        hit = self.index.lookup("User: What is the capital of France?")
        self.assertIsNotNone(hit)
        self.assertTrue(hit.generate_prompt.startswith(USER_PREFIX))
        self.assertIn("Assistant:", hit.generate_prompt)

    def test_unobtanium_miss_is_not_fuzzy(self):
        self.assertIsNone(self.index.lookup("What is unobtanium"))
        self.assertIsNone(self.index.lookup("What is sequential layer streaming?"))
        self.assertEqual(article_slot_candidates("What is unobtanium"), [])
        self.assertIsNone(self.index.resolve_trained("What is unobtanium"))

    def test_article_rewrite_unique_inventor(self):
        self.index.add("Who invented Plough?", "Ernesto Schiaparelli is credited with inventing Plough.")
        hit = self.index.resolve_trained("Who invented the plough?")
        self.assertIsNotNone(hit)
        fact, match = hit
        self.assertEqual(match, "trained_article")
        self.assertEqual(fact.user, "Who invented Plough?")

    def test_article_rewrite_ambiguous_is_miss(self):
        idx = CabinetIndex()
        idx.add("Who invented Plough?", "A is credited with inventing Plough.")
        idx.add("Who invented the plough?", "B is credited with inventing the plough.")
        self.assertIsNone(idx.resolve_trained("Who invented a plough?"))

    def test_file_alias_beats_learned_exact(self):
        idx = CabinetIndex()
        idx.add(
            "What is sequential layer streaming?",
            "layer_strategy=stream loads one transformer block.",
        )
        idx.add("what is layer streaming ?", "Howdy is an American streaming service.", source="learned")
        idx.load_file_aliases()
        hit = idx.resolve_trained("what is layer streaming ?")
        self.assertIsNotNone(hit)
        self.assertEqual(hit[1], "trained_alias")
        self.assertEqual(hit[0].user, "What is sequential layer streaming?")
        self.assertEqual(idx.resolve_learned("what is layer streaming ?")[0].source, "learned")

    def test_quarantine_skips_trained_add(self):
        idx = CabinetIndex()
        idx.set_quarantine(["Who invented Plough?"])
        self.assertIsNone(idx.add("Who invented Plough?", "Ernesto Schiaparelli is credited with inventing Plough."))
        self.assertIsNone(idx.resolve_trained("Who invented the plough?"))

    def test_neonics_hit(self):
        hit = self.index.lookup("Tell me about Neonics.")
        self.assertIsNotNone(hit)
        self.assertIn("Neonics", hit.assistant)

    def test_capital_entities_and_related_prompts(self):
        ents = extract_entities(
            "What is the capital of France?",
            "The capital of France is Paris.",
        )
        self.assertEqual(ents, frozenset({"france", "paris"}))
        related = self.index.related_prompts(ents)
        self.assertIn("What is the capital of France?", related)
        self.assertTrue(self.index.entities_mentioned("where is paris?"))
        self.assertFalse(self.index.entities_mentioned("What is unobtanium"))

    def test_conflicting_answers_not_indexed(self):
        idx = CabinetIndex()
        first = idx.add("What is the capital of Kashmir?", "Srinagar.")
        second = idx.add("What is the capital of Kashmir?", "Jammu.")
        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(idx.lookup("What is the capital of Kashmir?").assistant, "Srinagar.")

    def test_load_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "facts.jsonl"
            _write_jsonl(
                path,
                [
                    ("What is the capital of France?", "The capital of France is Paris."),
                    ("What is the capital of France?", "The capital of France is Paris."),
                ],
            )
            loaded = load_cabinet(path)
            self.assertEqual(len(loaded), 1)
            self.assertIsNotNone(loaded.lookup("What is the capital of France?"))
            self.assertEqual(loaded.lookup("What is the capital of France?").source, "trained")

    def test_load_jsonl_glued_records_and_junk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "facts.jsonl"
            a = json.dumps({"query": {"user": "Q1?"}, "response": {"assistant": "A1."}})
            b = json.dumps({"query": {"user": "Q2?"}, "response": {"assistant": "A2."}})
            path.write_text(a + b + "\nnot json\n", encoding="utf-8")
            loaded = load_cabinet(path)
            self.assertEqual(len(loaded), 2)
            self.assertIsNotNone(loaded.lookup("Q1?"))
            self.assertIsNotNone(loaded.lookup("Q2?"))

    def test_remember_pads_missing_newline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "learned.jsonl"
            idx = CabinetIndex()
            remember(idx, path, "first question", "First answer.")
            text = path.read_text(encoding="utf-8").rstrip("\n")
            path.write_text(text, encoding="utf-8")
            remember(idx, path, "second question", "Second answer.")
            lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[1])["query"]["user"], "second question")

    def test_remember_appends_jsonl_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "learned.jsonl"
            idx = CabinetIndex()
            first = remember(idx, path, "tell me about ford", "Ford Motor Company.")
            second = remember(idx, path, "Tell me about Ford", "Ford Motor Company.")
            self.assertIsNotNone(first)
            self.assertEqual(first.source, "learned")
            self.assertEqual(second, first)
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            rec = json.loads(lines[0])
            self.assertEqual(rec["query"]["user"], "tell me about ford")
            self.assertEqual(rec["response"]["assistant"], "Ford Motor Company.")

    def test_remember_does_not_overwrite_trained_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "learned.jsonl"
            idx = CabinetIndex()
            idx.add("What is the capital of France?", "The capital of France is Paris.")
            out = remember(idx, path, "what is the capital of france", "Paris is a city in Texas.")
            self.assertEqual(out.assistant, "The capital of France is Paris.")
            self.assertFalse(path.exists())

    def test_merge_learned_skips_trained_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            trained = Path(tmp) / "trained.jsonl"
            learned = Path(tmp) / "learned.jsonl"
            _write_jsonl(trained, [("What is the capital of France?", "The capital of France is Paris.")])
            _write_jsonl(
                learned,
                [
                    ("What is the capital of France?", "Paris, Texas."),
                    ("tell me about ford", "Ford Motor Company."),
                ],
            )
            idx = load_cabinet(trained)
            added = merge_cabinet(idx, learned, source="learned")
            self.assertEqual(added, 1)
            self.assertEqual(idx.lookup("What is the capital of France?").assistant, "The capital of France is Paris.")
            self.assertEqual(idx.lookup("tell me about ford").source, "learned")


class LiveMixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = _ROOT / "data" / "chat_facts.jsonl"
        cls.mix = load_cabinet(path) if path.is_file() else None

    def test_live_mix_neonics_unobtanium(self):
        if self.mix is None or len(self.mix) == 0:
            self.skipTest("data/chat_facts.jsonl missing")
        self.assertIsNotNone(self.mix.lookup("Tell me about Neonics."))
        self.assertIsNone(self.mix.lookup("What is unobtanium"))

    def test_live_v7_linked_paris(self):
        path = _ROOT / "data" / "chat_facts_v7.jsonl"
        if not path.is_file():
            self.skipTest("data/chat_facts_v7.jsonl missing")
        mix = load_cabinet(path)
        self.assertIsNotNone(mix.lookup("What is the capital of France?"))
        self.assertIsNotNone(mix.lookup("Where is Paris?"))
        self.assertIsNotNone(mix.lookup("What country is Paris in?"))
        self.assertIsNone(mix.lookup("What is unobtanium"))
        self.assertIsNone(mix.resolve_trained("Who invented Plough?"))
        self.assertIsNone(mix.resolve_trained("Who invented the plough?"))


if __name__ == "__main__":
    unittest.main()
