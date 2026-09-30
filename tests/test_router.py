"""Router: cabinet wins over calc; search miss falls through."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.cabinet_index import CabinetIndex, merge_cabinet
from training.router import (
    MISS_HINT,
    RELATED_MISS_HINT,
    alias_trained_topics,
    looks_like_fact_question,
    remember_search_hit,
    route,
)


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.index = CabinetIndex()
        self.index.add("What is the capital of France?", "The capital of France is Paris.")
        self.index.add("What is 0 factorial?", "0 factorial equals 1.")
        self.index.add("Tell me about Neonics.", "Neonics were banned in the EU.")

    def test_cabinet_france_uses_stored_prompt(self):
        d = route("what is the capital of france", self.index, search_enabled=False)
        self.assertEqual(d.kind, "cabinet")
        self.assertEqual(d.detail, "generate")
        self.assertEqual(d.text, "User: What is the capital of France? Assistant:")
        self.assertEqual(d.fact.assistant, "The capital of France is Paris.")

    def test_cabinet_wins_over_arithmetic_looking_question(self):
        d = route("What is 0 factorial?", self.index, search_fn=lambda q: "SHOULD NOT SEARCH")
        self.assertEqual(d.kind, "cabinet")
        self.assertIn("factorial", d.fact.assistant)

    def test_calc_two_plus_two(self):
        d = route("2+2", self.index, search_enabled=False)
        self.assertEqual(d.kind, "calc")
        self.assertEqual(d.text, "4")

    def test_what_is_one_plus_one_is_calc_not_wikipedia(self):
        def boom(_q):
            self.fail("search must not run for what is 1 + 1")

        d = route("what is 1 + 1", self.index, search_enabled=True, search_fn=boom)
        self.assertEqual(d.kind, "calc")
        self.assertEqual(d.text, "2")

    def test_search_uses_stripped_topic_not_full_prompt(self):
        seen = []

        def capture(q):
            seen.append(q)
            return "Ford Motor Company is an American automaker."

        d = route("tell me about ford", self.index, search_fn=capture)
        self.assertEqual(seen, ["ford"])
        self.assertEqual(d.kind, "search")

        seen.clear()
        d = route("what is a ford ?", self.index, search_fn=capture)
        self.assertEqual(seen, ["ford"])
        self.assertEqual(d.kind, "search")

    def test_search_topic_strips_wrappers(self):
        from training.router import search_topic
        self.assertEqual(search_topic("tell me about ford"), "ford")
        self.assertEqual(search_topic("tell me about Neonics."), "Neonics")
        self.assertEqual(search_topic("what is a ford ?"), "ford")
        self.assertEqual(search_topic("What is unobtanium"), "unobtanium")

    def test_unobtanium_search_success(self):
        d = route(
            "What is unobtanium",
            self.index,
            search_enabled=True,
            search_fn=lambda q: "A fictional metal.",
        )
        self.assertEqual(d.kind, "search")
        self.assertEqual(d.text, "A fictional metal.")

    def test_unobtanium_search_failure_is_miss(self):
        d = route(
            "What is unobtanium",
            self.index,
            search_enabled=True,
            search_fn=lambda q: None,
        )
        self.assertEqual(d.kind, "miss")
        self.assertEqual(d.text, MISS_HINT)

    def test_no_search_flag_skips_wikipedia(self):
        d = route("What is unobtanium", self.index, search_enabled=False)
        self.assertEqual(d.kind, "miss")

    def test_learned_hit_replays_extract_not_generate_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "learned.jsonl"
            remember_search_hit(
                self.index,
                path,
                "tell me about ford",
                "Ford Motor Company is an American automaker.",
            )
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            rec = json.loads(lines[0])
            self.assertEqual(rec["query"]["user"], "tell me about ford")

            d = route("tell me about ford", self.index, search_fn=lambda q: "SHOULD NOT SEARCH")
            self.assertEqual(d.kind, "cabinet")
            self.assertEqual(d.detail, "replay")
            self.assertEqual(d.text, "Ford Motor Company is an American automaker.")

            d2 = route("what is a ford ?", self.index, search_fn=lambda q: "SHOULD NOT SEARCH")
            self.assertEqual(d2.kind, "cabinet")
            self.assertEqual(d2.detail, "replay")

            reloaded = CabinetIndex()
            merge_cabinet(reloaded, path, source="learned")
            from training.router import alias_learned_topics
            alias_learned_topics(reloaded)
            d3 = route("what is a ford ?", reloaded, search_fn=lambda q: "SHOULD NOT SEARCH")
            self.assertEqual(d3.kind, "cabinet")
            self.assertEqual(len(reloaded), 1)

    def test_looks_like_fact_question(self):
        self.assertTrue(looks_like_fact_question("What is unobtanium"))
        self.assertTrue(looks_like_fact_question("what is a ford ?"))
        self.assertTrue(looks_like_fact_question("User: Tell me about widget"))
        self.assertTrue(looks_like_fact_question("first 5 prime numbers"))
        self.assertFalse(looks_like_fact_question("2+2"))
        self.assertFalse(looks_like_fact_question("once upon a time"))

    def test_short_topic_phrase_searches(self):
        seen = []

        def capture(q):
            seen.append(q)
            return "The first five primes are 2, 3, 5, 7 and 11."

        d = route("first 5 prime numbers", self.index, search_fn=capture)
        self.assertEqual(d.kind, "search")
        self.assertEqual(seen, ["first 5 prime numbers"])
        self.assertIn("2, 3, 5, 7", d.text)

    
    def test_article_alias_inventor_the(self):
        self.index.add("Who invented Plough?", "Ernesto Schiaparelli is credited with inventing Plough.")
        alias_trained_topics(self.index)
        d = route("Who invented the plough?", self.index, search_enabled=False)
        self.assertEqual(d.kind, "cabinet")
        self.assertEqual(d.fact.user, "Who invented Plough?")
        self.assertEqual(d.text, "User: Who invented Plough? Assistant:")
        self.assertEqual(d.match_type, "trained_article")
        self.assertEqual(d.canonical, "Who invented Plough?")

    def test_article_alias_capital_the(self):
        self.index.add("What is the capital of Netherlands?", "The capital of Netherlands is Amsterdam.")
        alias_trained_topics(self.index)
        d = route("What is the capital of the Netherlands?", self.index, search_enabled=False)
        self.assertEqual(d.kind, "cabinet")
        self.assertEqual(d.fact.user, "What is the capital of Netherlands?")

    def test_trained_topic_beats_learned_overlay(self):
        self.index.add(
            "What is sequential layer streaming?",
            "layer_strategy=stream loads one transformer block.",
        )
        self.index.add(
            "what is layer streaming ?",
            "Howdy is an American streaming service.",
            source="learned",
        )
        alias_trained_topics(self.index)
        d = route("what is layer streaming ?", self.index, search_enabled=False)
        self.assertEqual(d.kind, "cabinet")
        self.assertEqual(d.detail, "generate")
        self.assertEqual(d.fact.user, "What is sequential layer streaming?")
        self.assertEqual(d.fact.source, "trained")
        self.assertIn(d.match_type, ("trained_alias", "trained_article"))

    def test_short_topic_layer_streaming(self):

        self.index.add(
            "What is sequential layer streaming?",
            "layer_strategy=stream loads one transformer block.",
        )
        alias_trained_topics(self.index)
        d = route("what is layer streaming ?", self.index, search_enabled=False)
        self.assertEqual(d.kind, "cabinet")
        self.assertEqual(d.fact.user, "What is sequential layer streaming?")
        self.assertEqual(d.text, "User: What is sequential layer streaming? Assistant:")

    def test_trained_topic_alias_neonics(self):

        alias_trained_topics(self.index)
        d = route("what is neonics", self.index, search_fn=lambda q: "SHOULD NOT SEARCH")
        self.assertEqual(d.kind, "cabinet")
        self.assertEqual(d.detail, "generate")
        self.assertIn("Neonics", d.fact.assistant)

    def test_after_france_followup_does_not_search(self):
        seen = []

        def boom(q):
            seen.append(q)
            self.fail("search must not run after a cabinet generate")

        first = route("what is the capital of france", self.index, search_fn=boom)
        ents = self.index.entities_of(first.fact)
        d = route(
            "where is paris?",
            self.index,
            search_fn=boom,
            last_entities=ents,
        )
        self.assertEqual(d.kind, "miss")
        self.assertEqual(d.detail, "related_miss")
        self.assertEqual(d.text, RELATED_MISS_HINT)
        self.assertIn("What is the capital of France?", d.related)
        self.assertEqual(seen, [])

    def test_paris_fact_generates_when_present(self):
        self.index.add("Where is Paris?", "Paris is the capital of France.")
        first = route("what is the capital of france", self.index, search_enabled=False)
        ents = self.index.entities_of(first.fact)
        d = route(
            "where is paris?",
            self.index,
            search_fn=lambda q: "SHOULD NOT SEARCH",
            last_entities=ents,
        )
        self.assertEqual(d.kind, "cabinet")
        self.assertEqual(d.detail, "generate")
        self.assertEqual(d.fact.user, "Where is Paris?")

    def test_unobtanium_never_hits_france_or_neonics(self):
        first = route("what is the capital of france", self.index, search_enabled=False)
        ents = self.index.entities_of(first.fact)
        d = route(
            "What is unobtanium",
            self.index,
            search_fn=lambda q: "SHOULD NOT SEARCH",
            last_entities=ents,
        )
        self.assertEqual(d.kind, "miss")
        self.assertEqual(d.detail, "related_miss")
        self.assertIsNone(d.fact)
        self.assertIn("What is the capital of France?", d.related)
        self.assertNotIn("Tell me about Neonics.", d.related)

    def test_v7_logged_prompts_route_without_fuzzy(self):
        path = Path(__file__).resolve().parents[1] / "data" / "chat_facts_v7.jsonl"
        if not path.is_file():
            self.skipTest("data/chat_facts_v7.jsonl missing")
        idx = __import__("training.cabinet_index", fromlist=["load_cabinet"]).load_cabinet(path)
        alias_trained_topics(idx)
        d_stream = route("what is layer streaming ?", idx, search_enabled=False)
        self.assertEqual(d_stream.kind, "cabinet")
        self.assertEqual(d_stream.fact.user, "What is sequential layer streaming?")
        d_plough = route("Who invented the plough?", idx, search_enabled=False)
        self.assertNotEqual(d_plough.kind, "cabinet")
        d_cubitt = route("What did William Cubitt invent?", idx, search_enabled=False)
        self.assertEqual(d_cubitt.kind, "cabinet")
        self.assertEqual(d_cubitt.match_type, "trained_exact")
        d_unk = route("What is unobtanium", idx, search_enabled=False)
        self.assertEqual(d_unk.kind, "miss")
        self.assertIsNone(d_unk.fact)

    def test_cold_where_is_paris_may_search(self):
        d = route(
            "where is paris?",
            self.index,
            search_fn=lambda q: "Paris extract",
        )
        self.assertEqual(d.kind, "search")
        self.assertEqual(d.text, "Paris extract")
