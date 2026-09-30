"""Tests for fact-overfit mix and explicit dataset path loading."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from setup.config_loader import resolve_dataset_corpus
from tools.make_fact_mix import (
    _NEONICS_USER,
    build_pairs,
    load_learned_extras,
    load_user_facts,
    main as make_fact_mix_main,
    neonics_pair,
    write_outputs,
)
from training.cabinet_index import drop_quarantined
from tools.wikidata_to_facts import expand_linked_pairs, linked_variants


class FactMixTests(unittest.TestCase):
    def test_load_user_facts_skips_hash_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "user_facts.txt"
            path.write_text(
                "# ignore this User: Tell me about X. Assistant: skip\n"
                "User: What is the current version? Assistant: Version 0.0.5.\n",
                encoding="utf-8",
            )
            pairs = load_user_facts([path])
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0][0], "What is the current version?")

    def test_neonics_repeated_and_written(self):
        pairs = build_pairs(user_facts=[], wiki_core=[], user_repeat=3, wiki_repeat=1)
        self.assertGreaterEqual(len(pairs), 3)
        self.assertTrue(all(p[0] == _NEONICS_USER for p in pairs[:3]))
        with tempfile.TemporaryDirectory() as tmp:
            txt = Path(tmp) / "fact_overfit.txt"
            jsonl = Path(tmp) / "fact_overfit_train.jsonl"
            n = write_outputs(pairs, txt, jsonl)
            self.assertEqual(n, len(pairs))
            text = txt.read_text(encoding="utf-8")
            self.assertIn("Tell me about Neonics.", text)
            self.assertIn("banning of Neonics", text)
            rec = json.loads(jsonl.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(rec["query"]["user"], _NEONICS_USER)

    def test_resolve_path_jsonl_not_whole_data_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "fact_overfit_train.jsonl"
            user, assistant = neonics_pair()
            jsonl.write_text(
                json.dumps({"query": {"user": user}, "response": {"assistant": assistant}}) + "\n",
                encoding="utf-8",
            )
            (Path(tmp) / "train.txt").write_text("this should not be loaded\n" * 5, encoding="utf-8")
            corpus = resolve_dataset_corpus(
                {"name": "ignored", "combine": True, "path": str(jsonl)},
                data_dir=tmp,
            )
            self.assertEqual(len(corpus), 1)
            self.assertIn("Neonics", corpus[0])
            self.assertNotIn("this should not be loaded", corpus[0])
            alias = resolve_dataset_corpus(
                {"name": "ignored", "combine": True, "dataset_path": str(jsonl)},
                data_dir=tmp,
            )
            self.assertEqual(alias, corpus)

    def test_load_drops_conflicting_user_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "user_facts.txt"
            path.write_text(
                "User: What is the capital of Kashmir? Assistant: Srinagar.\n"
                "User: What is the capital of Kashmir? Assistant: Jammu.\n"
                "User: What is the atomic number of Oxygen? Assistant: The atomic number of Oxygen is 8.\n",
                encoding="utf-8",
            )
            pairs = load_user_facts([path])
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0][0], "What is the atomic number of Oxygen?")

    def test_load_learned_extras_skips_trained_keys_and_repeats(self):
        with tempfile.TemporaryDirectory() as tmp:
            learned = Path(tmp) / "learned.jsonl"
            rows = [
                {"query": {"user": "Tell me about Neonics."}, "response": {"assistant": "This has led to the recent banning of Neonics in the EU, however the US and Canada are still using this chemical pesticide."}},
                {"query": {"user": "tell me about python"}, "response": {"assistant": "Python is a programming language."}},
                {"query": {"user": "tell me about python"}, "response": {"assistant": "Python is a programming language."}},
                {"query": {"user": "what is a ford"}, "response": {"assistant": "Python may refer to:"}},
                {"query": {"user": "too long"}, "response": {"assistant": "x" * 600}},
            ]
            learned.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            extras = load_learned_extras(
                learned,
                existing=[neonics_pair()],
                max_assistant_chars=512,
            )
            self.assertEqual(extras, [("tell me about python", "Python is a programming language.")])

    def test_expand_linked_capital_questions(self):
        extra = linked_variants(
            "What is the capital of France?",
            "The capital of France is Paris.",
        )
        users = [p[0] for p in extra]
        self.assertIn("Where is Paris?", users)
        self.assertIn("What country is Paris in?", users)
        expanded = expand_linked_pairs([
            ("What is the capital of France?", "The capital of France is Paris."),
        ])
        self.assertGreater(len(expanded), 1)

    def test_quarantine_drops_plough_cluster(self):
        pairs = [
            ("Who invented Plough?", "Ernesto Schiaparelli is credited with inventing Plough."),
            ("Who invented penal treadmill?", "William Cubitt is credited with inventing penal treadmill."),
            ("What did Ernesto Schiaparelli invent?", "Ernesto Schiaparelli is credited with inventing Plough."),
        ]
        kept = drop_quarantined(pairs)
        users = [p[0] for p in kept]
        self.assertNotIn("Who invented Plough?", users)
        self.assertNotIn("What did Ernesto Schiaparelli invent?", users)
        self.assertIn("Who invented penal treadmill?", users)

    def test_user_only_skips_data_facts_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            user = root / "user_facts.txt"
            user.write_text(
                "User: What is Apple MLX GPT? Assistant: A tiny inspectable GPT.\n",
                encoding="utf-8",
            )
            facts = root / "data" / "facts"
            facts.mkdir(parents=True)
            (facts / "extra.txt").write_text(
                "User: What is boron? Assistant: Boron is element 5.\n",
                encoding="utf-8",
            )
            out_txt = root / "tiny.txt"
            out_jsonl = root / "tiny.jsonl"
            rc = make_fact_mix_main(
                [
                    "--user-facts",
                    str(user),
                    "--user-only",
                    "--user-repeat",
                    "2",
                    "--max-wiki-facts",
                    "0",
                    "--output",
                    str(out_txt),
                    "--jsonl",
                    str(out_jsonl),
                ]
            )
            self.assertEqual(rc, 0)
            text = out_txt.read_text(encoding="utf-8")
            self.assertIn("Apple MLX GPT", text)
            self.assertNotIn("boron", text.casefold())


if __name__ == "__main__":
    unittest.main()
