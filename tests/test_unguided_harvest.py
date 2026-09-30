"""Harvester: offset cursor, dedupe, quarantine, closed variants, min-queue."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.make_fact_mix import write_outputs
from training.cabinet_index import normalize_question
from training.unguided.harvest import (
    ANCHOR_QUESTIONS,
    build_bounded_mix,
    closed_retrain_variants,
    harvest_retrain_log,
)


def _row(question: str, *, classification: str = "TARGET_MISMATCH", expected: str = "Paris.") -> dict:
    return {
        "typed_question": question,
        "canonical_question": question,
        "expected": expected,
        "classification": classification,
    }


class HarvestTests(unittest.TestCase):
    def test_min_queue_does_not_advance_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "retrain.jsonl"
            path.write_text(
                json.dumps(_row("Where is Paris?")) + "\n" + json.dumps(_row("What is sequential layer streaming?")) + "\n",
                encoding="utf-8",
            )
            result = harvest_retrain_log(path, 0, min_queue_size=5, max_queue_size=40)
            self.assertFalse(result.ready)
            self.assertEqual(result.byte_offset, 0)
            self.assertEqual(len(result.rows), 2)

    def test_ready_dedupes_and_advances(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "retrain.jsonl"
            rows = [
                _row("Where is Paris?"),
                _row("where is paris?"),
                _row("What is sequential layer streaming?"),
                _row("How do I disable layer streaming?"),
                _row("How do I force layer streaming?"),
                _row("What did William Cubitt invent?"),
            ]
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            result = harvest_retrain_log(path, 0, min_queue_size=5, max_queue_size=40)
            self.assertTrue(result.ready)
            self.assertEqual(len(result.rows), 5)
            self.assertGreater(result.byte_offset, 0)
            keys = {normalize_question(r["canonical_question"]) for r in result.rows}
            self.assertEqual(len(keys), 5)

    def test_skip_keys_do_not_count_toward_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "retrain.jsonl"
            rows = [
                _row("Where is Paris?"),
                _row("What is sequential layer streaming?"),
                _row("How do I disable layer streaming?"),
                _row("How do I force layer streaming?"),
                _row("What did William Cubitt invent?"),
            ]
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            banned = {normalize_question("Where is Paris?")}
            result = harvest_retrain_log(
                path, 0, min_queue_size=5, max_queue_size=40, skip_keys=banned,
            )
            self.assertFalse(result.ready)
            self.assertEqual(len(result.rows), 4)

    def test_offset_past_eof_clamps_not_replays(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "retrain.jsonl"
            path.write_text(json.dumps(_row("Where is Paris?")) + "\n", encoding="utf-8")
            result = harvest_retrain_log(path, 10**9, min_queue_size=1, max_queue_size=40)
            self.assertFalse(result.ready)
            self.assertEqual(result.rows, [])
            self.assertLessEqual(result.byte_offset, path.stat().st_size)

    def test_skips_other_classifications(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "retrain.jsonl"
            rows = [_row("Where is Paris?", classification="MATCH") for _ in range(6)]
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            result = harvest_retrain_log(path, 0, min_queue_size=1, max_queue_size=40)
            self.assertFalse(result.ready)
            self.assertEqual(result.rows, [])

    def test_inventor_variants_are_closed(self):
        variants = closed_retrain_variants(
            "Who invented penal treadmill?",
            "William Cubitt is credited with inventing penal treadmill.",
        )
        questions = {normalize_question(q) for q, _a in variants}
        self.assertIn(normalize_question("What did William Cubitt invent?"), questions)
        self.assertTrue(any("the penal treadmill" in q for q in questions))

    def test_bounded_mix_includes_anchors_and_drops_quarantine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broad = root / "broad.jsonl"
            write_outputs(
                [
                    ("What is the capital of France?", "The capital of France is Paris."),
                    ("Where is Paris?", "Paris is the capital of France."),
                    ("What did William Cubitt invent?", "William Cubitt is credited with inventing penal treadmill."),
                    ("Who invented Plough?", "Ernesto Schiaparelli is credited with inventing Plough."),
                ],
                root / "broad.txt",
                broad,
            )
            user = root / "user_facts.txt"
            user.write_text(
                "User: What is sequential layer streaming? Assistant: layer_strategy=stream loads one block.\n"
                "User: How do I disable layer streaming? Assistant: Pass --no-layer-stream.\n"
                "User: How do I force layer streaming? Assistant: Pass --layer-stream.\n",
                encoding="utf-8",
            )
            harvested = [
                _row("Where is Paris?", expected="Paris is the capital of France."),
                _row("What did William Cubitt invent?", expected="William Cubitt is credited with inventing penal treadmill."),
            ]
            stats = build_bounded_mix(
                harvested,
                broad_jsonl=broad,
                user_facts=user,
                out_txt=root / "out.txt",
                out_jsonl=root / "out.jsonl",
                retrain_weight=1,
                anchor_weight=1,
                broad_weight=1,
                drop_quarantine=True,
            )
            text = (root / "out.txt").read_text(encoding="utf-8")
            self.assertIn("Where is Paris?", text)
            self.assertIn("What is sequential layer streaming?", text)
            self.assertNotIn("Who invented Plough?", text)
            self.assertGreater(stats["n"], 0)
            self.assertTrue(any(normalize_question(q) == normalize_question("Where is Paris?") for q in ANCHOR_QUESTIONS))


if __name__ == "__main__":
    unittest.main()
