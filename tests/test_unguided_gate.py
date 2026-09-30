"""Promotion gate fails closed when anchors miss."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.cabinet_index import normalize_question
from training.unguided.gate import gate_eval
from training.unguided.harvest import ANCHOR_QUESTIONS


def _hits(ok: bool = True) -> dict:
    return {normalize_question(q): ok for q in ANCHOR_QUESTIONS}


class GateTests(unittest.TestCase):
    def test_fail_closed_without_summary(self):
        result = gate_eval(None)
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "missing_eval_summary")

    def test_fail_when_anchor_misses(self):
        hits = _hits(True)
        hits[normalize_question("Where is Paris?")] = False
        result = gate_eval({"anchor_hits": hits, "harvested_entity_swaps": 0})
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "anchor_miss")
        self.assertIn(normalize_question("Where is Paris?"), result.missing_anchors)

    def test_fail_on_new_entity_swap(self):
        result = gate_eval({"anchor_hits": _hits(True), "harvested_entity_swaps": 1})
        self.assertFalse(result.passed)
        self.assertIn("harvested_entity_swaps", result.reason)

    def test_pass_when_anchors_hold(self):
        result = gate_eval({
            "anchor_hits": _hits(True),
            "harvested_entity_swaps": 0,
            "harvested_exact": None,
        })
        self.assertTrue(result.passed)
        self.assertEqual(result.reason, "ok")


if __name__ == "__main__":
    unittest.main()
