"""Router fixture eval and teacher-forced rank helper (no Metal)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.eval_cabinet_bindings import eval_router, load_fixture
from training.cabinet_diagnostics import score_teacher_forced, token_rank_from_logits
from training.cabinet_index import CabinetIndex
from training.router import alias_trained_topics


class BindingEvalTests(unittest.TestCase):
    def test_token_rank_top1(self):
        logits = np.array([0.1, 3.0, 0.2], dtype=np.float32)
        rec = token_rank_from_logits(logits, 1, k=2)
        self.assertEqual(rec["rank"], 1)
        self.assertGreater(rec["prob"], 0.5)

    def test_teacher_forced_two_tokens(self):
        # prompt length 2; expected ids 1, 0 predicted at positions 1 and 2
        logits = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        rows = score_teacher_forced(logits, 2, [1, 0], k=2)
        self.assertEqual(rows[0]["rank"], 1)
        self.assertEqual(rows[1]["rank"], 1)

    def test_eval_router_fixture_subset(self):
        idx = CabinetIndex()
        idx.add(
            "What is sequential layer streaming?",
            "layer_strategy=stream loads one transformer block.",
        )
        idx.add("Where is Paris?", "Paris is the capital of France.")
        idx.add("What did William Cubitt invent?", "William Cubitt is credited with inventing penal treadmill.")
        alias_trained_topics(idx)
        rows = [
            {"query": "what is layer streaming?", "intent": "cabinet", "canonical": "What is sequential layer streaming?"},
            {"query": "Where is Paris?", "intent": "cabinet", "canonical": "Where is Paris?"},
            {"query": "What is unobtanium", "intent": "search_or_miss"},
            {"query": "Who invented the plough?", "intent": "miss"},
        ]
        report = eval_router(idx, rows)
        self.assertEqual(report["false_hit"], 0)
        self.assertGreaterEqual(report["router_recall"], 1.0)
        self.assertGreaterEqual(report["router_precision"], 1.0)

    def test_default_fixture_parses(self):
        rows = load_fixture(_ROOT / "data" / "cabinet_binding_eval.jsonl")
        self.assertGreaterEqual(len(rows), 6)


if __name__ == "__main__":
    unittest.main()
