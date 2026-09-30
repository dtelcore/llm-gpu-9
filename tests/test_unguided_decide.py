"""Unit tests for the pure decide() engine. No Metal required."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.unguided.decide import Decision, DecideContext, decide


def _ctx(**kwargs) -> DecideContext:
    base = dict(
        step=100,
        max_steps=500,
        wall_s=60.0,
        max_wall_s=7200.0,
        val_loss=0.9,
        best_val_loss=1.0,
        recent_val_losses=[1.1, 1.0, 0.95],
        nan_detected=False,
        cabinet_exact_match=0.5,
        policy={
            "loss_spike_ratio": 2.0,
            "early_stop_patience": 4,
            "remix_if": {
                "cabinet_exact_match_below": 0.15,
                "after_steps": 50,
            },
        },
        no_improvement_count=0,
    )
    base.update(kwargs)
    return DecideContext(**base)


class DecideTests(unittest.TestCase):
    def test_nan_aborts(self):
        result = decide(_ctx(nan_detected=True))
        self.assertEqual(result.action, Decision.ABORT_SPIKE)
        self.assertIn("non_finite", result.reason)

    def test_spike_aborts(self):
        result = decide(_ctx(val_loss=3.0, recent_val_losses=[1.0, 1.1, 0.9]))
        self.assertEqual(result.action, Decision.ABORT_SPIKE)
        self.assertIn("spike", result.reason)

    def test_promote_on_best(self):
        result = decide(_ctx(val_loss=0.8, best_val_loss=1.0))
        self.assertEqual(result.action, Decision.PROMOTE)
        self.assertTrue(result.promote)

    def test_continue_when_ok(self):
        result = decide(_ctx(val_loss=1.05, best_val_loss=1.0, no_improvement_count=1))
        self.assertEqual(result.action, Decision.CONTINUE)
        self.assertFalse(result.promote)

    def test_early_stop_on_patience(self):
        result = decide(_ctx(no_improvement_count=4, val_loss=1.05, best_val_loss=1.0))
        self.assertEqual(result.action, Decision.EARLY_STOP)

    def test_remix_abort(self):
        result = decide(_ctx(
            step=100,
            cabinet_exact_match=0.05,
            policy={
                "loss_spike_ratio": 2.0,
                "early_stop_patience": 4,
                "remix_if": {"cabinet_exact_match_below": 0.15, "after_steps": 50},
                "recipe": "legacy/setup/chat_facts_v7_config.json",
            },
        ))
        self.assertEqual(result.action, Decision.ABORT_REMIX)
        self.assertIsNotNone(result.next_mix_recipe)
        self.assertIn("suggested_cmd", result.next_mix_recipe)

    def test_max_steps(self):
        result = decide(_ctx(step=500, max_steps=500))
        self.assertEqual(result.action, Decision.STOP_LIMIT)
        self.assertIn("max_steps", result.reason)

    def test_wall_clock(self):
        result = decide(_ctx(wall_s=8000, max_wall_s=7200))
        self.assertEqual(result.action, Decision.STOP_LIMIT)
        self.assertIn("max_wall_s", result.reason)


if __name__ == "__main__":
    unittest.main()
