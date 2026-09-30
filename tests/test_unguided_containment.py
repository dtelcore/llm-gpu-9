"""Crash quarantine, backoff, and train-rate cap. No Metal."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.cabinet_index import normalize_question
from training.unguided.containment import (
    DaemonState,
    load_state,
    on_crash,
    on_gate_fail,
    on_promoted,
    refuse_train_reason,
    resolve_log_cursor,
    save_state,
)
from training.unguided.harvest import harvest_retrain_log


def _row(question: str) -> dict:
    return {
        "typed_question": question,
        "canonical_question": question,
        "expected": "x",
        "classification": "TARGET_MISMATCH",
    }


class ContainmentTests(unittest.TestCase):
    def test_crash_poisons_keys_and_backs_off(self):
        state = DaemonState()
        keys = [normalize_question("Where is Paris?")]
        on_crash(state, keys, 99, reason="non_finite_loss", now=1000.0, base_s=300, max_s=3600)
        self.assertIn(keys[0], state.poison)
        self.assertEqual(state.byte_offset, 99)
        self.assertEqual(refuse_train_reason(state, now=1001.0, max_trains_per_hour=8), "backoff")
        self.assertIsNone(refuse_train_reason(state, now=1400.0, max_trains_per_hour=8))

    def test_reappended_poison_does_not_make_queue_ready(self):
        state = DaemonState()
        questions = [
            "Where is Paris?",
            "What is sequential layer streaming?",
            "How do I disable layer streaming?",
            "How do I force layer streaming?",
            "What did William Cubitt invent?",
        ]
        on_crash(
            state,
            [normalize_question(q) for q in questions],
            10,
            reason="exit_1",
            now=1.0,
            base_s=1,
            max_s=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "retrain.jsonl"
            path.write_text("".join(json.dumps(_row(q)) + "\n" for q in questions), encoding="utf-8")
            result = harvest_retrain_log(
                path, 0, min_queue_size=5, max_queue_size=40, skip_keys=state.skip_keys(),
            )
            self.assertFalse(result.ready)
            self.assertEqual(result.rows, [])
            self.assertGreater(result.byte_offset, 0)

    def test_gate_fail_keeps_offset_until_retry_cap(self):
        state = DaemonState(byte_offset=0)
        keys = ["where is paris"]
        kept = on_gate_fail(
            state, keys, 50, reason="anchor_miss", max_retries=2, now=10.0, base_s=1, max_s=1,
        )
        self.assertEqual(kept, "keep")
        self.assertEqual(state.byte_offset, 0)
        self.assertNotIn("where is paris", state.poison)
        exhausted = on_gate_fail(
            state, keys, 50, reason="anchor_miss", max_retries=2, now=20.0, base_s=1, max_s=1,
        )
        self.assertEqual(exhausted, "gate_retry_exhausted")
        self.assertEqual(state.byte_offset, 50)
        self.assertIn("where is paris", state.poison)

    def test_max_trains_per_hour(self):
        state = DaemonState(recent_trains=[100.0, 200.0])
        self.assertEqual(
            refuse_train_reason(state, now=300.0, max_trains_per_hour=2),
            "max_trains_per_hour",
        )
        self.assertIsNone(refuse_train_reason(state, now=3800.0, max_trains_per_hour=2))

    def test_truncate_clamps_without_clearing_poison(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "retrain.jsonl"
            path.write_text("xxxxxxxxxxxxxxxxxxxx", encoding="utf-8")
            state = DaemonState(byte_offset=20, poison={"where is paris"})
            resolve_log_cursor(path, state)
            path.write_text("", encoding="utf-8")
            resolve_log_cursor(path, state)
            self.assertEqual(state.byte_offset, 0)
            self.assertIn("where is paris", state.poison)

    def test_promoted_keys_are_consumed_not_poison(self):
        state = DaemonState()
        on_promoted(state, ["where is paris"], 12)
        self.assertIn("where is paris", state.consumed)
        self.assertNotIn("where is paris", state.poison)
        self.assertEqual(state.consecutive_failures, 0)

    def test_state_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            state = DaemonState(byte_offset=7, poison={"a"}, consumed={"b"})
            save_state(path, state, now=50.0)
            loaded = load_state(path)
            self.assertEqual(loaded.byte_offset, 7)
            self.assertEqual(loaded.poison, {"a"})
            self.assertEqual(loaded.consumed, {"b"})


if __name__ == "__main__":
    unittest.main()
