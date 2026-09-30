"""Phase 1 English unguided path: val-only eval, OOD stop probe, no cabinet remix."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import unguided_trainer
from training.unguided.decide import DecideContext, Decision, NextStepContext, decide, decide_next_step
from training.unguided.eval_suite import run_eval_suite
from training.unguided.prober import (
    PHASE1_OOD_PROMPTS,
    looks_like_cabinet_dump,
    render_markdown,
    run_english_stop_probe,
    write_probe_reports,
)


class EvalSuiteEnglishTests(unittest.TestCase):
    def test_english_skips_cabinet_index(self):
        session = SimpleNamespace(
            model=object(),
            val_dataset=None,
            args=SimpleNamespace(seed=42),
            tokenizer=object(),
            dataset_path="data/train.txt",
        )
        with mock.patch("training.eval.evaluate_val_loss", return_value=(1.25, 3.49)):
            rec = run_eval_suite(session, {"probe_mode": "english"})
        self.assertEqual(rec.val_loss, 1.25)
        self.assertIsNone(rec.cabinet_exact_match)
        self.assertEqual(rec.anchor_hits, {})


class DecideEnglishTests(unittest.TestCase):
    def test_english_never_remixes(self):
        result = decide(
            DecideContext(
                step=500,
                max_steps=10000,
                wall_s=10.0,
                max_wall_s=28800.0,
                val_loss=1.2,
                best_val_loss=1.3,
                recent_val_losses=[1.4, 1.3],
                nan_detected=False,
                cabinet_exact_match=0.0,
                policy={
                    "probe_mode": "english",
                    "loss_spike_ratio": 2.0,
                    "early_stop_patience": 40,
                    "remix_if": {"cabinet_exact_match_below": -1.0, "after_steps": 1},
                },
            )
        )
        self.assertEqual(result.action, Decision.PROMOTE)

    def test_next_step_english_hold_at_cap(self):
        rec = decide_next_step(
            NextStepContext(
                step=10000,
                max_steps=10000,
                val_loss=1.1,
                best_val_loss=1.1,
                cabinet_exact_match=None,
                generate_exact_rate=None,
                generate_swap_rate=0.0,
                ood_mix_copies=1,
                ood_n=5,
                policy={"probe_mode": "english"},
            )
        )
        self.assertEqual(rec.mode, "english_foundation")
        self.assertEqual(rec.primary, "hold")
        self.assertFalse(rec.understands)


class EnglishProbeTests(unittest.TestCase):
    def test_dump_heuristic(self):
        self.assertTrue(looks_like_cabinet_dump("The capital of France is Paris."))
        self.assertFalse(looks_like_cabinet_dump("the plants used sunlight to grow."))

    def test_stop_probe_markdown(self):
        class Tok:
            def encode(self, text):
                return [1, 2]

            def decode(self, ids):
                return " used sunlight to make sugar."

        class M:
            def generate(self, *args, **kwargs):
                return [1, 2, 3, 4]

        report = run_english_stop_probe(model=M(), tokenizer=Tok(), step=10, checkpoint="x")
        self.assertEqual(report["probe_mode"], "english")
        self.assertEqual(report["ood_n"], len(PHASE1_OOD_PROMPTS))
        self.assertEqual(report["ood_mix_copies"], 0)
        verdict = decide_next_step(
            NextStepContext(
                step=10,
                max_steps=10000,
                val_loss=2.0,
                best_val_loss=2.0,
                cabinet_exact_match=None,
                generate_exact_rate=None,
                generate_swap_rate=0.0,
                ood_mix_copies=report["ood_mix_copies"],
                ood_n=report["ood_n"],
                policy={"probe_mode": "english"},
            )
        )
        md = render_markdown(report, verdict)
        self.assertIn("Once upon a time in a valley", md)
        self.assertIn("Out-of-distribution prose", md)
        self.assertNotIn("This checkpoint is **memorizing**", md)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_probe_reports(Path(tmp), report, verdict)
            self.assertTrue(path.is_file())


class Phase1DryRunTests(unittest.TestCase):
    def test_phase1_dry_run(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = unguided_trainer.main([
                "--config", str(_ROOT / "setup" / "english_phase1_config.json"),
                "--policy", str(_ROOT / "setup" / "unguided_phase1_policy.json"),
                "--name", "English-Phase1",
                "--dry-run",
            ])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("data/train.txt", out)
        self.assertIn("probe_mode:    english", out)
        self.assertIn("max_steps:     10000", out)
        self.assertIn("C=256", out)
        self.assertIn("L=6", out)
        self.assertIn("No Metal init", out)


if __name__ == "__main__":
    unittest.main()
