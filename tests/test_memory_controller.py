"""Memory controller: autoscale batch/context/activations to the 2 GB cap."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.cuda.env import PROCESS_BUDGET_BYTES, MemoryBudgetError
from training.memory_controller import (
    apply_train_plan,
    estimate_train_bytes,
    plan_generate,
    plan_train,
    process_budget_bytes,
    usable_bytes,
)


class MemoryControllerTests(unittest.TestCase):
    def test_budget_is_hardcoded_2gb(self):
        self.assertEqual(process_budget_bytes(), 2 * 1024 ** 3)
        self.assertEqual(process_budget_bytes(), PROCESS_BUDGET_BYTES)
        self.assertLess(usable_bytes(0.15), process_budget_bytes())

    def test_story_l6_unchanged(self):
        plan = plan_train(
            n_params=6_000_000,
            batch_size=4,
            max_len=256,
            embedding_dim=256,
            num_heads=8,
            num_layers=6,
            vocab_size=4112,
            grad_accum=4,
            gradient_checkpointing=False,
        )
        self.assertTrue(plan.fits)
        self.assertEqual(plan.batch_size, 4)
        self.assertEqual(plan.max_len, 256)
        self.assertEqual(plan.grad_accum, 4)
        self.assertFalse(plan.gradient_checkpointing)
        self.assertFalse(plan.eval_per_layer)
        self.assertEqual(plan.actions, [])
        self.assertLess(plan.estimated_bytes, usable_bytes())

    def test_l32_autoscales_under_2gb(self):
        plan = plan_train(
            n_params=25_362_847,
            batch_size=8,
            max_len=256,
            embedding_dim=256,
            num_heads=16,
            num_layers=32,
            vocab_size=4112,
            grad_accum=2,
            gradient_checkpointing=False,
        )
        self.assertTrue(plan.fits, msg=plan.summary_line())
        self.assertLessEqual(plan.estimated_bytes, plan.usable_bytes)
        self.assertTrue(plan.eval_per_layer)
        self.assertTrue(plan.gradient_checkpointing)
        self.assertEqual(plan.requested_batch, 8)
        self.assertLessEqual(plan.batch_size, 8)
        self.assertEqual(plan.batch_size * plan.grad_accum, 8 * 2)
        self.assertIn("eval_per_layer", plan.actions)

    def test_no_autoscale_refuses_l32(self):
        plan = plan_train(
            n_params=25_362_847,
            batch_size=8,
            max_len=256,
            embedding_dim=256,
            num_heads=16,
            num_layers=32,
            vocab_size=4112,
            grad_accum=2,
            autoscale=False,
        )
        self.assertFalse(plan.fits)
        self.assertGreater(plan.estimated_bytes, PROCESS_BUDGET_BYTES)

    def test_giant_params_refuse(self):
        with self.assertRaises(MemoryBudgetError) as ctx:
            class _Cfg:
                max_len = 32
                embedding_dim = 1024
                num_heads = 16
                num_layers = 48
                vocab_size = 32000
                gradient_checkpointing = False

            apply_train_plan(
                _Cfg(),
                {"batch_size": 1, "gradient_accumulation_steps": 1},
                n_params=400_000_000,
                autoscale=True,
            )
        self.assertIn("2 GB", str(ctx.exception))

    def test_generate_caps_new_tokens_to_context(self):
        plan = plan_generate(
            n_params=6_000_000,
            max_len=256,
            embedding_dim=256,
            num_heads=8,
            num_layers=6,
            vocab_size=4112,
            prompt_len=200,
            max_new_tokens=200,
        )
        self.assertTrue(plan.fits)
        self.assertEqual(plan.max_new_tokens, 56)

    def test_generate_stream_estimate_is_positive(self):
        common = dict(
            max_len=256,
            embedding_dim=256,
            vocab_size=4112,
            prompt_len=1,
            max_new_tokens=80,
            layer_strategy="stream",
        )
        l6 = plan_generate(
            n_params=6_000_000, num_heads=8, num_layers=6, **common,
        )
        self.assertGreater(l6.estimated_bytes, 0, msg=l6.summary_line())
        self.assertTrue(l6.fits)
        l32 = plan_generate(
            n_params=25_362_847, num_heads=16, num_layers=32, **common,
        )
        self.assertGreater(l32.estimated_bytes, 0, msg=l32.summary_line())
        self.assertLess(l32.estimated_bytes, PROCESS_BUDGET_BYTES)
        self.assertTrue(l32.fits)

    def test_estimate_checkpoint_smaller_than_full(self):
        common = dict(
            n_params=25_000_000,
            batch_size=4,
            max_len=256,
            embedding_dim=256,
            num_heads=8,
            num_layers=32,
            vocab_size=4112,
            eval_per_layer=True,
            grad_accum=4,
        )
        full = estimate_train_bytes(**common, gradient_checkpointing=False)
        ckpt = estimate_train_bytes(**common, gradient_checkpointing=True)
        self.assertLess(ckpt.total, full.total)

    def test_l32_stream_estimate_fits_resident_does_not(self):
        common = dict(
            n_params=25_362_847,
            batch_size=4,
            max_len=256,
            embedding_dim=256,
            num_heads=16,
            num_layers=32,
            vocab_size=4112,
            grad_accum=4,
        )
        stream = estimate_train_bytes(**common, layer_strategy="stream", eval_per_layer=True)
        resident = estimate_train_bytes(**common, layer_strategy="resident")
        self.assertLess(stream.total, PROCESS_BUDGET_BYTES, msg=stream.as_mb())
        self.assertLess(stream.total, usable_bytes())
        self.assertGreater(resident.total, PROCESS_BUDGET_BYTES, msg=resident.as_mb())

    def test_autoscale_streams_before_shrinking_t(self):
        common = dict(
            n_params=25_362_847,
            batch_size=4,
            max_len=2048,
            embedding_dim=256,
            num_heads=16,
            num_layers=32,
            vocab_size=4112,
            grad_accum=4,
            gradient_checkpointing=False,
        )
        plan = plan_train(**common, allow_stream=True)
        self.assertTrue(plan.fits, msg=plan.summary_line())
        self.assertEqual(plan.layer_strategy, "stream")
        self.assertIn("layer_strategy=stream", plan.actions)

        plan_no = plan_train(**common, allow_stream=False)
        self.assertTrue(plan_no.fits, msg=plan_no.summary_line())
        self.assertEqual(plan_no.layer_strategy, "resident")
        self.assertLess(plan_no.max_len, 2048)
        self.assertGreaterEqual(plan.max_len, plan_no.max_len)

    def test_no_layer_stream_refuses_to_enable(self):
        plan = plan_train(
            n_params=25_362_847,
            batch_size=4,
            max_len=256,
            embedding_dim=256,
            num_heads=16,
            num_layers=32,
            vocab_size=4112,
            grad_accum=4,
            autoscale=False,
            allow_stream=False,
            layer_strategy="resident",
        )
        self.assertFalse(plan.fits)
        self.assertEqual(plan.layer_strategy, "resident")


if __name__ == "__main__":
    unittest.main()
