"""Train-step preflight: refuse a step that exceeds the process budget."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.cuda.env import (
    PEAK_TRANSIENT_BYTES,
    PROCESS_BUDGET_BYTES,
    MemoryBudgetError,
    process_budget_exceeded,
)
from training.memory_preflight import assert_train_fits_budget, estimate_train_step_bytes


class TrainMemoryPreflightTests(unittest.TestCase):
    def test_story_sub1m_fits(self):
        estimated = estimate_train_step_bytes(
            n_params=830_000,
            batch_size=8,
            max_len=128,
            embedding_dim=128,
            num_heads=8,
            num_layers=4,
            grad_accum=2,
        )
        self.assertLess(estimated, PROCESS_BUDGET_BYTES)
        assert_train_fits_budget(
            n_params=830_000,
            batch_size=8,
            max_len=128,
            embedding_dim=128,
            num_heads=8,
            num_layers=4,
            grad_accum=2,
        )

    def test_c256_l4_t256_batch4_fits(self):
        estimated = estimate_train_step_bytes(
            n_params=3_900_000,
            batch_size=4,
            max_len=256,
            embedding_dim=256,
            num_heads=8,
            num_layers=4,
            grad_accum=4,
        )
        self.assertLess(estimated, PROCESS_BUDGET_BYTES)

    def test_l32_c256_t256_batch8_refused(self):
        with self.assertRaises(MemoryBudgetError) as ctx:
            assert_train_fits_budget(
                n_params=25_362_847,
                batch_size=8,
                max_len=256,
                embedding_dim=256,
                num_heads=16,
                num_layers=32,
                grad_accum=2,
            )
        msg = str(ctx.exception)
        self.assertIn("L=32", msg)
        self.assertIn("5 GB", msg)

    def test_peak_transient_does_not_trip_when_active_is_under(self):
        active = 638 * 1024 * 1024
        peak = PROCESS_BUDGET_BYTES + 20 * 1024 * 1024  # the run8+16 first-step miss
        self.assertFalse(
            process_budget_exceeded(active, peak, PROCESS_BUDGET_BYTES, PEAK_TRANSIENT_BYTES)
        )
        self.assertTrue(
            process_budget_exceeded(
                PROCESS_BUDGET_BYTES + 1, peak, PROCESS_BUDGET_BYTES, PEAK_TRANSIENT_BYTES
            )
        )
        self.assertTrue(
            process_budget_exceeded(
                active,
                PROCESS_BUDGET_BYTES + PEAK_TRANSIENT_BYTES + 1,
                PROCESS_BUDGET_BYTES,
                PEAK_TRANSIENT_BYTES,
            )
        )


if __name__ == "__main__":
    unittest.main()
