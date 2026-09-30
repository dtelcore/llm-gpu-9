"""Per-layer grad-norm grouping and health flags."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.layer_health import format_layer_grad_line, summarize_layer_grad_norms


def _block(scale: float) -> dict:
    return {
        "qkv_proj": np.full((2, 2), scale, dtype=np.float32),
        "attn_out_proj": np.full((2, 2), scale, dtype=np.float32),
        "ln1_gamma": np.full((2,), scale, dtype=np.float32),
        "mlp_expand": np.full((2, 2), scale, dtype=np.float32),
        "mlp_contract": np.full((2, 2), scale, dtype=np.float32),
        "ln2_gamma": np.full((2,), scale, dtype=np.float32),
    }


class LayerHealthTests(unittest.TestCase):
    def test_balanced_layers_are_healthy(self):
        grads = {}
        for i, scale in enumerate((0.1, 0.12, 0.11)):
            for name, arr in _block(scale).items():
                grads[f"layer_{i}.{name}"] = arr
        grads["token_embedding"] = np.ones((3, 2), dtype=np.float32) * 0.05
        grads["final_ln_gamma"] = np.ones((2,), dtype=np.float32) * 0.08
        report = summarize_layer_grad_norms(grads, 3)
        self.assertTrue(report["healthy"])
        self.assertLess(report["ratio"], 5.0)
        self.assertEqual(report["dead"], [])
        self.assertIn("ratio=", format_layer_grad_line(report, step=1))

    def test_dead_early_layer_is_unhealthy(self):
        grads = {}
        for name, arr in _block(0.0).items():
            grads[f"layer_0.{name}"] = arr
        for name, arr in _block(0.2).items():
            grads[f"layer_1.{name}"] = arr
        report = summarize_layer_grad_norms(grads, 2)
        self.assertFalse(report["healthy"])
        self.assertEqual(report["dead"], [0])
        self.assertIn("UNHEALTHY", format_layer_grad_line(report))


if __name__ == "__main__":
    unittest.main()
