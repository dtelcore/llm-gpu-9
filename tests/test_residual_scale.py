"""GPT-2 residual_scale parsing and legacy default."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.config import GPTConfig, parse_residual_scale


def _model(layers=6, **extra):
    d = {
        "name": "t",
        "vocab_size": 32,
        "max_len": 16,
        "embedding_dim": 32,
        "num_heads": 4,
        "num_layers": layers,
    }
    d.update(extra)
    return d


class ResidualScaleTests(unittest.TestCase):
    def test_legacy_checkpoint_is_unscaled(self):
        cfg = GPTConfig(_model())
        self.assertEqual(cfg.residual_scale, 1.0)

    def test_gpt2_flag(self):
        self.assertAlmostEqual(parse_residual_scale(True, 16), 1.0 / math.sqrt(32))
        cfg = GPTConfig(_model(layers=6, residual_scale=True))
        self.assertAlmostEqual(cfg.residual_scale, 1.0 / math.sqrt(12))
        self.assertAlmostEqual(cfg.to_dict()["residual_scale"], cfg.residual_scale)


if __name__ == "__main__":
    unittest.main()
