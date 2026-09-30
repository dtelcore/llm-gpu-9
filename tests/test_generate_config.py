"""Unit tests for generate_config.py."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import generate_config as gc


class GenerateConfigTests(unittest.TestCase):
    def setUp(self):
        self.base = gc.load_recipe(_ROOT / "legacy" / "setup" / "chat_c256_l6_config.json")

    def test_clone_widen_c(self):
        cfg = gc.apply_overrides(self.base, embedding_dim=384, num_heads=8, num_layers=8)
        self.assertEqual(cfg["model"]["embedding_dim"], 384)
        self.assertEqual(cfg["model"]["num_layers"], 8)
        self.assertTrue(cfg["model"]["residual_scale"])
        self.assertEqual(cfg["dataset"]["name"], "chat_train")
        self.assertFalse(cfg["dataset"]["combine"])
        self.assertEqual(cfg["hyperparameters"]["batch_size"], 4)

    def test_heads_must_divide_c(self):
        with self.assertRaises(ValueError):
            gc.apply_overrides(self.base, embedding_dim=256, num_heads=7)

    def test_deep_stack_requires_residual_scale(self):
        with self.assertRaises(ValueError):
            gc.apply_overrides(self.base, num_layers=12, residual_scale=False)

    def test_write_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "chat_c384_l8_config.json"
            code = gc.main([
                "--from", str(_ROOT / "legacy" / "setup" / "chat_c256_l6_config.json"),
                "--embedding-dim", "384",
                "--num-layers", "8",
                "--batch-size", "4",
                "--grad-accum", "4",
                "--layer-strategy", "stream",
                "--no-prompt",
                "--output", str(dest),
            ])
            self.assertEqual(code, 0)
            loaded = json.loads(dest.read_text(encoding="utf-8"))
            self.assertEqual(loaded["model"]["embedding_dim"], 384)
            self.assertEqual(loaded["model"]["layer_strategy"], "stream")
            self.assertTrue(loaded["model"]["residual_scale"])

    def test_cli_refuses_clobber_base(self):
        code = gc.main([
            "--from", str(_ROOT / "legacy" / "setup" / "chat_c256_l6_config.json"),
            "--no-prompt",
            "--output", str(_ROOT / "legacy" / "setup" / "chat_c256_l6_config.json"),
        ])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
