"""Focused tests for the Chat 5M (~5M param) preset and ready config."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from setup.dataset_setup import COMBINED_DATASET_NAME, recommend_dataset_for_config
from setup.model_config import PRESETS, estimate_vram_footprint
from setup.training_presets import (
    SCALE_PRESET_MENU_ORDER,
    SCALE_PRESETS,
    apply_scale_preset,
    model_from_preset,
)


class Chat5mPresetTests(unittest.TestCase):
    def test_menu_order_exposes_chat_5m(self):
        self.assertEqual(
            SCALE_PRESET_MENU_ORDER,
            ["toy", "story_sub1m", "tiny_stories", "chat_5m"],
        )
        self.assertIn("chat_5m", SCALE_PRESETS)
        self.assertEqual(SCALE_PRESETS["chat_5m"]["dataset"], COMBINED_DATASET_NAME)

    def test_apply_scale_preset_batching(self):
        model, hyperparams, dataset_name = apply_scale_preset("chat_5m", vocab_size=256)
        self.assertEqual(dataset_name, COMBINED_DATASET_NAME)
        self.assertEqual(model["embedding_dim"], 256)
        self.assertEqual(model["num_heads"], 8)
        self.assertEqual(model["num_layers"], 6)
        self.assertEqual(model["max_len"], 128)
        self.assertTrue(model["tie_embeddings"])
        self.assertEqual(model["norm_type"], "rmsnorm")
        self.assertEqual(model["pos_encoding"], "rope")
        self.assertEqual(model["dropout_prob"], 0.0)
        self.assertEqual(hyperparams["batch_size"], 4)
        self.assertEqual(hyperparams["gradient_accumulation_steps"], 4)
        self.assertEqual(hyperparams["window_stride"], 64)
        self.assertAlmostEqual(hyperparams["learning_rate"], 3e-4)
        self.assertEqual(hyperparams["warmup_steps"], 1000)
        self.assertAlmostEqual(hyperparams["min_lr_ratio"], 0.1)
        self.assertEqual(hyperparams["gradient_clip"], 1.0)

    def test_params_near_five_million(self):
        for vocab in (256, 512, 1024):
            cfg = model_from_preset("chat_5m", vocab_size=vocab)
            n = estimate_vram_footprint(cfg)["total_params"]
            self.assertGreaterEqual(
                n,
                4_500_000,
                f"chat_5m below 4.5M params at vocab_size={vocab}: {n}",
            )
            self.assertLessEqual(
                n,
                5_500_000,
                f"chat_5m above 5.5M params at vocab_size={vocab}: {n}",
            )

    def test_architecture_preset_registered(self):
        self.assertIn("chat_5m", PRESETS)
        self.assertEqual(PRESETS["chat_5m"]["embedding_dim"], 256)
        self.assertEqual(PRESETS["chat_5m"]["num_layers"], 6)

    def test_recommends_combined_dataset(self):
        cfg = model_from_preset("chat_5m", vocab_size=256)
        self.assertEqual(recommend_dataset_for_config(cfg), COMBINED_DATASET_NAME)

    def test_ready_config_matches_preset(self):
        config_path = _ROOT / "setup" / "chat_5m_config.json"
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        model, hyperparams, dataset_name = apply_scale_preset("chat_5m", vocab_size=256)
        self.assertEqual(config["model"]["embedding_dim"], model["embedding_dim"])
        self.assertEqual(config["model"]["num_layers"], model["num_layers"])
        self.assertEqual(config["model"]["max_len"], model["max_len"])
        self.assertEqual(config["dataset"]["name"], dataset_name)
        self.assertTrue(config["dataset"]["combine"])
        self.assertEqual(config["dataset"]["tokenizer"], "bpe")
        self.assertEqual(config["dataset"]["bpe_merges"], 200)
        self.assertEqual(config["hyperparameters"]["batch_size"], hyperparams["batch_size"])
        self.assertEqual(
            config["hyperparameters"]["gradient_accumulation_steps"],
            hyperparams["gradient_accumulation_steps"],
        )
        self.assertEqual(config["hyperparameters"]["window_stride"], hyperparams["window_stride"])
        self.assertAlmostEqual(
            config["hyperparameters"]["learning_rate"],
            hyperparams["learning_rate"],
        )


if __name__ == "__main__":
    unittest.main()
