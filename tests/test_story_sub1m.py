"""Focused tests for combined data-dir loading and the Fast Stories (<1M) preset."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from setup.config_loader import is_combined_dataset, resolve_dataset_corpus
from setup.dataset_setup import COMBINED_DATASET_NAME, DatasetLoader
from setup.model_config import PRESETS, estimate_vram_footprint
from setup.training_presets import (
    SCALE_PRESET_MENU_ORDER,
    SCALE_PRESETS,
    apply_scale_preset,
    model_from_preset,
)


class CombinedDirectoryTests(unittest.TestCase):
    def test_combines_sorted_files_and_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "b_second.txt").write_text(
                "beta story one\n\n  \nbeta story two\n",
                encoding="utf-8",
            )
            (data_dir / "a_first.txt").write_text(
                "alpha story one\n\nalpha story two\n",
                encoding="utf-8",
            )
            (data_dir / "notes.md").write_text("ignored\n", encoding="utf-8")

            loader = DatasetLoader(data_dir=str(data_dir), auto_discover=True)
            corpus = loader.load_combined_directory(str(data_dir))

            self.assertEqual(
                corpus,
                [
                    "alpha story one",
                    "alpha story two",
                    "beta story one",
                    "beta story two",
                ],
            )
            self.assertEqual(loader.current_dataset, COMBINED_DATASET_NAME)

    def test_load_by_reserved_name_combines_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "s.txt").write_text("once upon a time\n", encoding="utf-8")
            loader = DatasetLoader(data_dir=str(data_dir), auto_discover=True)
            corpus = loader.load_by_name(COMBINED_DATASET_NAME)
            self.assertEqual(corpus, ["once upon a time"])

    def test_missing_directory_raises(self):
        missing = Path(tempfile.gettempdir()) / "story_sub1m_missing_dir_xyz"
        loader = DatasetLoader(data_dir=str(missing), auto_discover=False)
        with self.assertRaises(FileNotFoundError):
            loader.load_combined_directory(str(missing))

    def test_empty_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            loader = DatasetLoader(data_dir=tmp, auto_discover=True)
            with self.assertRaises(FileNotFoundError):
                loader.load_combined_directory(tmp)

    def test_blank_only_files_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "empty.txt").write_text("\n  \n\n", encoding="utf-8")
            loader = DatasetLoader(data_dir=str(data_dir), auto_discover=True)
            with self.assertRaises(ValueError):
                loader.load_combined_directory(str(data_dir))

    def test_single_file_and_builtin_still_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "tiny_stories.txt").write_text("hello world\n", encoding="utf-8")
            loader = DatasetLoader(data_dir=str(data_dir), auto_discover=True)
            self.assertEqual(loader.load_by_name("tiny_stories"), ["hello world"])
            builtin = loader.load_by_name("minimal")
            self.assertGreaterEqual(len(builtin), 1)

    def test_resolve_dataset_corpus_combine_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "z.txt").write_text("zeta\n", encoding="utf-8")
            (data_dir / "a.txt").write_text("alpha\n", encoding="utf-8")
            corpus = resolve_dataset_corpus(
                {"name": "ignored", "combine": True},
                data_dir=str(data_dir),
            )
            self.assertEqual(corpus, ["alpha", "zeta"])

    def test_resolve_dataset_corpus_reserved_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "s.txt").write_text("story\n", encoding="utf-8")
            corpus = resolve_dataset_corpus(
                {"name": COMBINED_DATASET_NAME},
                data_dir=str(data_dir),
            )
            self.assertEqual(corpus, ["story"])

    def test_inline_corpus_still_wins(self):
        corpus = resolve_dataset_corpus(
            {"name": COMBINED_DATASET_NAME, "combine": True, "corpus": ["inline"]},
            data_dir="missing",
        )
        self.assertEqual(corpus, ["inline"])

    def test_is_combined_dataset(self):
        self.assertTrue(is_combined_dataset({"combine": True}))
        self.assertTrue(is_combined_dataset({"name": COMBINED_DATASET_NAME}))
        self.assertFalse(is_combined_dataset({"name": "tiny_stories"}))


class StorySub1mPresetTests(unittest.TestCase):
    def test_menu_order_exposes_story_sub1m(self):
        self.assertEqual(
            SCALE_PRESET_MENU_ORDER,
            ["toy", "story_sub1m", "tiny_stories", "chat_5m"],
        )
        self.assertIn("story_sub1m", SCALE_PRESETS)
        self.assertEqual(SCALE_PRESETS["story_sub1m"]["dataset"], COMBINED_DATASET_NAME)

    def test_apply_scale_preset_batching(self):
        model, hyperparams, dataset_name = apply_scale_preset("story_sub1m", vocab_size=256)
        self.assertEqual(dataset_name, COMBINED_DATASET_NAME)
        self.assertEqual(model["embedding_dim"], 128)
        self.assertEqual(model["num_heads"], 8)
        self.assertEqual(model["num_layers"], 4)
        self.assertEqual(model["max_len"], 128)
        self.assertTrue(model["tie_embeddings"])
        self.assertEqual(model["norm_type"], "rmsnorm")
        self.assertEqual(model["pos_encoding"], "rope")
        self.assertEqual(model["dropout_prob"], 0.0)
        self.assertEqual(hyperparams["batch_size"], 8)
        self.assertEqual(hyperparams["gradient_accumulation_steps"], 2)
        self.assertEqual(hyperparams["window_stride"], 64)
        self.assertAlmostEqual(hyperparams["learning_rate"], 5e-4)
        self.assertEqual(hyperparams["warmup_steps"], 500)
        self.assertAlmostEqual(hyperparams["min_lr_ratio"], 0.1)
        self.assertEqual(hyperparams["gradient_clip"], 1.0)

    def test_params_under_one_million(self):
        for vocab in (110, 256, 512, 1024):
            cfg = model_from_preset("story_sub1m", vocab_size=vocab)
            n = estimate_vram_footprint(cfg)["total_params"]
            self.assertLess(
                n,
                1_000_000,
                f"story_sub1m exceeded 1M params at vocab_size={vocab}: {n}",
            )
        default_n = estimate_vram_footprint(
            model_from_preset("story_sub1m", vocab_size=256)
        )["total_params"]
        self.assertGreater(default_n, 500_000)

    def test_architecture_preset_registered(self):
        self.assertIn("story_sub1m", PRESETS)
        self.assertEqual(PRESETS["story_sub1m"]["embedding_dim"], 128)

    def test_ready_config_matches_preset(self):
        config_path = _ROOT / "setup" / "story_sub1m_config.json"
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        model, hyperparams, dataset_name = apply_scale_preset("story_sub1m", vocab_size=256)
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
