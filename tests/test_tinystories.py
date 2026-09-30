"""TinyStories 0.1.0 path: config/policy parse, probe prompts, prepare helpers."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import unguided_prober
import unguided_trainer
from tools.prepare_tinystories import collapse_story_line, write_manifest, write_text_shard
from training.dataset import WindowedDataset
from training.tinystories_tokens import ConcatenatedMemmap, dataset_uses_prebuilt_tokens
from training.unguided.decide import NextStepContext, decide, decide_next_step, DecideContext, Decision
from training.unguided.eval_suite import probe_mode, run_eval_suite
from training.unguided.prober import TINYSTORIES_MAX_NEW_TOKENS, TINYSTORIES_PROMPTS


class ConfigPolicyTests(unittest.TestCase):
    def test_recipe_has_no_text_path(self):
        recipe = json.loads((_ROOT / "legacy" / "setup" / "english_tinystories_c256_l6_config.json").read_text(encoding="utf-8"))
        self.assertNotIn("path", recipe["dataset"])
        self.assertEqual(recipe["dataset"]["vocab_path"], "data/tinystories/vocab.json")
        self.assertEqual(recipe["dataset"]["token_dir"], "data/tinystories")
        self.assertFalse(recipe["dataset"]["combine"])
        self.assertEqual(recipe["model"]["embedding_dim"], 256)
        self.assertEqual(recipe["model"]["num_layers"], 6)
        self.assertEqual(recipe["model"]["num_heads"], 8)
        self.assertEqual(recipe["model"]["max_len"], 256)
        self.assertTrue(recipe["model"]["residual_scale"])
        self.assertEqual(recipe["hyperparameters"]["batch_size"], 4)
        self.assertEqual(recipe["hyperparameters"]["gradient_accumulation_steps"], 4)
        self.assertTrue(dataset_uses_prebuilt_tokens(recipe["dataset"]))
        self.assertFalse(dataset_uses_prebuilt_tokens({"name": "chat_facts_v7", "path": "data/chat_facts_v7.jsonl"}))

    def test_archived_c512_file_is_the_wide_stream_smoke(self):
        recipe = json.loads((_ROOT / "legacy" / "setup" / "english_tinystories_c512_l6_config.json").read_text(encoding="utf-8"))
        narrow = json.loads((_ROOT / "legacy" / "setup" / "english_tinystories_c256_l6_config.json").read_text(encoding="utf-8"))
        self.assertEqual(recipe["dataset"]["vocab_path"], "data/tinystories_packed/vocab.json")
        self.assertEqual(recipe["dataset"]["token_dir"], "data/tinystories_packed")
        self.assertNotEqual(recipe["dataset"]["token_dir"], narrow["dataset"]["token_dir"])
        self.assertEqual(recipe["model"]["embedding_dim"], 1024)
        self.assertEqual(recipe["model"]["layer_strategy"], "stream")
        self.assertEqual(recipe["model"]["embedding_dim"] // recipe["model"]["num_heads"], 128)
        self.assertEqual(recipe["hyperparameters"]["batch_size"], 4)
        self.assertEqual(recipe["hyperparameters"]["gradient_accumulation_steps"], 16)
        self.assertEqual(recipe["hyperparameters"]["warmup_steps"], 600)

    def test_policy_budget(self):
        policy = json.loads((_ROOT / "legacy" / "setup" / "unguided_tinystories_policy.json").read_text(encoding="utf-8"))
        self.assertEqual(policy["probe_mode"], "tinystories")
        self.assertEqual(policy["max_steps"], 50000)
        self.assertEqual(policy["probe_every"], 2000)
        self.assertEqual(policy["eval_every"], 500)
        self.assertEqual(policy["max_wall_s"], 259200)
        self.assertGreaterEqual(policy["early_stop_patience"], 50000)
        self.assertEqual(probe_mode(policy), "tinystories")


class V011RecipeTests(unittest.TestCase):
    def test_c1024_recipe_uses_packed_stories(self):
        recipe = json.loads((_ROOT / "setup" / "english_tinystories_c1024_l6_config.json").read_text(encoding="utf-8"))
        policy = json.loads((_ROOT / "setup" / "unguided_tinystories_policy.json").read_text(encoding="utf-8"))
        self.assertEqual(recipe["model"]["embedding_dim"], 1024)
        self.assertEqual(recipe["model"]["num_layers"], 6)
        self.assertEqual(recipe["model"]["num_heads"], 8)
        self.assertEqual(recipe["model"]["max_len"], 256)
        self.assertEqual(recipe["model"]["layer_strategy"], "stream")
        self.assertTrue(recipe["model"]["residual_scale"])
        self.assertEqual(recipe["dataset"]["vocab_path"], "data/tinystories_packed/vocab.json")
        self.assertEqual(recipe["dataset"]["token_dir"], "data/tinystories_packed")
        self.assertFalse(recipe["dataset"]["combine"])
        self.assertEqual(recipe["hyperparameters"]["batch_size"], 4)
        self.assertEqual(recipe["hyperparameters"]["gradient_accumulation_steps"], 16)
        self.assertEqual(recipe["hyperparameters"]["learning_rate"], 0.0012)
        self.assertEqual(recipe["hyperparameters"]["warmup_steps"], 600)
        self.assertEqual(recipe["hyperparameters"]["min_lr_ratio"], 0.05)
        self.assertTrue(dataset_uses_prebuilt_tokens(recipe["dataset"]))
        self.assertEqual(policy["recipe"], "setup/english_tinystories_c1024_l6_config.json")
        self.assertEqual(policy["max_steps"], 4000)
        self.assertEqual(policy["probe_mode"], "tinystories")
        self.assertEqual((_ROOT / "VERSION").read_text(encoding="utf-8").strip(), "0.1.1")


class PromptAndDecideTests(unittest.TestCase):
    def test_fixed_prompt_list(self):
        self.assertEqual(len(TINYSTORIES_PROMPTS), 6)
        self.assertEqual(TINYSTORIES_MAX_NEW_TOKENS, 160)
        self.assertIn("Once upon a time there was a little girl named Lily who found a", TINYSTORIES_PROMPTS)
        self.assertIn("Tell me a short story about a brave mouse.", TINYSTORIES_PROMPTS)
        self.assertIn("Who are you?", TINYSTORIES_PROMPTS)
        self.assertIn("What is the capital of Atlantis?", TINYSTORIES_PROMPTS)
        self.assertIn("What is 17 + 4?", TINYSTORIES_PROMPTS)
        self.assertTrue(any("France" in p for p in TINYSTORIES_PROMPTS))

    def test_eval_skips_cabinet(self):
        session = SimpleNamespace(
            model=object(),
            val_dataset=None,
            args=SimpleNamespace(seed=42),
            tokenizer=object(),
            dataset_path="data/tinystories",
        )
        with mock.patch("training.eval.evaluate_val_loss", return_value=(2.4, 11.0)):
            rec = run_eval_suite(session, {"probe_mode": "tinystories"})
        self.assertEqual(rec.val_loss, 2.4)
        self.assertIsNone(rec.cabinet_exact_match)

    def test_never_remixes(self):
        result = decide(
            DecideContext(
                step=2000,
                max_steps=50000,
                wall_s=10.0,
                max_wall_s=259200.0,
                val_loss=2.1,
                best_val_loss=2.2,
                recent_val_losses=[2.3, 2.2],
                nan_detected=False,
                cabinet_exact_match=0.0,
                policy={
                    "probe_mode": "tinystories",
                    "loss_spike_ratio": 2.0,
                    "early_stop_patience": 100000,
                    "remix_if": {"cabinet_exact_match_below": 0.5, "after_steps": 1},
                },
            )
        )
        self.assertEqual(result.action, Decision.PROMOTE)

    def test_next_step_mode(self):
        rec = decide_next_step(
            NextStepContext(
                step=50000,
                max_steps=50000,
                val_loss=1.8,
                best_val_loss=1.8,
                cabinet_exact_match=None,
                generate_exact_rate=None,
                generate_swap_rate=0.0,
                ood_mix_copies=0,
                ood_n=6,
                policy={"probe_mode": "tinystories"},
            )
        )
        self.assertEqual(rec.mode, "tinystories_english")
        self.assertFalse(rec.understands)


class PrepareHelperTests(unittest.TestCase):
    def test_collapse_and_manifest(self):
        self.assertEqual(collapse_story_line("Once upon\na time\n\nLily  "), "Once upon a time Lily")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "text" / "train-00000.txt"
            n = write_text_shard(shard, ["Once upon\na time", "", "  Lily found a key.  "])
            self.assertEqual(n, 2)
            lines = shard.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines, ["Once upon a time", "Lily found a key."])
            vocab = root / "vocab.json"
            vocab.write_text('{"type":"bpe","vocab":["a"],"merges":[]}', encoding="utf-8")
            payload = write_manifest(
                root / "manifest.json",
                vocab_path=vocab,
                train_shards=["tokens/train-00000.npy"],
                valid_shards=["tokens/valid.npy"],
                train_tokens=12,
                valid_tokens=4,
                vocab_size=8,
                bpe_merges=6000,
            )
            self.assertEqual(payload["train_shards"], ["tokens/train-00000.npy"])
            self.assertEqual(payload["dtype"], "int32")
            self.assertEqual(len(payload["vocab_sha256"]), 64)


class MemmapWindowTests(unittest.TestCase):
    def test_concat_slice(self):
        a = np.arange(10, dtype=np.int32)
        b = np.arange(10, 18, dtype=np.int32)
        cat = ConcatenatedMemmap([a, b])
        self.assertEqual(len(cat), 18)
        np.testing.assert_array_equal(cat[8:14], np.arange(8, 14, dtype=np.int32))
        tok = SimpleNamespace(vocab_size=32)
        ds = WindowedDataset(
            [], tok, max_len=4, batch_size=2, window_stride=2, tokens=cat, copy_tokens=False,
        )
        batch = next(ds.iter_batches(shuffle=False))
        self.assertEqual(len(batch), 2)
        self.assertEqual(batch[0][0].dtype, np.int64)


class DryRunTests(unittest.TestCase):
    def test_trainer_dry_run(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = unguided_trainer.main(
                [
                    "--config", str(_ROOT / "legacy" / "setup" / "english_tinystories_c256_l6_config.json"),
                    "--policy", str(_ROOT / "legacy" / "setup" / "unguided_tinystories_policy.json"),
                    "--dry-run",
                ]
            )
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("probe_mode:    tinystories", out)
        self.assertIn("probe_every:   2000", out)
        self.assertIn("max_steps:     50000", out)

    def test_prober_dry_run(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = unguided_prober.main(["--probe-mode", "tinystories", "--dry-run"])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("Tell me a short story about a brave mouse.", out)
        self.assertIn("What is 17 + 4?", out)


if __name__ == "__main__":
    unittest.main()
