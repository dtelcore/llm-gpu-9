"""Phase 2 mix uniqueness and resume corpus swap (no Metal, no BPE rebuild)."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cli_common
import unguided_prober
from tools.build_english_phase2_mix import (
    build_mix,
    generate_frames,
    max_identical_count,
    unique_pairs_from_jsonl,
)
from training.resume_inject import (
    ResumeInjectError,
    apply_resume_dataset_path,
    verify_resume_inject,
)
from training.unguided.decide import NextStepContext, decide_next_step
from training.unguided.prober import (
    PHASE1_OOD_PROMPTS,
    PHASE2_INJECT_PROMPTS,
    render_markdown,
    run_inject_probe,
    write_inject_probe_reports,
)


def _jsonl_line(user: str, assistant: str) -> str:
    return json.dumps({"query": {"user": user}, "response": {"assistant": assistant}})


class MixBuilderTests(unittest.TestCase):
    def test_unique_pairs_collapse_repeats(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v9.jsonl"
            path.write_text(
                "\n".join([
                    _jsonl_line("What is the capital of France?", "The capital of France is Paris."),
                    _jsonl_line("What is the capital of France?", "The capital of France is Paris."),
                    _jsonl_line("Where is Paris?", "Paris is a city in France."),
                ]) + "\n",
                encoding="utf-8",
            )
            pairs = unique_pairs_from_jsonl(path)
            self.assertEqual(len(pairs), 2)
            self.assertEqual(pairs["What is the capital of France?"], "The capital of France is Paris.")

    def test_four_distinct_frames_no_identical_dup(self):
        frames = generate_frames(
            "What is the capital of France?",
            "The capital of France is Paris.",
        )
        self.assertGreaterEqual(len(frames), 3)
        self.assertEqual(max_identical_count(frames), 1)
        joined = "\n".join(frames)
        self.assertIn("User: What is the capital of France? Assistant: The capital of France is Paris.", joined)
        self.assertIn("Regarding What is the capital of France", joined)
        self.assertIn("Question:", joined)

    def test_interleave_keeps_all_prose(self):
        prose = [f"prose {i}" for i in range(12)]
        pairs = {
            "What is the capital of France?": "The capital of France is Paris.",
            "Where is Paris?": "Paris is a city in France.",
        }
        mixed, frames = build_mix(pairs=pairs, prose_lines=prose, seed=1)
        self.assertEqual(len(frames), 8)
        self.assertEqual(max_identical_count(frames), 1)
        self.assertEqual(len(mixed), len(prose) + len(frames))
        self.assertEqual(sum(1 for line in mixed if line.startswith("prose ")), 12)
        self.assertGreater(sum(1 for line in mixed if line.startswith("User:")), 0)

    def test_dense_target_and_fraction(self):
        from tools.build_english_phase2_mix import select_target_pairs

        pairs = {
            "What is the capital of France?": "The capital of France is Paris.",
            "Where is Paris?": "Paris is a city in France.",
            "Which organ system does the kidney belong to?": "The kidney belongs to the urinary system.",
            "Is 17 a prime number?": "Yes, 17 is a prime number.",
            "Who invented Z4?": "Konrad Zuse is credited with inventing Z4.",
            "What pathogen causes COVID-19?": "SARS-CoV-2 is the pathogen that causes COVID-19.",
        }
        picked = select_target_pairs(pairs, max_facts=60, max_per_family=15)
        self.assertIn("What is the capital of France?", picked)
        self.assertIn("Which organ system does the kidney belong to?", picked)
        self.assertNotIn("Who invented Z4?", picked)
        self.assertNotIn("What pathogen causes COVID-19?", picked)
        frames = generate_frames(
            "What is the capital of France?",
            "The capital of France is Paris.",
            dense=True,
        )
        self.assertGreaterEqual(len(frames), 6)
        self.assertEqual(max_identical_count(frames), 1)
        prose = [f"wiki sentence {i} about rivers and trade." for i in range(400)]
        mixed, fact_frames = build_mix(
            pairs=picked,
            prose_lines=prose,
            seed=2,
            dense=True,
            fact_fraction=0.15,
        )
        frac = len(fact_frames) / len(mixed)
        self.assertGreater(frac, 0.10)
        self.assertLess(frac, 0.22)
        self.assertLess(len(mixed), 400)

    def test_refuse_duplicate_fact_frames(self):
        with mock.patch(
            "tools.build_english_phase2_mix.all_fact_frames",
            return_value=["dup frame", "dup frame"],
        ):
            with self.assertRaises(ValueError):
                build_mix(pairs={"a": "b"}, prose_lines=["prose 0"], seed=0)


class ResumeInjectTests(unittest.TestCase):
    def test_dataset_path_without_resume_errors(self):
        with self.assertRaises(ResumeInjectError):
            verify_resume_inject(
                resume=False,
                dataset_path="data/english_phase2_mix.txt",
                checkpoint_dir="/tmp/missing",
            )

    def test_missing_vocab_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                verify_resume_inject(
                    resume=True,
                    dataset_path="data/english_phase2_mix.txt",
                    checkpoint_dir=tmp,
                )

    def test_apply_keeps_val_swaps_train_no_tokenizer_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            mix = Path(tmp) / "mix.txt"
            mix.write_text(
                "new train line one\nheld out val sentence\nnew train line two\n",
                encoding="utf-8",
            )
            ckpt = Path(tmp) / "ckpt"
            ckpt.mkdir()
            (ckpt / "vocab.json").write_text("{}", encoding="utf-8")
            (ckpt / "val_corpus.json").write_text(
                json.dumps(["held out val sentence"]),
                encoding="utf-8",
            )
            tokenizer = SimpleNamespace(vocab_size=4112)
            config = {
                "model": {"embedding_dim": 256, "num_layers": 6, "max_len": 256},
                "dataset": {
                    "path": "data/train.txt",
                    "combine": True,
                    "corpus": ["old phase1 train"],
                    "val_corpus": ["held out val sentence"],
                },
            }
            with mock.patch("tokenizer.factory.build_tokenizer") as forbidden:
                train, val = apply_resume_dataset_path(
                    config, ckpt, str(mix), tokenizer=tokenizer,
                )
                forbidden.assert_not_called()
            self.assertEqual(val, ["held out val sentence"])
            self.assertIn("new train line one", train)
            self.assertIn("new train line two", train)
            self.assertNotIn("held out val sentence", train)
            self.assertNotIn("old phase1 train", train)
            self.assertEqual(config["dataset"]["path"], str(mix))
            self.assertFalse(config["dataset"]["combine"])
            self.assertIs(tokenizer.vocab_size, 4112)

    def test_cli_flags_exist_without_redeclaring_resume(self):
        parser = argparse.ArgumentParser()
        cli_common.add_training_length_args(parser)
        args = parser.parse_args([
            "--dataset-path", "data/english_phase2_mix.txt",
            "--reset-lr-schedule",
            "--learning-rate", "5e-5",
            "--steps", "1500",
        ])
        self.assertEqual(args.dataset_path, "data/english_phase2_mix.txt")
        self.assertTrue(args.reset_lr_schedule)
        names = {action.dest for action in parser._actions}
        self.assertNotIn("resume", names)
        self.assertNotIn("checkpoint", names)


class InjectProbeTests(unittest.TestCase):
    def test_inject_probe_markdown(self):
        class Tok:
            def encode(self, text):
                return [1, 2]

            def decode(self, ids):
                return " Paris remains the political center."

        class M:
            def generate(self, *args, **kwargs):
                return [1, 2, 3, 4]

        report = run_inject_probe(model=M(), tokenizer=Tok(), step=10000, checkpoint="English-Phase1")
        self.assertEqual(report["probe_mode"], "inject")
        self.assertEqual(report["ood_n"], len(PHASE1_OOD_PROMPTS))
        self.assertEqual(len(report["inject"]), len(PHASE2_INJECT_PROMPTS))
        verdict = decide_next_step(
            NextStepContext(
                step=10000,
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
        self.assertIn("Held-out inject frames", md)
        self.assertIn("User: What city is the capital of France? Assistant:", md)
        self.assertIn("Once upon a time in a valley", md)
        self.assertIn("not v9 96% exact", md)
        self.assertNotIn("| generate exact |", md)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_inject_probe_reports(Path(tmp), report, verdict)
            self.assertEqual(path.name, "inject_probe.md")
            self.assertTrue(path.is_file())
            self.assertFalse((Path(tmp) / "generate_probe.md").exists())

    def test_inject_dry_run(self):
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = unguided_prober.main([
                "--name", "English-Phase1",
                "--probe-mode", "inject",
                "--dry-run",
            ])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("probe_mode:    inject", out)
        self.assertIn("User: What city is the capital of France? Assistant:", out)
        self.assertIn("No Metal init", out)


class Phase2RecipeTests(unittest.TestCase):
    def test_recipe_matches_phase1_arch(self):
        phase1 = json.loads((_ROOT / "setup" / "english_phase1_config.json").read_text(encoding="utf-8"))
        phase2 = json.loads((_ROOT / "setup" / "english_phase2_config.json").read_text(encoding="utf-8"))
        for key in ("embedding_dim", "num_layers", "num_heads", "max_len"):
            self.assertEqual(phase1["model"][key], phase2["model"][key])
        self.assertFalse(phase2["dataset"]["combine"])
        self.assertEqual(phase2["dataset"]["path"], "data/english_phase2_mix.txt")
        self.assertEqual(phase2["hyperparameters"]["learning_rate"], 5e-5)
        phase2b = json.loads((_ROOT / "setup" / "english_phase2b_config.json").read_text(encoding="utf-8"))
        for key in ("embedding_dim", "num_layers", "num_heads", "max_len"):
            self.assertEqual(phase1["model"][key], phase2b["model"][key])
        self.assertEqual(phase2b["dataset"]["path"], "data/english_phase2b_mix.txt")
        self.assertFalse(phase2b["dataset"]["combine"])
        self.assertEqual(phase2b["hyperparameters"]["learning_rate"], 6e-5)


if __name__ == "__main__":
    unittest.main()
