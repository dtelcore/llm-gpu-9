"""v10 mix: v9 uniques, gold pins, native User:/Assistant:, 40–50% facts, no 300×."""

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

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import unguided_trainer
from setup.dataset_setup import DatasetLoader, _native_line_from_jsonl
from tools.build_chat_facts_v10 import (
    DEFAULT_CORE_REPEAT,
    DEFAULT_TAIL_REPEAT,
    GOLD_FIXES,
    RECITATION_REPEAT_REFUSE,
    apply_gold_fixes,
    assert_gold_pins,
    build_records,
    canonical_line,
    native_fact_lines,
    write_jsonl,
)
from tools.build_english_phase2_mix import max_identical_count, unique_pairs_from_jsonl
from training.cabinet_index import load_cabinet, normalize_question
from training.unguided.decide import DecideContext, Decision, NextStepContext, decide, decide_next_step
from training.unguided.eval_suite import run_eval_suite
from training.unguided.prober import france_neighbour_bleed, write_session_stop_reports


def _jsonl_line(user: str, assistant: str) -> str:
    return json.dumps({"query": {"user": user}, "response": {"assistant": assistant}})


def _fixture_pairs() -> dict[str, str]:
    return {
        "What is the capital of France?": "The capital of France is Paris.",
        "What is France's capital?": "The capital of France is Paris.",
        "Where is Paris?": "Paris is a city in France.",
        "What country is Paris in?": "Paris is in France.",
        "What is the capital of Belgium?": "The capital of Belgium is Brussels.",
        "Where is Brussels?": "Brussels is a city in Belgium.",
        "Who is L'Hopital's rule named after?": "Cramer's rule is named after Gabriel Cramer.",
        "Who is Cramer's rule named after?": "L'Hopital's rule is named after Guillaume de l'Hopital.",
        "Is 1 a prime number?": "No, 1 is a prime number.",
        "Which organ system does the kidney belong to?": "The kidney belongs to the urinary system.",
    }


class GoldPinTests(unittest.TestCase):
    def test_fixes_named_after_and_prime_not(self):
        pairs = apply_gold_fixes(_fixture_pairs())
        assert_gold_pins(pairs)
        hopital = normalize_question("Who is L'Hopital's rule named after?")
        cramer = normalize_question("Who is Cramer's rule named after?")
        by_norm = {normalize_question(u): a for u, a in pairs.items()}
        self.assertEqual(by_norm[hopital], GOLD_FIXES["Who is L'Hopital's rule named after?"][1])
        self.assertEqual(by_norm[cramer], GOLD_FIXES["Who is Cramer's rule named after?"][1])
        self.assertNotEqual(by_norm[hopital], by_norm[cramer])
        self.assertIn("not", by_norm[normalize_question("Is 1 a prime number?")])
        self.assertEqual(
            pairs["Where is Paris?"],
            "Paris is a city in France.",
        )
        self.assertEqual(
            pairs["What country is Paris in?"],
            "Paris is in France.",
        )


class NativeMixTests(unittest.TestCase):
    def test_native_only_core_and_tail_repeats(self):
        pairs = apply_gold_fixes(_fixture_pairs())
        lines = native_fact_lines(pairs)
        france = canonical_line(
            "What is the capital of France?",
            "The capital of France is Paris.",
        )
        hopital = canonical_line(
            "Who is L'Hopital's rule named after?",
            GOLD_FIXES["Who is L'Hopital's rule named after?"][1],
        )
        self.assertEqual(lines.count(france), DEFAULT_CORE_REPEAT)
        self.assertEqual(lines.count(hopital), DEFAULT_TAIL_REPEAT)
        self.assertEqual(max_identical_count(lines), DEFAULT_CORE_REPEAT)
        joined = "\n".join(lines)
        self.assertNotIn("It is known that", joined)
        self.assertNotIn("A short note on", joined)
        self.assertTrue(all(line.startswith("User:") for line in lines))

    def test_mix_fraction_and_cabinet_skips_prose(self):
        pairs = apply_gold_fixes(_fixture_pairs())
        prose = [f"wiki sentence {i} about rivers valleys and trade routes." for i in range(4000)]
        records, facts, mixed = build_records(
            pairs=pairs,
            prose_lines=prose,
            fact_fraction=0.45,
            seed=2,
        )
        self.assertLess(max_identical_count(facts), RECITATION_REPEAT_REFUSE)
        self.assertEqual(max_identical_count(facts), DEFAULT_CORE_REPEAT)
        frac = len(facts) / len(mixed)
        self.assertGreaterEqual(frac, 0.40)
        self.assertLessEqual(frac, 0.50)
        self.assertEqual(len(records), len(mixed))
        n_gold = sum(1 for rec in records if rec.get("query"))
        n_frame = sum(1 for rec in records if rec.get("kind") == "frame")
        self.assertEqual(n_gold, len(facts))
        self.assertEqual(n_frame, 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chat_facts_v10.jsonl"
            write_jsonl(path, records)
            index = load_cabinet(path)
            trained = list(index.unique_facts())
            self.assertEqual(len(trained), len(pairs))
            hopital = index.lookup("Who is L'Hopital's rule named after?")
            self.assertIsNotNone(hopital)
            self.assertIn("Guillaume de l'Hopital", hopital.assistant)
            self.assertNotIn("Cramer", hopital.assistant)
            prime = index.lookup("Is 1 a prime number?")
            self.assertIsNotNone(prime)
            self.assertIn("not", prime.assistant)
            loader = DatasetLoader(data_dir=tmp, auto_discover=False)
            corpus = loader.load_from_file(str(path))
            self.assertEqual(len(corpus), len(mixed))
            self.assertTrue(any(line.startswith("wiki sentence") for line in corpus))
            self.assertTrue(any(line.startswith("User:") for line in corpus))
            chat = [line for line in corpus if line.startswith("User:")]
            self.assertTrue(chat)
            self.assertTrue(all(line.endswith(" User:") for line in chat))
            france_gold = index.lookup("What is the capital of France?")
            self.assertIsNotNone(france_gold)
            self.assertFalse(france_gold.assistant.endswith("User:"))
            self.assertFalse(any("It is known that" in line for line in chat))

    def test_refuse_fraction_outside_band(self):
        with self.assertRaises(ValueError):
            build_records(
                pairs={"What is the capital of France?": "The capital of France is Paris."},
                prose_lines=["wiki 0"],
                fact_fraction=0.15,
            )

    def test_refuse_300x(self):
        with self.assertRaises(ValueError):
            native_fact_lines(
                {"What is the capital of France?": "The capital of France is Paris."},
                core_repeat=RECITATION_REPEAT_REFUSE,
                tail_repeat=10,
            )


class JsonlLoaderTests(unittest.TestCase):
    def test_prose_and_frame_records_become_train_lines(self):
        self.assertEqual(
            _native_line_from_jsonl(json.dumps({"kind": "prose", "text": "Once upon a time in a valley"})),
            "Once upon a time in a valley",
        )
        self.assertEqual(
            _native_line_from_jsonl(json.dumps({
                "query": {"user": "Is 1 a prime number?"},
                "response": {"assistant": "No, 1 is not a prime number."},
            })),
            "User: Is 1 a prime number? Assistant: No, 1 is not a prime number.",
        )


class ProbePolicyTests(unittest.TestCase):
    def test_inject_eval_skips_cabinet(self):
        session = SimpleNamespace(
            model=object(),
            val_dataset=None,
            args=SimpleNamespace(seed=42),
            tokenizer=object(),
            dataset_path="data/chat_facts_v10.jsonl",
        )
        with mock.patch("training.eval.evaluate_val_loss", return_value=(2.5, 12.1)):
            rec = run_eval_suite(session, {"probe_mode": "inject"})
        self.assertEqual(rec.val_loss, 2.5)
        self.assertIsNone(rec.cabinet_exact_match)

    def test_inject_never_remixes(self):
        result = decide(
            DecideContext(
                step=500,
                max_steps=8000,
                wall_s=10.0,
                max_wall_s=28800.0,
                val_loss=1.2,
                best_val_loss=1.3,
                recent_val_losses=[1.4, 1.3],
                nan_detected=False,
                cabinet_exact_match=0.0,
                policy={
                    "probe_mode": "inject",
                    "loss_spike_ratio": 2.0,
                    "early_stop_patience": 40,
                    "remix_if": {"cabinet_exact_match_below": 0.15, "after_steps": 1},
                },
            )
        )
        self.assertEqual(result.action, Decision.PROMOTE)

    def test_next_step_not_v9_exact(self):
        rec = decide_next_step(
            NextStepContext(
                step=8000,
                max_steps=8000,
                val_loss=2.0,
                best_val_loss=2.0,
                cabinet_exact_match=None,
                generate_exact_rate=0.96,
                generate_swap_rate=0.02,
                ood_mix_copies=0,
                ood_n=5,
                policy={"probe_mode": "inject"},
            )
        )
        self.assertEqual(rec.mode, "inject_integration")
        self.assertFalse(rec.understands)
        self.assertIn("Belgium", rec.headline)
        self.assertIn("96%", rec.headline)

    def test_belgium_on_france_is_bleed(self):
        self.assertTrue(
            france_neighbour_bleed(
                "User: What city is the capital of France? Assistant:",
                "known that What is Belgium is",
            )
        )
        self.assertFalse(
            france_neighbour_bleed(
                "User: What city is the capital of France? Assistant:",
                "Paris, the capital of France.",
            )
        )

    def test_stop_writes_inject_and_generate(self):
        from training.unguided.decide import NextStepResult

        verdict = NextStepResult(
            mode="inject_integration",
            understands=False,
            headline="v10",
            primary="hold",
            items=[],
            reasons=[],
        )
        inject = {
            "probe_mode": "inject",
            "facts": "data/chat_facts_v10.jsonl",
            "ood_n": 1,
            "ood_mix_copies": 0,
            "ood": [{"typed": "Once upon a time in a valley", "generated": "the river ran.", "copies_mix": False}],
            "inject": [],
            "n_neighbour_bleed": 0,
            "cabinet_report": {
                "facts": "data/chat_facts_v10.jsonl",
                "n_unique": 1,
                "n_cabinet": 1,
                "n_match": 0,
                "n_swap": 0,
                "exact_rate": 0.0,
                "swap_rate": 0.0,
                "cabinet": [{
                    "kind": "cabinet",
                    "family": "capital_of",
                    "canonical": "What is the capital of France?",
                    "expected": "The capital of France is Paris.",
                    "generated": "Paris is in France.",
                    "match": False,
                    "classification": "TARGET_MISMATCH",
                }],
                "ood": [],
                "ood_n": 0,
                "ood_mix_copies": 0,
                "by_family": {},
                "mix_smells": {"dirty_n": 0, "shared_n": 0, "dirty_gold": [], "shared_assistants": []},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            md = write_session_stop_reports(out, inject, verdict)
            self.assertEqual(md.name, "inject_probe.md")
            self.assertTrue((out / "generate_probe.md").is_file())
            gen = (out / "generate_probe.md").read_text(encoding="utf-8")
            self.assertIn("What is the capital of France?", gen)


class RecipeDryRunTests(unittest.TestCase):
    def test_v10_recipe_arch_and_dry_run(self):
        recipe = json.loads((_ROOT / "setup" / "chat_facts_v10_config.json").read_text(encoding="utf-8"))
        self.assertFalse(recipe["dataset"]["combine"])
        self.assertEqual(recipe["dataset"]["path"], "data/chat_facts_v10.jsonl")
        self.assertEqual(recipe["hyperparameters"]["learning_rate"], 6e-5)
        policy = json.loads((_ROOT / "setup" / "unguided_v10_policy.json").read_text(encoding="utf-8"))
        self.assertEqual(policy["probe_mode"], "inject")
        self.assertEqual(policy["early_stop_patience"], 200)
        self.assertEqual(policy["max_steps"], 10000)
        self.assertGreaterEqual(policy["remix_if"]["after_steps"], policy["max_steps"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = unguided_trainer.main([
                "--config", str(_ROOT / "setup" / "chat_facts_v10_config.json"),
                "--policy", str(_ROOT / "setup" / "unguided_v10_policy.json"),
                "--name", "chat_facts_v10",
                "--dry-run",
            ])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("UNGUIDED DRY-RUN", out)
        self.assertIn("probe_mode:    inject", out)
        self.assertIn("data/chat_facts_v10.jsonl", out)
        self.assertIn("No Metal init", out)

    def test_live_v9_uniques_if_present(self):
        v9 = _ROOT / "data" / "chat_facts_v9.jsonl"
        if not v9.is_file():
            self.skipTest("data/chat_facts_v9.jsonl missing")
        raw = unique_pairs_from_jsonl(v9)
        self.assertEqual(len(raw), 570)
        pairs = apply_gold_fixes(raw)
        assert_gold_pins(pairs)
        self.assertGreaterEqual(len(pairs), 570)
        self.assertIn("Is 17 a prime number?", pairs)
        self.assertIn("Is 42 a prime number?", pairs)


if __name__ == "__main__":
    unittest.main()
