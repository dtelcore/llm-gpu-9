"""CPU tests for the unguided generate prober and post-stop next-step decide."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import unguided_prober
from training.cabinet_index import CabinetIndex
from training.unguided.decide import NextStepContext, decide_next_step
from training.unguided.prober import (
    classify_probe_row,
    family,
    gold_omits_slot,
    mix_smells,
    next_step_context,
    ood_copies_mix,
    render_markdown,
    select_prompts,
    summarize_rows,
)


def _index() -> CabinetIndex:
    idx = CabinetIndex()
    idx.add("What is the capital of France?", "The capital of France is Paris.")
    idx.add("What is France's capital?", "The capital of France is Paris.")
    idx.add("Where is Paris?", "Paris is the capital of France.")
    idx.add("What country is Paris in?", "Paris is the capital of France.")
    idx.add("Which organ system does the ureter belong to?", "The ureter belongs to the urinary system.")
    idx.add("Which organ system does the kidney belong to?", "The kidney belongs to the urinary system.")
    idx.add("What pathogen causes vomiting?", "Rotavirus is the pathogen that causes vomiting.")
    idx.add("Is 17 a prime number?", "Yes, 17 is a prime number.")
    idx.add("Is 13 a prime number?", "Yes, 13 is a prime number.")
    idx.add("Is 42 a prime number?", "No, 42 is a composite number.")
    idx.add("Which organ system does the spleen belong to?", "The spleen belongs to the immunity.")
    return idx


class FamilyAndSelectTests(unittest.TestCase):
    def test_family_templates(self):
        self.assertEqual(family("What is the capital of France?"), "capital_of")
        self.assertEqual(family("Which organ system does the ureter belong to?"), "organ_system")
        self.assertEqual(family("Is 17 a prime number?"), "prime")

    def test_select_includes_must_and_caps(self):
        idx = _index()
        picked = select_prompts(idx, 4, seed=1)
        self.assertEqual(len(picked), 4)
        self.assertEqual(picked[0].user, "What is the capital of France?")

    def test_dirty_and_shared_gold(self):
        idx = _index()
        self.assertTrue(gold_omits_slot("Is 17 a prime number?", "Yes, 13 is a prime number."))
        self.assertFalse(gold_omits_slot("Is 17 a prime number?", "Yes, 17 is a prime number."))
        smells = mix_smells(idx)
        self.assertGreaterEqual(smells["shared_n"], 2)
        self.assertGreaterEqual(smells["dirty_n"], 0)

    def test_neighbour_swap_class(self):
        idx = _index()
        fact = idx.lookup("Which organ system does the ureter belong to?")
        cls, src = classify_probe_row(idx, fact, "The kidney belongs to the urinary system.")
        self.assertEqual(cls, "TEMPLATE_NEIGHBOUR_SWAP")
        self.assertIn("kidney", (src or "").casefold())

    def test_ood_detects_mix_shape(self):
        idx = _index()
        self.assertTrue(ood_copies_mix(idx, "The capital of France is Paris."))
        self.assertTrue(ood_copies_mix(idx, "Atlantis is the capital of Nowhere."))
        self.assertFalse(ood_copies_mix(idx, "Rain taps the roof and the street shines."))


class NextStepTests(unittest.TestCase):
    def test_run2_like_snapshot_is_memorizing_change_mix(self):
        result = decide_next_step(
            NextStepContext(
                step=700,
                max_steps=1500,
                val_loss=0.54,
                best_val_loss=0.54,
                cabinet_exact_match=0.0,
                generate_exact_rate=0.54,
                generate_swap_rate=0.30,
                generate_n=46,
                ood_mix_copies=3,
                ood_n=4,
                dirty_gold=2,
                shared_assistants=8,
                collapsing_families=("pathogen", "prime", "year_released"),
                long_unique_fail=1,
                short_template_exact=0.75,
            )
        )
        self.assertFalse(result.understands)
        self.assertEqual(result.mode, "mixed_recitation")
        self.assertEqual(result.primary, "change_mix")
        needed = {i.action: i.needed for i in result.items}
        self.assertTrue(needed["change_data"])
        self.assertTrue(needed["change_mix"])
        self.assertTrue(needed["more_steps"])
        self.assertFalse(needed["new_config"])
        self.assertTrue(needed["new_policy"])

    def test_high_exact_clean_ood_is_generalizing(self):
        result = decide_next_step(
            NextStepContext(
                step=1500,
                max_steps=1500,
                val_loss=0.2,
                best_val_loss=0.2,
                cabinet_exact_match=0.9,
                generate_exact_rate=0.92,
                generate_swap_rate=0.02,
                generate_n=50,
                ood_mix_copies=0,
                ood_n=4,
                dirty_gold=0,
                shared_assistants=0,
            )
        )
        self.assertTrue(result.understands)
        self.assertEqual(result.mode, "generalizing")
        self.assertEqual(result.primary, "hold")
        self.assertFalse(any(i.needed for i in result.items if i.action != "new_policy"))

    def test_swaps_dominate_low_exact_mix_first(self):
        result = decide_next_step(
            NextStepContext(
                step=200,
                max_steps=1500,
                val_loss=1.2,
                best_val_loss=1.2,
                cabinet_exact_match=0.0,
                generate_exact_rate=0.10,
                generate_swap_rate=0.40,
                generate_n=50,
                collapsing_families=("prime",),
            )
        )
        self.assertEqual(result.mode, "not_reciting")
        self.assertEqual(result.primary, "change_mix")
        needed = {i.action: i.needed for i in result.items}
        self.assertTrue(needed["change_mix"])
        self.assertFalse(needed["more_steps"])

    def test_markdown_has_next_steps_table(self):
        rows = [
            {
                "family": "organ_system",
                "canonical": "Which organ system does the ureter belong to?",
                "expected": "The ureter belongs to the urinary system.",
                "generated": "The kidney belongs to the urinary system.",
                "match": False,
                "classification": "TEMPLATE_NEIGHBOUR_SWAP",
                "swapped_from": "Which organ system does the kidney belong to?",
            }
        ]
        scores = summarize_rows(rows)
        report = {
            "checkpoint": "output/checkpoints/demo",
            "facts": "data/chat_facts_v7.jsonl",
            "step": 700,
            "n_unique": 1,
            "n_cabinet": scores["n_cabinet"],
            "n_match": scores["n_match"],
            "n_swap": scores["n_swap"],
            "exact_rate": scores["exact_rate"],
            "swap_rate": scores["swap_rate"],
            "by_family": scores["by_family"],
            "ood_n": 0,
            "ood_mix_copies": 0,
            "mix_smells": {"dirty_n": 0, "shared_n": 1, "dirty_gold": [], "shared_assistants": []},
            "cabinet": rows,
            "ood": [],
        }
        verdict = decide_next_step(next_step_context(step=700, max_steps=1500, policy={}, last_eval={}, report=report))
        md = render_markdown(report, verdict)
        self.assertIn("## Next steps", md)
        self.assertIn("Change data mix", md)
        self.assertIn("understands: **no**", md)


class ProberCliTests(unittest.TestCase):
    def test_dry_run_no_metal(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = unguided_prober.main([
                "--facts", str(_ROOT / "data" / "chat_facts_v7.jsonl"),
                "--n", "8",
                "--dry-run",
            ])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("UNGUIDED PROBER DRY-RUN", out)
        self.assertIn("No Metal init", out)
        self.assertIn("selected:", out)

    def test_write_reports(self):
        from training.unguided.prober import write_probe_reports
        from training.unguided.decide import NextStepResult, NextStepItem

        verdict = NextStepResult(
            mode="memorizing",
            understands=False,
            headline="memorizing",
            primary="change_mix",
            items=[NextStepItem("change_mix", True, "swaps")],
            reasons=["demo"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_probe_reports(
                Path(tmp),
                {
                    "checkpoint": "x",
                    "facts": "y",
                    "step": 1,
                    "n_unique": 0,
                    "n_cabinet": 0,
                    "n_match": 0,
                    "n_swap": 0,
                    "exact_rate": 0.0,
                    "swap_rate": 0.0,
                    "by_family": {},
                    "ood_n": 0,
                    "ood_mix_copies": 0,
                    "mix_smells": {},
                    "cabinet": [],
                    "ood": [],
                },
                verdict,
            )
            self.assertTrue(path.is_file())
            self.assertTrue((Path(tmp) / "NEXT_STEP.json").is_file())


if __name__ == "__main__":
    unittest.main()
