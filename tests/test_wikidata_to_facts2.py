"""Tests for Wikidata tech/health/maths facts (no live SPARQL)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.make_fact_mix import load_user_facts
from tools.wikidata_to_facts import apply_limit, drop_conflicts
from tools.wikidata_to_facts2 import (
    DEFAULT_OUTPUTS,
    DOMAIN_KINDS,
    HEALTH_SEEDS,
    QUERIES,
    THEOREM_STATEMENTS,
    collect_domain,
    collect_kinds,
    fold_label,
    format_numeric,
    format_year,
    looks_like_health_advice,
    main as wiki2_main,
    parse_args,
    verbalize,
)


def _bind(**kwargs):
    return {key: {"value": value} for key, value in kwargs.items()}


class WikidataFacts2Tests(unittest.TestCase):
    def test_verbalize_distinct_templates(self):
        c_lang = verbalize(
            _bind(langLabel="C", designerLabel="Dennis Ritchie"),
            "lang_designers",
        )
        self.assertEqual(
            c_lang,
            (
                "Who designed the C programming language?",
                "Dennis Ritchie designed the C programming language.",
            ),
        )
        lang = verbalize(
            _bind(langLabel="Python", designerLabel="Guido van Rossum"),
            "lang_designers",
        )
        self.assertEqual(
            lang,
            (
                "Who designed the Python programming language?",
                "Guido van Rossum designed the Python programming language.",
            ),
        )
        year = verbalize(_bind(langLabel="Python", year="1991"), "lang_years")
        self.assertEqual(
            year[0],
            "In which year was the Python programming language first released?",
        )
        self.assertIsNone(
            verbalize(_bind(langLabel="Cascading Style Sheets", year="1996"), "lang_years"),
        )
        proto = verbalize(_bind(protocolLabel="HTTP", year="1991.0"), "protocols")
        self.assertEqual(
            proto,
            ("In which year was HTTP introduced?", "HTTP was introduced in 1991."),
        )
        unit = verbalize(
            _bind(quantityLabel="force", unitLabel="newton"),
            "si_units",
        )
        self.assertEqual(
            unit,
            ("What is the SI unit of force?", "The SI unit of force is the newton."),
        )
        vit = verbalize(
            _bind(vitaminLabel="ascorbic acid", formula="C6H8O6"),
            "vitamin_formulas",
        )
        self.assertEqual(
            vit,
            (
                "What is the chemical formula of ascorbic acid?",
                "The chemical formula of ascorbic acid is C6H8O6.",
            ),
        )
        organ = verbalize(
            _bind(organLabel="heart", systemLabel="circulatory system"),
            "organ_systems",
        )
        self.assertIn("organ system", organ[0])
        path = verbalize(
            _bind(diseaseLabel="malaria", pathogenLabel="Plasmodium"),
            "disease_pathogens",
        )
        self.assertEqual(
            path,
            (
                "What pathogen causes malaria?",
                "Plasmodium is the pathogen that causes malaria.",
            ),
        )
        horm = verbalize(
            _bind(aaLabel="glycine", formula="C₂H₅NO₂"),
            "amino_acids",
        )
        self.assertEqual(
            horm,
            (
                "What is the chemical formula of the amino acid glycine?",
                "The chemical formula of the amino acid glycine is C2H5NO2.",
            ),
        )
        const = verbalize(
            _bind(constLabel="pi", value="3.141592653589793"),
            "math_constants",
        )
        self.assertEqual(
            const,
            (
                "What is the approximate numerical value of pi?",
                "The approximate value of pi is 3.14159.",
            ),
        )
        self.assertIsNone(
            verbalize(_bind(constLabel="imaginary unit", value="1"), "math_constants"),
        )
        thm = verbalize(
            _bind(theoremLabel="Pythagorean theorem", personLabel="Pythagoras"),
            "theorem_names",
        )
        self.assertEqual(
            thm,
            (
                "Who is Pythagorean theorem named after?",
                "Pythagorean theorem is named after Pythagoras.",
            ),
        )
        prime = verbalize(_bind(value="7"), "prime_numbers")
        self.assertEqual(prime, ("Is 7 a prime number?", "Yes, 7 is a prime number."))
        comp = verbalize(_bind(value="8"), "composite_numbers")
        self.assertEqual(comp, ("Is 8 a prime number?", "No, 8 is a composite number."))

        templates = {
            lang[0],
            year[0],
            proto[0],
            unit[0],
            vit[0],
            organ[0],
            path[0],
            horm[0],
            const[0],
            thm[0],
            prime[0],
        }
        self.assertGreaterEqual(len(templates), 10)

    def test_drops_qid_and_url_labels(self):
        self.assertIsNone(
            verbalize(_bind(langLabel="Q123", designerLabel="Ada"), "lang_designers"),
        )
        self.assertIsNone(
            verbalize(
                _bind(vitaminLabel="https://example.com", formula="C6H8O6"),
                "vitamin_formulas",
            ),
        )

    def test_health_advice_is_rejected(self):
        self.assertTrue(
            looks_like_health_advice(
                "How to treat malaria?",
                "Take 500 mg/day of a drug.",
            )
        )
        self.assertFalse(
            looks_like_health_advice(
                "What pathogen causes malaria?",
                "Plasmodium is the pathogen that causes malaria.",
            )
        )
        self.assertIsNone(
            verbalize(
                _bind(diseaseLabel="flu", pathogenLabel="Take 2 mg/day of rest"),
                "disease_pathogens",
            )
        )

    def test_drop_conflicts_entire_question(self):
        pairs = [
            ("Who designed the Java programming language?", "James Gosling designed Java."),
            ("Who designed the Java programming language?", "Sun Microsystems designed Java."),
            ("What is the SI unit of force?", "The SI unit of force is the newton."),
            ("What is the SI unit of force?", "The SI unit of force is the newton."),
        ]
        out = drop_conflicts(pairs)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], "What is the SI unit of force?")

    def test_collect_dedupes_and_drops_conflicts(self):
        rows = {
            "lang_designers": [
                _bind(langLabel="Python", designerLabel="Guido van Rossum"),
                _bind(langLabel="Python", designerLabel="Guido van Rossum"),
                _bind(langLabel="Java", designerLabel="James Gosling"),
                _bind(langLabel="Java", designerLabel="Sun Microsystems"),
            ],
        }

        def fake_query(query: str):
            for kind, payload in rows.items():
                if f"# query: {kind}" in query:
                    return list(payload)
            return []

        pairs = collect_domain("technology", query_fn=fake_query, sleep_s=0)
        users = [p[0] for p in pairs]
        self.assertIn("Who designed the Python programming language?", users)
        self.assertNotIn("Who designed the Java programming language?", users)

    def test_maths_seeds_and_query_shaping(self):
        def fake_query(query: str):
            if "# query: math_constants" in query:
                return [_bind(constLabel="pi", value="3.141592653589793")]
            if "# query: theorem_names" in query:
                return [_bind(theoremLabel="Bayes' theorem", personLabel="Thomas Bayes")]
            if "# query: prime_numbers" in query:
                return [_bind(value="13")]
            if "# query: composite_numbers" in query:
                return [_bind(value="9")]
            self.fail(f"unexpected query: {query}")

        pairs = collect_domain("maths", query_fn=fake_query, sleep_s=0)
        text = " ".join(u + " " + a for u, a in pairs)
        self.assertIn("approximate value of pi is 3.14159", text)
        self.assertIn("Bayes' theorem is named after Thomas Bayes", text)
        self.assertIn("Yes, 13 is a prime number.", text)
        self.assertIn("No, 9 is a composite number.", text)
        self.assertIn("State the Pythagorean theorem.", text)
        self.assertTrue(any(u == THEOREM_STATEMENTS[0][0] for u, _ in pairs))

    def test_health_seeds_and_malaria_conflict_drop(self):
        def fake_query(query: str):
            if "# query: disease_pathogens" in query:
                return [
                    _bind(diseaseLabel="malaria", pathogenLabel="Plasmodium falciparum"),
                    _bind(diseaseLabel="malaria", pathogenLabel="Plasmodium vivax"),
                    _bind(diseaseLabel="tuberculosis", pathogenLabel="Mycobacterium tuberculosis"),
                ]
            return []

        pairs = collect_domain("health", query_fn=fake_query, sleep_s=0)
        users = [p[0] for p in pairs]
        self.assertNotIn("What pathogen causes malaria?", users)
        self.assertIn("What pathogen causes tuberculosis?", users)
        self.assertIn(HEALTH_SEEDS[0][0], users)

    def test_apply_limit_rewrites_marked_query(self):
        q = apply_limit(QUERIES["lang_designers"], 12)
        self.assertIn("LIMIT 12", q)
        self.assertIn("# query: lang_designers", q)

    def test_format_helpers(self):
        self.assertEqual(format_year("1991.0"), "1991")
        self.assertIsNone(format_year("99999"))
        self.assertEqual(format_numeric("8.0"), "8")
        self.assertEqual(format_numeric("2.718281828"), "2.71828")
        self.assertEqual(fold_label("C₆H₈O₆"), "C6H8O6")
        self.assertEqual(fold_label("β-carotene"), "beta-carotene")

    def test_default_output_names_do_not_collide(self):
        names = {path.name for path in DEFAULT_OUTPUTS.values()}
        self.assertEqual(names, {"tech_facts.txt", "health_facts.txt", "maths_facts.txt"})
        for banned in ("capitals_priority.txt", "elements.txt", "wikidata_facts.txt"):
            self.assertNotIn(banned, names)
        self.assertEqual(set(DOMAIN_KINDS), {"technology", "health", "maths"})

    def test_cli_writes_separate_packs(self):
        def fake_query(query: str):
            if "# query: lang_designers" in query:
                return [_bind(langLabel="Python", designerLabel="Guido van Rossum")]
            if "# query: vitamin_formulas" in query:
                return [_bind(vitaminLabel="ascorbic acid", formula="C6H8O6")]
            if "# query: math_constants" in query:
                return [_bind(constLabel="e", value="2.718281828")]
            return []

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            code = wiki2_main(
                [
                    "--output-dir", str(dest),
                    "--sleep", "0",
                    "--queries",
                    "lang_designers",
                    "vitamin_formulas",
                    "math_constants",
                ],
                query_fn=fake_query,
            )
            self.assertEqual(code, 0)
            tech = (dest / "tech_facts.txt").read_text(encoding="utf-8")
            health = (dest / "health_facts.txt").read_text(encoding="utf-8")
            maths = (dest / "maths_facts.txt").read_text(encoding="utf-8")
            self.assertTrue(tech.startswith("User: Who designed the Python"))
            self.assertIn(" Assistant: Guido van Rossum designed", tech)
            self.assertIn("chemical formula of ascorbic acid", health)
            self.assertIn("How many chambers does the human heart have?", health)
            self.assertNotIn("take 500", health.lower())
            self.assertIn("approximate value of e", maths)
            self.assertIn("State the Pythagorean theorem.", maths)
            loaded = load_user_facts([dest / "tech_facts.txt"])
            self.assertEqual(loaded[0][0], "Who designed the Python programming language?")
            self.assertFalse((dest / "capitals_priority.txt").exists())
            self.assertFalse((dest / "elements.txt").exists())

    def test_cli_help_and_parse_defaults(self):
        args = parse_args([])
        self.assertEqual(args.domains, ["technology", "health", "maths"])
        self.assertTrue(str(args.output_dir).endswith("data/facts"))
        self.assertEqual(args.sleep, 1.0)

    def test_collect_kinds_unknown_raises(self):
        with self.assertRaises(ValueError):
            collect_kinds(["not_a_kind"], query_fn=lambda q: [], sleep_s=0)


if __name__ == "__main__":
    unittest.main()
