"""Safe calculator: AST + Decimal, no eval."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.calc import try_calc


class CalcTests(unittest.TestCase):
    def test_add(self):
        self.assertEqual(try_calc("2+2"), "4")
        self.assertEqual(try_calc(" 2 + 2 "), "4")

    def test_power_caret(self):
        self.assertEqual(try_calc("(3^2)/4"), "2.25")

    def test_rejects_import_and_names(self):
        self.assertIsNone(try_calc("__import__('os')"))
        self.assertIsNone(try_calc("os.system('ls')"))
        self.assertIsNone(try_calc("foo + 1"))

    def test_rejects_prose_and_factorial_question(self):
        self.assertIsNone(try_calc("What is 0 factorial?"))
        self.assertIsNone(try_calc("What is the capital of France?"))

    def test_division_by_zero_is_not_an_answer(self):
        self.assertIsNone(try_calc("1/0"))
