"""CLI flags for the interactive router (no Metal import)."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.router import router_enabled


def _parse(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", dest="router", action="store_true")
    parser.add_argument("--no-router", dest="router", action="store_false")
    parser.set_defaults(router=None)
    return parser.parse_args(argv)


class RouterEnabledTests(unittest.TestCase):
    def test_default_follows_chat_model_name(self):
        parsed = _parse([])
        self.assertIsNone(parsed.router)
        self.assertTrue(router_enabled(parsed.router, "Chat C=512 L=6 T=256"))
        self.assertFalse(router_enabled(parsed.router, "Stories C=256 L=6 T=256"))

    def test_no_router_flag(self):
        parsed = _parse(["--no-router"])
        self.assertFalse(router_enabled(parsed.router, "Chat C=512 L=6 T=256"))

    def test_router_flag_on_story_name(self):
        parsed = _parse(["--router"])
        self.assertTrue(router_enabled(parsed.router, "Stories C=256 L=6 T=256"))
