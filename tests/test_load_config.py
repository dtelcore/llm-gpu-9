"""First-run config fallback: missing output/configs/training_config.json."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cli_common
from paths import BUNDLED_DEFAULT_CONFIG, BUNDLED_SMOKE_CORPUS


class LoadConfigFallbackTests(unittest.TestCase):
    def test_bundled_story_sub1m_exists(self):
        self.assertTrue(BUNDLED_DEFAULT_CONFIG.is_file())
        with BUNDLED_DEFAULT_CONFIG.open(encoding="utf-8") as f:
            cfg = json.load(f)
        self.assertEqual(cfg["dataset"]["name"], "data_dir")
        self.assertTrue(cfg["dataset"]["combine"])

    def test_bundled_smoke_corpus_is_long_enough_for_story_sub1m(self):
        self.assertTrue(BUNDLED_SMOKE_CORPUS.is_file())
        text = BUNDLED_SMOKE_CORPUS.read_text(encoding="utf-8")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        self.assertGreaterEqual(len(lines), 40)
        self.assertGreaterEqual(len(text), 2000)

    def test_missing_default_copies_bundled_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            default = tmp_path / "training_config.json"
            legacy = tmp_path / "legacy.json"
            self.assertFalse(default.exists())
            with patch.object(cli_common, "DEFAULT_CONFIG_PATH", default), \
                 patch.object(cli_common, "LEGACY_CONFIG_PATH", legacy), \
                 patch.object(cli_common, "ensure_output_dirs", lambda: None):
                cfg = cli_common.load_config(str(default))
            self.assertTrue(default.is_file())
            self.assertEqual(cfg["dataset"]["name"], "data_dir")
            self.assertEqual(cfg["model"]["embedding_dim"], 128)

    def test_legacy_wins_over_bundled(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            default = tmp_path / "training_config.json"
            legacy = tmp_path / "legacy.json"
            legacy.write_text(
                json.dumps({"model": {"name": "legacy"}, "dataset": {"name": "minimal"}}),
                encoding="utf-8",
            )
            with patch.object(cli_common, "DEFAULT_CONFIG_PATH", default), \
                 patch.object(cli_common, "LEGACY_CONFIG_PATH", legacy):
                cfg = cli_common.load_config(str(default))
            self.assertFalse(default.exists())
            self.assertEqual(cfg["model"]["name"], "legacy")

    def test_explicit_missing_path_raises(self):
        missing = Path(tempfile.gettempdir()) / "no_such_training_config.json"
        if missing.exists():
            missing.unlink()
        with self.assertRaises(FileNotFoundError):
            cli_common.load_config(str(missing))


if __name__ == "__main__":
    unittest.main()
