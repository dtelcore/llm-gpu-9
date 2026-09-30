"""Disk cache for encoded token streams used on train resume."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.dataset import WindowedDataset
from training.token_cache import fingerprint, save_tokens, split_cache_paths, try_load_tokens


class _DummyTok:
    vocab_size = 4
    vocab = ["a", "b", "c", " "]
    merges = [("a", "b")]

    def encode(self, text: str):
        table = {ch: i for i, ch in enumerate(self.vocab)}
        return [table[c] for c in text if c in table]


class TokenCacheTests(unittest.TestCase):
    def test_round_trip_and_mismatch(self):
        corpus = ["ab c", "aa"]
        tok = _DummyTok()
        expected = fingerprint(tok, corpus)
        with tempfile.TemporaryDirectory() as tmp:
            npy, meta = split_cache_paths(Path(tmp), "train")
            self.assertIsNone(try_load_tokens(npy, meta, expected))
            tokens = np.arange(16, dtype=np.int64)
            save_tokens(npy, meta, tokens, expected)
            loaded = try_load_tokens(npy, meta, expected)
            self.assertIsNotNone(loaded)
            np.testing.assert_array_equal(np.asarray(loaded), tokens)

            other = dict(expected)
            other["n_docs"] = 99
            self.assertIsNone(try_load_tokens(npy, meta, other))

    def test_windowed_dataset_accepts_preencoded(self):
        tok = _DummyTok()
        tokens = np.arange(20, dtype=np.int64)
        ds = WindowedDataset(["unused"], tok, max_len=8, batch_size=2, tokens=tokens)
        self.assertEqual(ds.total_tokens, 20)
        self.assertEqual(ds.num_windows(), 12)


if __name__ == "__main__":
    unittest.main()
