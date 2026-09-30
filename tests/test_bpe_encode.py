"""Parallel BPE encode must match the serial token stream."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tokenizer.bpe import BPETokenizer


class BpeEncodeTests(unittest.TestCase):
    def test_parallel_matches_serial_and_join_encode(self):
        corpus = [
            "the cat sat on the mat",
            "the cat sat",
            "hello world",
            "spaces  stay",
        ] * 40
        tok = BPETokenizer.from_corpus(corpus, num_merges=20, max_chars=None)
        serial = tok.encode_corpus(corpus, workers=1)
        tok._encode_cache.clear()
        parallel = tok.encode_corpus(corpus, workers=2)
        np.testing.assert_array_equal(serial, parallel)
        joined = np.asarray(tok.encode(" ".join(corpus)), dtype=np.int64)
        np.testing.assert_array_equal(serial, joined)

    def test_empty_corpus(self):
        tok = BPETokenizer.from_corpus(["abc"], num_merges=4, max_chars=None)
        tokens = tok.encode_corpus([], workers=2)
        self.assertEqual(len(tokens), 0)

    def test_train_roundtrip_and_repeated_pair(self):
        text = "the cat sat on the mat. " * 40 + "hello hello hello"
        tok = BPETokenizer()
        tok.train(text, num_merges=40)
        sample = "the cat sat"
        self.assertEqual(tok.decode(tok.encode(sample)), sample)
        self.assertGreaterEqual(len(tok.merges), 1)

        simple = BPETokenizer()
        simple.train("aaa aaa aaa", num_merges=2)
        self.assertEqual(simple.merges[0], ("a", "a"))
        self.assertEqual(simple.decode(simple.encode("aaa")), "aaa")


if __name__ == "__main__":
    unittest.main()
