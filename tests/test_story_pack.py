"""Story spans, intra-story windows, and padding ignored by the loss."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tokenizer.bpe import END_OF_STORY, BPETokenizer
from training.dataset import WindowedDataset, build_story_windows
from training.loss import softmax_cross_entropy


class StoryWindowTests(unittest.TestCase):
    def test_windows_stay_inside_one_story_and_pad_the_short_one(self):
        # story A: 1 2 <eos>     story B: 3 4 5 6 7 <eos>
        tokens = np.array([1, 2, 9, 3, 4, 5, 6, 7, 9], dtype=np.int64)
        spans = np.array([[0, 3], [3, 9]], dtype=np.int64)
        starts, widths = build_story_windows(spans, max_len=4, stride=128)
        self.assertEqual(len(starts), 3)
        tok = type("T", (), {"vocab_size": 10})()
        ds = WindowedDataset(
            [], tok, max_len=4, batch_size=4, window_stride=128,
            tokens=tokens, copy_tokens=False, spans=spans, pad_id=9,
        )
        batch = next(ds.iter_batches(shuffle=False))
        self.assertEqual(len(batch), 3)
        x0, y0 = batch[0]
        np.testing.assert_array_equal(x0, [1, 2, 9, 9])
        np.testing.assert_array_equal(y0, [2, 9, -1, -1])
        x1, y1 = batch[1]
        np.testing.assert_array_equal(x1, [3, 4, 5, 6])
        np.testing.assert_array_equal(y1, [4, 5, 6, 7])
        x2, y2 = batch[2]
        np.testing.assert_array_equal(x2, [4, 5, 6, 7])
        np.testing.assert_array_equal(y2, [5, 6, 7, 9])
        for x, y in batch:
            real = y[y >= 0]
            self.assertTrue(np.all((real == 9) | ((real >= 1) & (real <= 7))))
            if 1 in x or 2 in y:
                self.assertNotIn(3, real.tolist())

    def test_long_story_includes_the_end_token(self):
        eos = 4
        body = np.arange(1, 21, dtype=np.int64)
        tokens = np.concatenate([body, np.array([eos], dtype=np.int64)])
        spans = np.array([[0, len(tokens)]], dtype=np.int64)
        starts, widths = build_story_windows(spans, max_len=8, stride=4)
        last = int(starts[-1])
        self.assertEqual(int(widths[-1]), 9)
        self.assertEqual(int(tokens[last + 8]), eos)
        self.assertTrue(np.all(starts + widths <= len(tokens)))


class PackEncodeTests(unittest.TestCase):
    def test_end_token_is_appended_and_ids_do_not_move(self):
        corpus = ["lily found a key", "tom went home"]
        tok = BPETokenizer.from_corpus(corpus, num_merges=8, max_chars=None)
        before = dict(tok._token_to_id)
        eos = tok.add_special_token(END_OF_STORY)
        self.assertEqual(eos, len(before))
        for word, idx in before.items():
            self.assertEqual(tok.token_to_id(word), idx)
        ids, spans = tok.encode_stories(corpus, eos, workers=1)
        self.assertEqual(int(spans[0, 0]), 0)
        self.assertEqual(int(spans[-1, 1]), len(ids))
        np.testing.assert_array_equal(spans[1:, 0], spans[:-1, 1])
        self.assertTrue(np.all(ids[spans[:, 1] - 1] == eos))
        self.assertEqual(tok.decode(ids[int(spans[0, 0]): int(spans[0, 1]) - 1]), corpus[0])
        again, _ = tok.encode_stories(corpus, eos, workers=2)
        np.testing.assert_array_equal(ids, again)


class MaskedLossTests(unittest.TestCase):
    def test_padding_is_dropped_from_the_mean_and_the_gradient(self):
        logits = np.array(
            [[0.0, 4.0], [3.0, 0.0], [1.0, -1.0]],
            dtype=np.float64,
        )
        full, d_full = softmax_cross_entropy(logits[:2], np.array([1, 0]))
        masked, d_masked = softmax_cross_entropy(logits, np.array([1, 0, -1]))
        self.assertAlmostEqual(masked, full, places=6)
        np.testing.assert_allclose(d_masked[:2], d_full, atol=1e-6)
        np.testing.assert_array_equal(d_masked[2], 0.0)


if __name__ == "__main__":
    unittest.main()
