"""
training/dataset.py

Turns the corpus (list of sentence strings) into a contiguous token stream
and yields (X, Y) sliding-window batches, Y being X shifted by one position
(next-character prediction).
"""

from typing import Iterator, List, Optional, Tuple

import numpy as np

from logging_config import logger


def build_story_windows(
    spans: np.ndarray,
    max_len: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Window starts that stay inside one story span.

    Returns ``(starts, widths)``. ``widths`` is the number of source tokens
    (including the end-of-story id). A width of ``max_len + 1`` is a full
    window. A shorter width is one padded example for a short story.
    The last window of a long story is aligned to the story end so the
    end token is a training target.
    """
    spans = np.asarray(spans, dtype=np.int64).reshape(-1, 2)
    need = int(max_len) + 1
    stride = max(1, int(stride))
    starts: List[int] = []
    widths: List[int] = []
    for a, b in spans:
        start = int(a)
        story_len = int(b) - start
        if story_len < 2:
            continue
        if story_len <= need:
            starts.append(start)
            widths.append(story_len)
            continue
        last = story_len - need
        rel = 0
        while True:
            starts.append(start + rel)
            widths.append(need)
            if rel >= last:
                break
            nxt = rel + stride
            rel = last if nxt > last else nxt
    if not starts:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32)
    return np.asarray(starts, dtype=np.int64), np.asarray(widths, dtype=np.int32)


class WindowedDataset:
    def __init__(
        self,
        corpus: List[str],
        tokenizer,
        max_len: int,
        batch_size: int,
        window_stride: int = 1,
        tokens: Optional[np.ndarray] = None,
        copy_tokens: bool = True,
        spans: Optional[np.ndarray] = None,
        pad_id: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.window_stride = max(1, int(window_stride))
        self.pad_id = int(pad_id)
        self.spans = None if spans is None else np.asarray(spans, dtype=np.int64).reshape(-1, 2)

        # Prefer a resume cache (tokens.npy) so BPE encode is not repeated.
        # Else streaming encode (BPE): avoids " ".join of entire TinyStories
        # (~100M chars) and reuses a per-word encode cache across docs.
        if tokens is not None and (
            not copy_tokens or getattr(tokens, "keep_memmap", False)
        ):
            self.tokens = tokens
            logger.info("Using prebuilt token stream (%s tokens, no host copy)", len(self.tokens))
        elif tokens is not None:
            self.tokens = np.asarray(tokens, dtype=np.int64)
            logger.info("Using pre-encoded token stream (%s tokens)", len(self.tokens))
        elif hasattr(tokenizer, "encode_corpus"):
            self.tokens = np.asarray(tokenizer.encode_corpus(corpus), dtype=np.int64)
        else:
            logger.info("Encoding corpus to tokens (%s documents)...", len(corpus))
            joined_text = " ".join(corpus)
            self.tokens = np.array(tokenizer.encode(joined_text), dtype=np.int64)
            logger.info("Encoded %s tokens from %s documents", len(self.tokens), len(corpus))
        self.total_tokens = len(self.tokens)

        if self.total_tokens < max_len + 1 and self.spans is None:
            raise ValueError(
                f"Corpus has only {self.total_tokens} tokens, need at least "
                f"{max_len + 1} for a single window (max_len={max_len}). "
                f"Use a larger dataset (e.g. tiny_stories) or lower max_len / "
                f"context window to <= {max(1, self.total_tokens - 1)}."
            )
        self._win_starts: Optional[np.ndarray] = None
        self._win_widths: Optional[np.ndarray] = None
        if self.spans is not None:
            self._win_starts, self._win_widths = build_story_windows(
                self.spans, self.max_len, self.window_stride,
            )
            logger.info(
                "Story windows: %s (spans=%s, max_len=%s, stride=%s)",
                len(self._win_starts), len(self.spans), self.max_len, self.window_stride,
            )

    def num_dense_windows(self) -> int:
        """Sliding starts with stride=1 (upper bound on unique windows)."""
        return self.total_tokens - self.max_len

    def num_windows(self) -> int:
        if self._win_starts is not None:
            return int(len(self._win_starts))
        dense = self.num_dense_windows()
        if dense <= 0:
            return 0
        return int(np.arange(0, dense, self.window_stride).size)

    def num_batches(self) -> int:
        return max(1, self.num_windows() // self.batch_size)

    def iter_batches(self, shuffle: bool = True, rng: np.random.Generator = None) -> Iterator[List[Tuple[np.ndarray, np.ndarray]]]:
        """Yields lists of (x, y) sequence pairs, one list per mini-batch."""
        rng = rng or np.random.default_rng()
        if self._win_starts is not None:
            order = np.arange(len(self._win_starts))
            if shuffle:
                rng.shuffle(order)
            need = self.max_len + 1
            for b in range(self.num_batches()):
                batch_ix = order[b * self.batch_size : (b + 1) * self.batch_size]
                if len(batch_ix) == 0:
                    continue
                pairs = []
                for ix in batch_ix:
                    start = int(self._win_starts[ix])
                    width = int(self._win_widths[ix])
                    if width == need:
                        x = np.asarray(self.tokens[start : start + self.max_len], dtype=np.int64)
                        y = np.asarray(self.tokens[start + 1 : start + width], dtype=np.int64)
                    else:
                        raw = np.asarray(self.tokens[start : start + width], dtype=np.int64)
                        x = np.full(self.max_len, self.pad_id, dtype=np.int64)
                        y = np.full(self.max_len, -1, dtype=np.int64)
                        x[: width - 1] = raw[:-1]
                        y[: width - 1] = raw[1:]
                    pairs.append((x, y))
                yield pairs
            return

        starts = np.arange(0, self.num_dense_windows(), self.window_stride)
        if shuffle:
            rng.shuffle(starts)

        for b in range(self.num_batches()):
            batch_starts = starts[b * self.batch_size : (b + 1) * self.batch_size]
            if len(batch_starts) == 0:
                continue
            pairs = []
            for start in batch_starts:
                x = np.asarray(self.tokens[start : start + self.max_len], dtype=np.int64)
                y = np.asarray(self.tokens[start + 1 : start + self.max_len + 1], dtype=np.int64)
                pairs.append((x, y))
            yield pairs
