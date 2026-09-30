"""
tokenizer/bpe.py

Whitespace-aware BPE over character symbols. Default tokenizer for new training
runs (see tokenizer/factory.py). Char remains available via --tokenizer char.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

from logging_config import logger


def _word_tokens(text: str) -> List[str]:
    """Split on whitespace; keep spaces as separate tokens so decode round-trips."""
    parts: List[str] = []
    buf = []
    for ch in text:
        if ch.isspace():
            if buf:
                parts.append("".join(buf))
                buf = []
            parts.append(ch)
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return parts


def _get_stats(seqs: List[List[str]]) -> Counter:
    stats: Counter = Counter()
    for seq in seqs:
        for i in range(len(seq) - 1):
            stats[(seq[i], seq[i + 1])] += 1
    return stats


def _merge_pair(seqs: List[List[str]], pair: Tuple[str, str]) -> List[List[str]]:
    a, b = pair
    merged = a + b
    out: List[List[str]] = []
    for seq in seqs:
        new_seq: List[str] = []
        i = 0
        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                new_seq.append(merged)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        out.append(new_seq)
    return out


class BPETokenizer:
    """Whitespace-aware BPE; API-compatible with CharacterGPTTokenizer for train/trace."""

    def __init__(self) -> None:
        self.vocab: List[str] = []
        self.merges: List[Tuple[str, str]] = []
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self.vocab_size: int = 0
        self._encode_cache: Dict[str, List[int]] = {}

    @staticmethod
    def _sample_text(corpus: Iterable[str], max_chars: Optional[int]) -> str:
        """Join corpus docs with spaces, stopping once max_chars is reached.

        Avoids materializing the full TinyStories join (~100M+ chars) when only
        a BPE training sample is needed.
        """
        if max_chars is None:
            return " ".join(corpus)
        limit = int(max_chars)
        if limit <= 0:
            return ""
        parts: List[str] = []
        n = 0
        for doc in corpus:
            if n >= limit:
                break
            if n > 0:
                parts.append(" ")
                n += 1
                if n >= limit:
                    break
            take = limit - n
            if len(doc) <= take:
                parts.append(doc)
                n += len(doc)
            else:
                parts.append(doc[:take])
                n = limit
                break
        return "".join(parts)

    @classmethod
    def from_corpus(
        cls,
        corpus: Iterable[str],
        num_merges: int = 200,
        max_chars: Optional[int] = 200_000,
    ) -> "BPETokenizer":
        text = cls._sample_text(corpus, max_chars)
        logger.info(
            "BPE training sample: %s chars (cap=%s), merges=%s",
            len(text),
            max_chars if max_chars is not None else "none",
            num_merges,
        )
        inst = cls()
        inst.train(text, num_merges=num_merges)
        return inst

    def train(self, text: str, num_merges: int = 200) -> None:
        chars = sorted(set(text))
        seqs: List[List[str]] = []
        for word in _word_tokens(text):
            seqs.append(list(word))

        merges: List[Tuple[str, str]] = []
        vocab_set = set(chars)
        total = int(num_merges)
        log_every = max(1, total // 10)
        for step in range(total):
            stats = _get_stats(seqs)
            if not stats:
                break
            pair, _count = stats.most_common(1)[0]
            if _count < 2:
                break
            merges.append(pair)
            vocab_set.add(pair[0] + pair[1])
            seqs = _merge_pair(seqs, pair)
            if (step + 1) % log_every == 0 or (step + 1) == total:
                logger.info("BPE merge progress: %s/%s", step + 1, total)

        self.merges = merges
        self.vocab = sorted(vocab_set, key=lambda s: (len(s), s))
        self._token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self._id_to_token = {i: t for t, i in self._token_to_id.items()}
        self.vocab_size = len(self.vocab)

    def _apply_merges(self, symbols: List[str]) -> List[str]:
        seq = list(symbols)
        for a, b in self.merges:
            merged = a + b
            new_seq: List[str] = []
            i = 0
            while i < len(seq):
                if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                    new_seq.append(merged)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            seq = new_seq
        return seq

    def _encode_word(self, word: str) -> List[int]:
        """Encode one whitespace-delimited piece; cached (TinyStories has heavy reuse)."""
        cache = self._encode_cache
        cached = cache.get(word)
        if cached is not None:
            return cached
        pieces = self._apply_merges(list(word))
        ids: List[int] = []
        for p in pieces:
            tid = self._token_to_id.get(p)
            if tid is not None:
                ids.append(tid)
            else:
                for ch in p:
                    cid = self._token_to_id.get(ch)
                    if cid is not None:
                        ids.append(cid)
        cache[word] = ids
        return ids

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for word in _word_tokens(text):
            ids.extend(self._encode_word(word))
        return ids

    def encode_corpus(self, corpus: List[str], progress_every: int = 25_000) -> "np.ndarray":
        """Encode docs joined by single spaces (same token stream as ' '.join + encode)."""
        import numpy as np

        n = len(corpus)
        logger.info("BPE-encoding corpus: %s documents...", n)
        space_ids = self._encode_word(" ")
        chunks: List[List[int]] = []
        buf: List[int] = []
        for i, doc in enumerate(corpus):
            if i:
                buf.extend(space_ids)
            for word in _word_tokens(doc):
                buf.extend(self._encode_word(word))
            if len(buf) >= 1_000_000:
                chunks.append(buf)
                buf = []
            if progress_every and (i + 1) % progress_every == 0:
                logger.info("BPE encode progress: %s/%s docs (cache=%s words)", i + 1, n, len(self._encode_cache))
        if buf:
            chunks.append(buf)
        if not chunks:
            return np.array([], dtype=np.int64)
        tokens = np.concatenate([np.asarray(c, dtype=np.int64) for c in chunks])
        logger.info(
            "BPE encode done: %s tokens from %s docs (unique-word cache=%s)",
            len(tokens), n, len(self._encode_cache),
        )
        return tokens

    def decode(self, ids: List[int]) -> str:
        return "".join(self._id_to_token.get(i, "") for i in ids)

    def id_to_token(self, token_id: int) -> str:
        return self._id_to_token.get(token_id, "<unk>")

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, -1)

    def save_vocab(self, filepath: Union[str, Path]) -> None:
        self.save(filepath)

    def save(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "type": "bpe",
            "vocab": self.vocab,
            "merges": [[a, b] for a, b in self.merges],
            "token_to_id": self._token_to_id,
            "id_to_token": {str(k): v for k, v in self._id_to_token.items()},
        }
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("BPE vocab saved to %s (%s tokens)", path, self.vocab_size)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BPETokenizer":
        state = json.loads(Path(filepath).read_text(encoding="utf-8"))
        inst = cls()
        inst.vocab = state["vocab"]
        inst.merges = [tuple(p) for p in state["merges"]]
        inst._token_to_id = {str(k): int(v) for k, v in state["token_to_id"].items()}
        inst._id_to_token = {int(k): v for k, v in state["id_to_token"].items()}
        inst.vocab_size = len(inst.vocab)
        return inst

    def coverage_stats(self, text: str, context_tokens: int) -> Dict[str, float]:
        """Compare char vs BPE semantic span inside a fixed token window."""
        char_len = len(text)
        ids = self.encode(text)
        window = ids[:context_tokens]
        decoded = self.decode(window)
        return {
            "char_len": float(char_len),
            "bpe_token_count": float(len(ids)),
            "chars_per_token": float(char_len) / max(1, len(ids)),
            "context_token_window": float(context_tokens),
            "chars_covered_in_window": float(len(decoded)),
            "compression_vs_chars": float(char_len) / max(1, len(ids)),
        }
