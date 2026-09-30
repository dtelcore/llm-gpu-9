"""
training/token_cache.py

Persist the BPE/char token id stream next to a checkpoint so resume does not
re-encode TinyStories on every start.

Vocab/merges are already saved as vocab.json. This cache is the expensive
second half: 557k docs → int64 ids (~2 min of Python BPE encode on this box).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from logging_config import logger

_SPLIT_FILES = {
    "train": ("tokens.npy", "tokens.meta.json"),
    "val": ("val_tokens.npy", "val_tokens.meta.json"),
}


def split_cache_paths(run_dir: Path, split: str) -> Tuple[Path, Path]:
    try:
        npy_name, meta_name = _SPLIT_FILES[split]
    except KeyError as exc:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}") from exc
    root = Path(run_dir)
    return root / npy_name, root / meta_name


def fingerprint(tokenizer, corpus: List[str]) -> Dict:
    """Cheap identity of (tokenizer, corpus). Misses → re-encode, never silent mismatch."""
    merges = [list(p) for p in getattr(tokenizer, "merges", [])]
    vocab = list(getattr(tokenizer, "vocab", []))
    blob = json.dumps({"m": merges, "v": vocab}, ensure_ascii=False, separators=(",", ":"))
    return {
        "vocab_sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
        "n_docs": len(corpus),
        "n_chars": int(sum(len(s) for s in corpus)),
    }


def try_load_tokens(npy_path: Path, meta_path: Path, expected: Dict) -> Optional[np.ndarray]:
    if not npy_path.is_file() or not meta_path.is_file():
        return None
    try:
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Token cache meta unreadable (%s); re-encoding. %s", meta_path, exc)
        return None
    for key, value in expected.items():
        if stored.get(key) != value:
            logger.info(
                "Token cache miss at %s (%s %s != %s); re-encoding",
                npy_path, key, stored.get(key), value,
            )
            return None
    try:
        tokens = np.load(str(npy_path))
    except (OSError, ValueError) as exc:
        logger.warning("Token cache npy unreadable (%s); re-encoding. %s", npy_path, exc)
        return None
    n_tokens = stored.get("n_tokens")
    if n_tokens is not None and int(tokens.size) != int(n_tokens):
        logger.info(
            "Token cache miss at %s (n_tokens %s != %s); re-encoding",
            npy_path, tokens.size, n_tokens,
        )
        return None
    logger.info(
        "Loaded token stream cache %s (%s tokens)",
        npy_path, int(tokens.size),
    )
    return tokens


def save_tokens(npy_path: Path, meta_path: Path, tokens: np.ndarray, meta: Dict) -> None:
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(tokens, dtype=np.int64)
    np.save(str(npy_path), arr)
    payload = dict(meta)
    payload["n_tokens"] = int(arr.size)
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved token stream cache %s (%s tokens)", npy_path, int(arr.size))
