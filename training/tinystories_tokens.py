"""Load prebuilt TinyStories int32 token shards as a concatenated memmap.

Cabinet recipes omit dataset.vocab_path / dataset.token_dir and never enter here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from logging_config import logger
from paths import PROJECT_ROOT


class ConcatenatedMemmap:
    """Read-only concatenation of memmaps. Slices stay off the full-copy path."""

    keep_memmap = True

    def __init__(self, arrays: Sequence[np.ndarray]) -> None:
        if not arrays:
            raise ValueError("ConcatenatedMemmap needs at least one array")
        self._arrays = list(arrays)
        lengths = [int(a.shape[0]) for a in self._arrays]
        offsets = [0]
        for n in lengths:
            offsets.append(offsets[-1] + n)
        self._offsets = offsets
        self._n = offsets[-1]
        self.dtype = np.dtype(self._arrays[0].dtype)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, key: Union[int, slice]) -> np.ndarray:
        if isinstance(key, slice):
            start, stop, step = key.indices(self._n)
            if start >= stop:
                return np.empty(0, dtype=self.dtype)
            if step != 1:
                return np.asarray([self[i] for i in range(start, stop, step)], dtype=self.dtype)
            pieces: List[np.ndarray] = []
            idx = 0
            while idx < len(self._arrays) and self._offsets[idx + 1] <= start:
                idx += 1
            pos = start
            while pos < stop and idx < len(self._arrays):
                arr_start = self._offsets[idx]
                local = pos - arr_start
                take = min(stop - pos, int(self._arrays[idx].shape[0]) - local)
                pieces.append(self._arrays[idx][local : local + take])
                pos += take
                idx += 1
            if len(pieces) == 1:
                return pieces[0]
            return np.concatenate(pieces)
        if key < 0:
            key += self._n
        if key < 0 or key >= self._n:
            raise IndexError(key)
        idx = 0
        while idx < len(self._arrays) and self._offsets[idx + 1] <= key:
            idx += 1
        return self._arrays[idx][key - self._offsets[idx]]


def dataset_uses_prebuilt_tokens(dataset_cfg: Optional[dict]) -> bool:
    if not dataset_cfg:
        return False
    return bool(dataset_cfg.get("vocab_path") and dataset_cfg.get("token_dir"))


def resolve_under_root(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def load_manifest(token_dir: Path) -> dict[str, Any]:
    path = token_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"TinyStories manifest missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _open_shards(token_dir: Path, rels: Sequence[str]) -> ConcatenatedMemmap:
    arrays = []
    for rel in rels:
        path = _shard_path(token_dir, rel)
        if not path.is_file():
            raise FileNotFoundError(f"Token shard missing: {path}")
        arrays.append(np.load(str(path), mmap_mode="r"))
    return ConcatenatedMemmap(arrays)


def load_story_pack(token_dir: str | Path, split: str) -> Tuple[ConcatenatedMemmap, Optional[np.ndarray], int]:
    """Tokens, story spans in that token space, and the end-of-story id.

    Spans are None when the manifest has no story packing. The pad id is then 0.
    """
    root = resolve_under_root(token_dir)
    manifest = load_manifest(root)
    tokens = load_split_tokens(token_dir, split)
    span_key = "train_span_shards" if split == "train" else "valid_span_shards"
    token_key = "train_shards" if split == "train" else "valid_shards"
    span_rels = list(manifest.get(span_key) or [])
    if not span_rels:
        return tokens, None, 0
    token_rels = list(manifest.get(token_key) or [])
    if len(span_rels) != len(token_rels):
        raise ValueError(f"{span_key} and {token_key} length differ in {root / 'manifest.json'}")
    pieces = []
    offset = 0
    for span_rel, tok_rel in zip(span_rels, token_rels):
        span_path = _shard_path(root, span_rel)
        tok_path = _shard_path(root, tok_rel)
        local = np.load(str(span_path))
        local = np.asarray(local, dtype=np.int64).reshape(-1, 2) + offset
        pieces.append(local)
        offset += int(np.load(str(tok_path), mmap_mode="r").shape[0])
    spans = np.concatenate(pieces) if pieces else np.zeros((0, 2), dtype=np.int64)
    pad_id = int(manifest.get("eos_id", 0))
    logger.info(
        "Loaded TinyStories %s story spans: %s stories, eos_id=%s",
        split, len(spans), pad_id,
    )
    return tokens, spans, pad_id


def _shard_path(token_dir: Path, rel: str) -> Path:
    path = token_dir / rel
    if path.is_file():
        return path
    alt = token_dir / Path(rel).name
    return alt if alt.is_file() else path


def verify_story_pack(token_dir: str | Path) -> dict[str, Any]:
    """Check spans abut, cover each shard, and end on the end-of-story id."""
    root = resolve_under_root(token_dir)
    manifest = load_manifest(root)
    eos_id = int(manifest["eos_id"])
    problems: List[str] = []
    for split, span_key, tok_key in (
        ("train", "train_span_shards", "train_shards"),
        ("valid", "valid_span_shards", "valid_shards"),
    ):
        span_rels = list(manifest.get(span_key) or [])
        tok_rels = list(manifest.get(tok_key) or [])
        if len(span_rels) != len(tok_rels) or not span_rels:
            problems.append(f"{split}: span shards missing or miscounted")
            continue
        for span_rel, tok_rel in zip(span_rels, tok_rels):
            spans = np.load(str(_shard_path(root, span_rel)))
            spans = np.asarray(spans, dtype=np.int64).reshape(-1, 2)
            tokens = np.load(str(_shard_path(root, tok_rel)), mmap_mode="r")
            n = int(tokens.shape[0])
            label = f"{split}:{Path(tok_rel).name}"
            if len(spans) == 0 or int(spans[0, 0]) != 0 or int(spans[-1, 1]) != n:
                problems.append(f"{label}: spans do not cover 0..{n}")
                continue
            if len(spans) > 1 and not np.all(spans[1:, 0] == spans[:-1, 1]):
                problems.append(f"{label}: spans leave a gap or overlap")
            ends = spans[:, 1] - 1
            if not np.all(tokens[ends] == eos_id):
                problems.append(f"{label}: a story does not end on eos_id {eos_id}")
    if problems:
        raise ValueError("story pack check failed: " + "; ".join(problems))
    return manifest


def load_split_tokens(token_dir: str | Path, split: str) -> ConcatenatedMemmap:
    root = resolve_under_root(token_dir)
    manifest = load_manifest(root)
    key = "train_shards" if split == "train" else "valid_shards"
    rels = list(manifest.get(key) or [])
    if not rels:
        raise FileNotFoundError(f"manifest {root / 'manifest.json'} has no {key}")
    tokens = _open_shards(root, rels)
    logger.info(
        "Loaded TinyStories %s tokens from %s shards (%s tokens, dtype=%s)",
        split, len(rels), len(tokens), tokens.dtype,
    )
    return tokens


def resolve_vocab_path(dataset_cfg: dict) -> Path:
    path = resolve_under_root(dataset_cfg["vocab_path"])
    if not path.is_file():
        raise FileNotFoundError(
            f"Prebuilt vocab missing: {path}. Run python tools/prepare_tinystories.py first."
        )
    return path
