"""Download roneneldan/TinyStories parquet shards and write train-ready artifacts.

Output under data/tinystories/ (gitignored):

  text/train-XXXXX.txt   one story per line, internal newlines collapsed
  text/valid.txt
  vocab.json             BPE trained only on TinyStories (bpe_merges=6000)
  tokens/train-XXXXX.npy int32 id streams
  tokens/valid.npy
  manifest.json          shard order, vocab sha, token counts

--pack-stories writes a separate corpus (default data/tinystories_packed):
each story is followed by <|endofstory|>, and tokens/*.spans.npy records
[start, end) per story. Training windows are cut on those spans.
Reuse the existing 6000-merge vocab with --vocab so ids stay stable.

Does not read data/train.txt. Does not train a cabinet BPE.

Usage:
  python tools/prepare_tinystories.py
  python tools/prepare_tinystories.py --pack-stories --skip-download \\
      --text-dir data/tinystories/text --vocab data/tinystories/vocab.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HF_API_TREE = "https://huggingface.co/api/datasets/roneneldan/TinyStories/tree/main/data"
HF_RESOLVE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/{path}"
DEFAULT_OUT = ROOT / "data" / "tinystories"
PACKED_OUT = ROOT / "data" / "tinystories_packed"
DEFAULT_MERGES = 6000
DEFAULT_MAX_CHARS = 50_000_000


def collapse_story_line(text: str) -> str:
    """One story, one line. Internal newlines become spaces."""
    return " ".join((text or "").split())


def write_text_shard(path: Path, stories: Iterable[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as handle:
        for story in stories:
            line = collapse_story_line(story)
            if not line:
                continue
            handle.write(line + "\n")
            n += 1
    return n


def vocab_sha256(vocab_path: Path) -> str:
    payload = vocab_path.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def write_manifest(
    path: Path,
    *,
    vocab_path: Path,
    train_shards: Sequence[str],
    valid_shards: Sequence[str],
    train_tokens: int,
    valid_tokens: int,
    vocab_size: int,
    bpe_merges: int,
    dtype: str = "int32",
    extra: Optional[dict] = None,
) -> dict:
    payload = {
        "vocab_path": str(vocab_path.name),
        "vocab_sha256": vocab_sha256(vocab_path) if vocab_path.is_file() else "",
        "vocab_size": int(vocab_size),
        "bpe_merges": int(bpe_merges),
        "dtype": dtype,
        "train_shards": list(train_shards),
        "valid_shards": list(valid_shards),
        "train_tokens": int(train_tokens),
        "valid_tokens": int(valid_tokens),
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def iter_story_lines(paths: Sequence[Path]) -> Iterator[str]:
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.strip()
                if line:
                    yield line


def _list_parquet_files() -> tuple[list[str], list[str]]:
    with urllib.request.urlopen(HF_API_TREE, timeout=60) as resp:
        rows = json.loads(resp.read().decode("utf-8"))
    train: list[str] = []
    valid: list[str] = []
    for row in rows:
        path = str(row.get("path") or "")
        if not path.endswith(".parquet"):
            continue
        name = Path(path).name.lower()
        if name.startswith("validation") or name.startswith("valid"):
            valid.append(path)
        elif name.startswith("train"):
            train.append(path)
    if not train or not valid:
        raise RuntimeError(f"TinyStories parquet listing incomplete: train={train} valid={valid}")
    return sorted(train), sorted(valid)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def _read_parquet_text_column(path: Path) -> List[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to read TinyStories parquet shards. "
            "Install it in this venv: pip install pyarrow"
        ) from exc
    table = pq.read_table(path)
    names = set(table.column_names)
    col = "text" if "text" in names else ("story" if "story" in names else None)
    if col is None:
        raise RuntimeError(f"{path} has no text/story column; columns={table.column_names}")
    return [str(v) for v in table[col].to_pylist() if v]


def _shard_stem(hf_path: str, index: int, kind: str) -> str:
    if kind == "valid":
        return "valid"
    return f"train-{index:05d}"


def prepare(
    out_dir: Path,
    *,
    bpe_merges: int = DEFAULT_MERGES,
    max_chars: int = DEFAULT_MAX_CHARS,
    skip_download: bool = False,
    text_only: bool = False,
    pack_stories: bool = False,
    vocab_path: Optional[Path] = None,
    text_dir: Optional[Path] = None,
) -> dict:
    out_dir = Path(out_dir)
    owned_text = out_dir / "text"
    source_text = Path(text_dir) if text_dir is not None else owned_text
    token_dir = out_dir / "tokens"
    parquet_dir = out_dir / "parquet"
    token_dir.mkdir(parents=True, exist_ok=True)
    if text_dir is None:
        owned_text.mkdir(parents=True, exist_ok=True)

    if not skip_download:
        if text_dir is not None:
            raise ValueError("--text-dir is only valid with --skip-download")
        source_text = owned_text
        train_paths, valid_paths = _list_parquet_files()
        written_train: list[Path] = []
        for i, rel in enumerate(train_paths):
            dest = parquet_dir / Path(rel).name
            _download(HF_RESOLVE.format(path=rel), dest)
            stories = _read_parquet_text_column(dest)
            shard = source_text / f"{_shard_stem(rel, i, 'train')}.txt"
            write_text_shard(shard, stories)
            written_train.append(shard)
        written_valid: list[Path] = []
        for i, rel in enumerate(valid_paths):
            dest = parquet_dir / Path(rel).name
            _download(HF_RESOLVE.format(path=rel), dest)
            stories = _read_parquet_text_column(dest)
            shard = source_text / ("valid.txt" if i == 0 else f"valid-{i:05d}.txt")
            write_text_shard(shard, stories)
            written_valid.append(shard)
    else:
        written_train = sorted(source_text.glob("train-*.txt"))
        written_valid = [source_text / "valid.txt"] if (source_text / "valid.txt").is_file() else []
        if not written_train or not written_valid:
            raise FileNotFoundError(f"Need text/train-*.txt and text/valid.txt under {source_text}")

    if text_only:
        return {"text_train": [str(p) for p in written_train], "text_valid": [str(p) for p in written_valid]}

    from tokenizer.bpe import END_OF_STORY, BPETokenizer

    if vocab_path is not None:
        tokenizer = BPETokenizer.load(vocab_path)
    else:
        tokenizer = BPETokenizer.from_corpus(
            iter_story_lines(written_train),
            num_merges=int(bpe_merges),
            max_chars=int(max_chars),
        )
    eos_id = tokenizer.add_special_token(END_OF_STORY) if pack_stories else None
    saved_vocab = out_dir / "vocab.json"
    tokenizer.save(saved_vocab)

    def _encode_shard(shard: Path) -> tuple[object, Optional[str]]:
        docs = list(iter_story_lines([shard]))
        if pack_stories:
            ids, spans = tokenizer.encode_stories(docs, int(eos_id))
            span_path = token_dir / f"{shard.stem}.spans.npy"
            np_save_int64(span_path, spans)
            return ids.astype("int32"), f"tokens/{span_path.name}"
        return tokenizer.encode_corpus(docs).astype("int32"), None

    train_rel: list[str] = []
    train_spans: list[str] = []
    train_tokens = 0
    for shard in written_train:
        ids, span_rel = _encode_shard(shard)
        npy = token_dir / f"{shard.stem}.npy"
        np_save_int32(npy, ids)
        train_rel.append(f"tokens/{npy.name}")
        train_tokens += int(ids.size)
        if span_rel:
            train_spans.append(span_rel)

    valid_rel: list[str] = []
    valid_spans: list[str] = []
    valid_tokens = 0
    for shard in written_valid:
        ids, span_rel = _encode_shard(shard)
        npy = token_dir / f"{shard.stem}.npy"
        np_save_int32(npy, ids)
        valid_rel.append(f"tokens/{npy.name}")
        valid_tokens += int(ids.size)
        if span_rel:
            valid_spans.append(span_rel)

    extra = None
    if pack_stories:
        extra = {
            "pack": "story",
            "eos_token": END_OF_STORY,
            "eos_id": int(eos_id),
            "train_span_shards": train_spans,
            "valid_span_shards": valid_spans,
        }
    return write_manifest(
        out_dir / "manifest.json",
        vocab_path=saved_vocab,
        train_shards=train_rel,
        valid_shards=valid_rel,
        train_tokens=train_tokens,
        valid_tokens=valid_tokens,
        vocab_size=tokenizer.vocab_size,
        bpe_merges=bpe_merges,
        extra=extra,
    )


def np_save_int32(path: Path, ids) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), np.ascontiguousarray(ids, dtype=np.int32))


def np_save_int64(path: Path, ids) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), np.ascontiguousarray(ids, dtype=np.int64))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TinyStories text, BPE, and int32 token shards")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--bpe-merges", type=int, default=DEFAULT_MERGES)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--skip-download", action="store_true", help="Reuse existing text/ shards")
    parser.add_argument("--text-only", action="store_true", help="Write text shards only (no BPE)")
    parser.add_argument(
        "--pack-stories", action="store_true",
        help="Append <|endofstory|> and write per-story spans. Refuses to overwrite data/tinystories.",
    )
    parser.add_argument("--vocab", type=str, default=None, help="Reuse this BPE vocab instead of training one")
    parser.add_argument("--text-dir", type=str, default=None, help="Read story lines from this directory")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.pack_stories and out_dir.resolve() == DEFAULT_OUT.resolve():
        out_dir = PACKED_OUT
    payload = prepare(
        out_dir,
        bpe_merges=int(args.bpe_merges),
        max_chars=int(args.max_chars),
        skip_download=bool(args.skip_download),
        text_only=bool(args.text_only),
        pack_stories=bool(args.pack_stories),
        vocab_path=Path(args.vocab) if args.vocab else None,
        text_dir=Path(args.text_dir) if args.text_dir else None,
    )
    if args.pack_stories and not args.text_only:
        from training.tinystories_tokens import verify_story_pack
        verify_story_pack(out_dir)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
