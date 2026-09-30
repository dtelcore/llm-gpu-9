#!/usr/bin/env python3
"""Confirm the v0.1.1 story-packed TinyStories corpus. Does not rebuild it."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PACKED = ROOT / "data" / "tinystories_packed"
EXPECTED_VOCAB = 6103
EXPECTED_TRAIN = 771869497
EXPECTED_VALID = 7790045
EXPECTED_EOS = "<|endofstory|>"


def main() -> int:
    manifest_path = PACKED / "manifest.json"
    if not manifest_path.is_file():
        print(f"missing {manifest_path}", file=sys.stderr)
        print("Build it with:", file=sys.stderr)
        print("  python tools/prepare_tinystories.py", file=sys.stderr)
        print(
            "  python tools/prepare_tinystories.py --pack-stories --skip-download "
            "--text-dir data/tinystories/text --vocab data/tinystories/vocab.json",
            file=sys.stderr,
        )
        return 1
    from training.tinystories_tokens import verify_story_pack

    problems: list[str] = []
    try:
        manifest = verify_story_pack(PACKED)
    except Exception as exc:
        problems.append(str(exc))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("vocab_size", 0)) != EXPECTED_VOCAB:
        problems.append(f"vocab_size {manifest.get('vocab_size')} != {EXPECTED_VOCAB}")
    if int(manifest.get("train_tokens", 0)) != EXPECTED_TRAIN:
        problems.append(f"train_tokens {manifest.get('train_tokens')} != {EXPECTED_TRAIN}")
    if int(manifest.get("valid_tokens", 0)) != EXPECTED_VALID:
        problems.append(f"valid_tokens {manifest.get('valid_tokens')} != {EXPECTED_VALID}")
    if manifest.get("eos_token") != EXPECTED_EOS:
        problems.append(f"eos_token {manifest.get('eos_token')!r} != {EXPECTED_EOS!r}")
    if int(manifest.get("eos_id", -1)) != EXPECTED_VOCAB - 1:
        problems.append(f"eos_id {manifest.get('eos_id')} != {EXPECTED_VOCAB - 1}")
    if problems:
        print("data check failed:", file=sys.stderr)
        for item in problems:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print("selected: story-packed TinyStories")
    print(f"vocab: {EXPECTED_VOCAB} ({EXPECTED_EOS} id {EXPECTED_VOCAB - 1})")
    print(f"train_tokens: {EXPECTED_TRAIN}")
    print(f"valid_tokens: {EXPECTED_VALID}")
    print("windows stay inside one story")
    return 0


if __name__ == "__main__":
    sys.exit(main())
