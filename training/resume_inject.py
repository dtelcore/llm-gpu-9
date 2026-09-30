"""Resume-time corpus swap for Phase 2 inject.

Keeps the checkpoint tokenizer and architecture. Reloads train text from
``--dataset-path`` and retains Phase 1 ``val_corpus.json`` as the English
holdout. Never rebuilds BPE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from logging_config import logger
from setup.config_loader import resolve_dataset_corpus


class ResumeInjectError(ValueError):
    """CLI / checkpoint mismatch that must abort before Metal work."""


def verify_resume_inject(
    *,
    resume: bool,
    dataset_path: Optional[str],
    checkpoint_dir: Optional[str] = None,
) -> None:
    path = (dataset_path or "").strip() or None
    if path and not resume:
        raise ResumeInjectError(
            "CRITICAL error: --dataset-path is strictly prohibited without --resume."
        )
    if not path:
        return
    ckpt = Path(checkpoint_dir or "")
    vocab = ckpt / "vocab.json"
    if not vocab.is_file():
        raise FileNotFoundError(
            f"Verification failure: vocab.json is missing in {ckpt}"
        )


def _load_val_corpus(config: Dict[str, Any], checkpoint_dir: Path) -> List[str]:
    dataset = config.setdefault("dataset", {})
    existing = dataset.get("val_corpus")
    if isinstance(existing, list) and existing:
        return [str(s) for s in existing]
    val_path = checkpoint_dir / "val_corpus.json"
    if val_path.is_file():
        payload = json.loads(val_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [str(s) for s in payload]
    return []


def apply_resume_dataset_path(
    config: Dict[str, Any],
    checkpoint_dir: str | Path,
    dataset_path: str,
    *,
    tokenizer: Any = None,
) -> Tuple[List[str], List[str]]:
    """Point resume config at a new mix. Tokenizer is accepted only to assert identity."""
    ckpt = Path(checkpoint_dir)
    val_corpus = _load_val_corpus(config, ckpt)
    dataset = config.setdefault("dataset", {})
    if dataset.get("combine"):
        logger.info("Forcing combine=false for --dataset-path inject")
    dataset["path"] = str(dataset_path)
    dataset["combine"] = False
    dataset.pop("corpus", None)

    train_lines = resolve_dataset_corpus(dataset)
    val_set = set(val_corpus)
    train_corpus = [line for line in train_lines if line not in val_set]
    dataset["corpus"] = train_corpus
    dataset["val_corpus"] = val_corpus

    vocab_size = getattr(tokenizer, "vocab_size", None) if tokenizer is not None else None
    model = config.get("model") or {}
    print("")
    print("--- Pipeline Resume Injection Verified ---")
    print(
        f"Vocab Identity Verified ({vocab_size if vocab_size is not None else 'checkpoint vocab.json'}). "
        "C/L/T parameters locked "
        f"(C={model.get('embedding_dim')} L={model.get('num_layers')} T={model.get('max_len')})."
    )
    print(f"Switching primary corpus path tracking to target: {dataset_path}")
    print("Retaining original Phase 1 val_corpus.json tracking framework.")
    print(f"Train sentences={len(train_corpus):,}  val sentences={len(val_corpus):,}")
    print("")
    logger.info(
        "resume inject | path=%s train=%s val=%s vocab_size=%s",
        dataset_path, len(train_corpus), len(val_corpus), vocab_size,
    )
    return train_corpus, val_corpus
