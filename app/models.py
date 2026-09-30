"""Discover checkpoint dirs for the App.py model selector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from paths import DATA_DIR, OUTPUT_CHECKPOINTS, checkpoint_weights_relpath, relative_to_project

DEFAULT_FACTS = DATA_DIR / "chat_facts.jsonl"

_WEIGHT_NAME = "weights.npz"
_CONFIG_NAME = "config.json"


def infer_facts(checkpoint: Union[str, Path]) -> str:
    """Prefer the mix recorded in config.json, then data/<name>.jsonl."""
    ckpt = Path(checkpoint)
    if ckpt.is_file():
        ckpt = ckpt.parent
    cfg_path = ckpt / _CONFIG_NAME
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cfg = {}
        recorded = str((cfg.get("dataset") or {}).get("path") or "").strip()
        if recorded:
            rec = Path(recorded)
            if rec.is_file():
                return relative_to_project(rec)
    named = DATA_DIR / f"{ckpt.name}.jsonl"
    if named.is_file():
        return relative_to_project(named)
    return relative_to_project(DEFAULT_FACTS)


def _model_name(cfg: dict, fallback: str) -> str:
    model = cfg.get("model") if isinstance(cfg, dict) else None
    if isinstance(model, dict):
        name = str(model.get("name") or "").strip()
        if name:
            return name
    return fallback


def describe_checkpoint(path: Union[str, Path]) -> Optional[Dict[str, object]]:
    ckpt = Path(path)
    if ckpt.is_file():
        ckpt = ckpt.parent
    weights = ckpt / _WEIGHT_NAME
    if not weights.is_file():
        return None
    cfg: dict = {}
    cfg_path = ckpt / _CONFIG_NAME
    if cfg_path.is_file():
        try:
            loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                cfg = loaded
        except (OSError, json.JSONDecodeError):
            cfg = {}
    rel = relative_to_project(ckpt)
    return {
        "id": rel,
        "name": ckpt.name,
        "title": _model_name(cfg, ckpt.name),
        "checkpoint": rel,
        "weights": checkpoint_weights_relpath(ckpt),
        "facts": infer_facts(ckpt),
        "bytes": int(weights.stat().st_size),
        "chat": "chat" in ckpt.name.casefold() or "chat" in _model_name(cfg, "").casefold(),
    }


def list_models(root: Union[str, Path] = OUTPUT_CHECKPOINTS) -> List[Dict[str, object]]:
    base = Path(root)
    if not base.is_dir():
        return []
    found: List[Dict[str, object]] = []
    seen = set()
    for weights in sorted(base.rglob(_WEIGHT_NAME)):
        rec = describe_checkpoint(weights.parent)
        if rec is None or rec["id"] in seen:
            continue
        seen.add(rec["id"])
        found.append(rec)
    found.sort(key=lambda row: str(row["name"]))
    return found


def resolve_model(
    checkpoint: Union[str, Path],
    *,
    root: Union[str, Path] = OUTPUT_CHECKPOINTS,
) -> Optional[Dict[str, object]]:
    rec = describe_checkpoint(checkpoint)
    if rec is not None:
        return rec
    name = Path(checkpoint).name
    for row in list_models(root):
        if row["name"] == name or row["id"] == name or row["checkpoint"] == str(checkpoint):
            return row
    return None
