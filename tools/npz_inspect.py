"""Host-side NumPy weight inspector (npzviewer backend).

Reads .npz / .npy / .npx with mmap only. Does not load the GPT onto Metal.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from paths import OUTPUT_CACHE, OUTPUT_CHECKPOINTS, PROJECT_ROOT

WEIGHT_SUFFIXES = (".npz", ".npy", ".npx")
_LAYER_RE = re.compile(r"^layer_(\d+)\.(.+)$")
_MAX_TABLE = 48
_MAX_HEAT = 72
_MAX_HIST_ELEMS = 400_000
_MAX_PARTNERS = 32

_ROLE_BY_LEAF: Dict[str, Dict[str, str]] = {
    "token_embedding": {
        "family": "embed",
        "title": "Token embedding",
        "blurb": "One row per vocabulary token, one column per residual channel (V, C).",
    },
    "position_embedding": {
        "family": "embed",
        "title": "Learned position embedding",
        "blurb": "Absolute position table (max_len, C). Absent when the model uses RoPE.",
    },
    "qkv_proj": {
        "family": "attn",
        "title": "Attention QKV projection",
        "blurb": "Packed query/key/value map C → 3C. Split into heads at runtime.",
    },
    "qkv_bias": {
        "family": "attn",
        "title": "Attention QKV bias",
        "blurb": "Bias on the packed QKV projection (length 3C).",
    },
    "attn_out_proj": {
        "family": "attn",
        "title": "Attention output projection",
        "blurb": "Mixes head outputs back into the residual stream (C, C).",
    },
    "attn_out_bias": {
        "family": "attn",
        "title": "Attention output bias",
        "blurb": "Bias after the attention output projection.",
    },
    "ln1_gamma": {
        "family": "norm",
        "title": "Pre-attention scale",
        "blurb": "LayerNorm/RMSNorm γ before attention. Healthy values stay near 1.",
    },
    "ln1_beta": {
        "family": "norm",
        "title": "Pre-attention shift",
        "blurb": "LayerNorm β before attention. Missing when the checkpoint is RMSNorm.",
    },
    "ln2_gamma": {
        "family": "norm",
        "title": "Pre-MLP scale",
        "blurb": "LayerNorm/RMSNorm γ before the MLP. Healthy values stay near 1.",
    },
    "ln2_beta": {
        "family": "norm",
        "title": "Pre-MLP shift",
        "blurb": "LayerNorm β before the MLP. Missing when the checkpoint is RMSNorm.",
    },
    "mlp_expand": {
        "family": "mlp",
        "title": "MLP expand",
        "blurb": "First MLP matrix, usually C → 4C.",
    },
    "mlp_expand_bias": {
        "family": "mlp",
        "title": "MLP expand bias",
        "blurb": "Bias on the expand projection.",
    },
    "mlp_contract": {
        "family": "mlp",
        "title": "MLP contract",
        "blurb": "Second MLP matrix, usually 4C → C. Residual-scale init is smaller here.",
    },
    "mlp_contract_bias": {
        "family": "mlp",
        "title": "MLP contract bias",
        "blurb": "Bias on the contract projection.",
    },
    "final_ln_gamma": {
        "family": "head",
        "title": "Final norm scale",
        "blurb": "γ on the last LayerNorm/RMSNorm before the language-model head.",
    },
    "final_ln_beta": {
        "family": "head",
        "title": "Final norm shift",
        "blurb": "LayerNorm β before the head. Missing when the checkpoint is RMSNorm.",
    },
    "lm_head": {
        "family": "head",
        "title": "Language-model head",
        "blurb": "Projects residual channels onto vocabulary logits (C, V). Tied checkpoints store token_embedding.T.",
    },
    "lm_head_bias": {
        "family": "head",
        "title": "Language-model bias",
        "blurb": "Per-token bias on the logits (length V).",
    },
}


@dataclass
class WeightArchive:
    """Open handle so mmap views stay valid."""

    path: Path
    format: str
    arrays: Dict[str, np.ndarray]
    _npz: Optional[np.lib.npyio.NpzFile] = None

    def close(self) -> None:
        if self._npz is not None:
            self._npz.close()
            self._npz = None
        self.arrays = {}

    def __enter__(self) -> "WeightArchive":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def is_weight_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES


def resolve_weight_path(raw: str, root: Path = PROJECT_ROOT) -> Path:
    """Resolve a user path and refuse anything outside root."""
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty path")
    root = Path(root).resolve()
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    path = candidate.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path outside project: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if not is_weight_file(path):
        raise ValueError(f"unsupported suffix: {path.suffix}")
    return path


def discover_weight_files(search_roots: Optional[List[Path]] = None) -> List[Dict[str, Any]]:
    roots = search_roots or [OUTPUT_CHECKPOINTS, OUTPUT_CACHE]
    found: List[Dict[str, Any]] = []
    seen = set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not is_weight_file(path):
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            try:
                rel = str(path.resolve().relative_to(PROJECT_ROOT))
            except ValueError:
                rel = str(path)
            found.append(
                {
                    "path": rel,
                    "name": path.name,
                    "parent": path.parent.name,
                    "bytes": int(path.stat().st_size),
                    "kind": "checkpoint" if path.name == "weights.npz" else "array",
                }
            )
    return found


def open_weight_file(path: Path) -> WeightArchive:
    path = Path(path)
    suffix = path.suffix.lower()
    fmt = suffix.lstrip(".") or "npy"
    if suffix == ".npy":
        arr = np.load(str(path), mmap_mode="r", allow_pickle=False)
        return WeightArchive(path=path, format=fmt, arrays={path.stem: np.asarray(arr)})
    try:
        handle = np.load(str(path), mmap_mode="r", allow_pickle=False)
    except ValueError:
        # Some .npx files are a single .npy with a different suffix.
        arr = np.load(str(path), mmap_mode="r", allow_pickle=False)
        if isinstance(arr, np.lib.npyio.NpzFile):
            raise
        return WeightArchive(path=path, format=fmt, arrays={path.stem: np.asarray(arr)})
    if isinstance(handle, np.lib.npyio.NpzFile):
        arrays = {name: handle[name] for name in handle.files}
        return WeightArchive(path=path, format=fmt, arrays=arrays, _npz=handle)
    return WeightArchive(path=path, format=fmt, arrays={path.stem: np.asarray(handle)})


def classify_key(name: str) -> Dict[str, Any]:
    layer = None
    leaf = name
    match = _LAYER_RE.match(name)
    if match:
        layer = int(match.group(1))
        leaf = match.group(2)
    role = dict(_ROLE_BY_LEAF.get(leaf) or _guess_role(leaf))
    role["key"] = name
    role["leaf"] = leaf
    role["layer"] = layer
    return role


def _guess_role(leaf: str) -> Dict[str, str]:
    low = leaf.lower()
    if "embed" in low:
        family, title = "embed", "Embedding"
    elif any(part in low for part in ("qkv", "attn", "attention", "wq", "wk", "wv")):
        family, title = "attn", "Attention tensor"
    elif any(part in low for part in ("mlp", "ffn", "w1", "w2", "w3")):
        family, title = "mlp", "MLP tensor"
    elif any(part in low for part in ("ln", "norm", "gamma", "beta", "scale")):
        family, title = "norm", "Normalization tensor"
    elif any(part in low for part in ("lm_head", "head", "unembed")):
        family, title = "head", "Output head"
    else:
        family, title = "other", "Array"
    return {
        "family": family,
        "title": title,
        "blurb": "Unrecognized name; stats and a slice are still available.",
    }


def sidecar_notes(path: Path) -> Dict[str, Any]:
    notes: Dict[str, Any] = {}
    parent = Path(path).parent
    config_path = parent / "config.json"
    if config_path.is_file():
        try:
            with open(config_path, encoding="utf-8") as fh:
                cfg = json.load(fh)
        except (OSError, json.JSONDecodeError):
            cfg = {}
        model = cfg.get("model") or {}
        notes["model_name"] = model.get("name")
        notes["description"] = (cfg.get("metadata") or {}).get("description") or model.get("description")
        notes["config"] = {
            "vocab_size": model.get("vocab_size"),
            "embedding_dim": model.get("embedding_dim"),
            "num_layers": model.get("num_layers"),
            "num_heads": model.get("num_heads"),
            "max_len": model.get("max_len"),
            "norm_type": model.get("norm_type"),
            "pos_encoding": model.get("pos_encoding"),
            "tie_embeddings": model.get("tie_embeddings"),
            "layer_strategy": model.get("layer_strategy"),
        }
    state_path = parent / "state.json"
    if state_path.is_file():
        try:
            with open(state_path, encoding="utf-8") as fh:
                state = json.load(fh)
            notes["step"] = state.get("step")
            notes["epoch"] = state.get("epoch")
        except (OSError, json.JSONDecodeError):
            pass
    return notes


def _numeric(arr: np.ndarray) -> bool:
    return bool(np.issubdtype(arr.dtype, np.number))


def _finite_view(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr)


def tensor_stats(arr: np.ndarray) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "shape": [int(s) for s in arr.shape],
        "ndim": int(arr.ndim),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
        "nbytes": int(getattr(arr, "nbytes", arr.size * getattr(arr.dtype, "itemsize", 0))),
    }
    if arr.size == 0 or not _numeric(arr):
        info.update(
            {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "rms": None,
                "l2": None,
                "nan": 0,
                "inf": 0,
                "abs_max": None,
                "near_zero": None,
            }
        )
        return info
    values = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(values)
    nan_count = int(np.isnan(values).sum())
    inf_count = int(np.isinf(values).sum())
    info["nan"] = nan_count
    info["inf"] = inf_count
    if not finite.any():
        info.update(
            {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "rms": None,
                "l2": None,
                "abs_max": None,
                "near_zero": 1.0,
            }
        )
        return info
    good = values[finite]
    info["min"] = float(good.min())
    info["max"] = float(good.max())
    info["mean"] = float(good.mean())
    info["std"] = float(good.std())
    info["rms"] = float(np.sqrt(np.mean(np.square(good))))
    info["l2"] = float(np.sqrt(np.sum(np.square(good))))
    info["abs_max"] = float(np.max(np.abs(good)))
    info["near_zero"] = float(np.mean(np.abs(good) < 1e-8))
    return info


def health_flags(name: str, stats: Dict[str, Any], role: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    if stats.get("nan"):
        flags.append("nan")
    if stats.get("inf"):
        flags.append("inf")
    if stats.get("size", 0) == 0:
        return flags
    if stats.get("std") is None:
        return flags
    std = float(stats["std"])
    abs_max = float(stats.get("abs_max") or 0.0)
    near_zero = float(stats.get("near_zero") or 0.0)
    family = role.get("family")
    leaf = role.get("leaf") or name
    if abs_max < 1e-12:
        flags.append("dead")
    elif near_zero > 0.99 and family not in ("norm",) and "bias" not in leaf:
        flags.append("sparse")
    if std == 0.0 and int(stats.get("size") or 0) > 1 and "dead" not in flags:
        flags.append("constant")
    if abs_max > 50 or std > 8:
        flags.append("exploding")
    if family == "norm" and "gamma" in leaf and stats.get("mean") is not None:
        if abs(float(stats["mean"]) - 1.0) > 0.75:
            flags.append("norm-drift")
    return flags


def infer_architecture(shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, Any]:
    layers = sorted(
        {int(match.group(1)) for key in shapes if (match := _LAYER_RE.match(key))}
    )
    token = shapes.get("token_embedding")
    lm_head = shapes.get("lm_head")
    qkv = next((shapes[k] for k in shapes if k.endswith(".qkv_proj")), None)
    expand = next((shapes[k] for k in shapes if k.endswith(".mlp_expand")), None)
    vocab = int(token[0]) if token and len(token) == 2 else (int(lm_head[1]) if lm_head and len(lm_head) == 2 else None)
    width = int(token[1]) if token and len(token) == 2 else (int(lm_head[0]) if lm_head and len(lm_head) == 2 else None)
    if width is None and qkv and len(qkv) == 2:
        width = int(qkv[0])
    has_beta = any(key.endswith("ln1_beta") or key == "final_ln_beta" for key in shapes)
    has_gamma = any(key.endswith("ln1_gamma") or key == "final_ln_gamma" for key in shapes)
    has_pos = "position_embedding" in shapes
    tied = False
    if token and lm_head and len(token) == 2 and len(lm_head) == 2:
        tied = tuple(token) == (lm_head[1], lm_head[0])
    mlp_mult = None
    if width and expand and len(expand) == 2 and expand[0] == width:
        mlp_mult = round(expand[1] / float(width), 3)
    return {
        "kind": "gpt" if token or layers else "arrays",
        "vocab_size": vocab,
        "embedding_dim": width,
        "num_layers": len(layers),
        "layer_ids": layers,
        "mlp_multiplier": mlp_mult,
        "pos_encoding": "learned" if has_pos else ("rope" if layers else None),
        "norm_type": ("layernorm" if has_beta else "rmsnorm") if has_gamma else None,
        "tie_embeddings": tied,
        "has_lm_head": "lm_head" in shapes,
        "has_token_embedding": "token_embedding" in shapes,
    }


def layer_summaries(archive: WeightArchive) -> List[Dict[str, Any]]:
    grouped: Dict[int, Dict[str, List[str]]] = {}
    for name in archive.arrays:
        match = _LAYER_RE.match(name)
        if not match:
            continue
        idx = int(match.group(1))
        role = classify_key(name)
        grouped.setdefault(idx, {"attn": [], "mlp": [], "norm": [], "other": []})
        family = role["family"] if role["family"] in ("attn", "mlp", "norm") else "other"
        grouped[idx][family].append(name)

    rows: List[Dict[str, Any]] = []
    for idx in sorted(grouped):
        buckets = grouped[idx]
        row = {"layer": idx}
        for family in ("attn", "mlp", "norm", "other"):
            row[f"{family}_rms"] = _group_rms(archive, buckets[family])
            row[f"{family}_params"] = int(sum(archive.arrays[n].size for n in buckets[family]))
        names = [n for fam in buckets.values() for n in fam]
        row["total_rms"] = _group_rms(archive, names)
        row["params"] = int(sum(archive.arrays[n].size for n in names))
        rows.append(row)
    return rows


def _group_rms(archive: WeightArchive, names: List[str]) -> Optional[float]:
    total = 0.0
    count = 0
    for name in names:
        arr = archive.arrays[name]
        if not _numeric(arr) or arr.size == 0:
            continue
        values = np.asarray(arr, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        total += float(np.sum(np.square(finite)))
        count += int(finite.size)
    if count == 0:
        return None
    return float(np.sqrt(total / count))


def histogram_payload(arr: np.ndarray, bins: int = 48) -> Dict[str, Any]:
    if arr.size == 0 or not _numeric(arr):
        return {"counts": [], "edges": [], "sampled": False}
    values = np.asarray(arr, dtype=np.float64).ravel()
    sampled = False
    if values.size > _MAX_HIST_ELEMS:
        step = max(1, values.size // _MAX_HIST_ELEMS)
        values = values[::step]
        sampled = True
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"counts": [], "edges": [], "sampled": sampled}
    counts, edges = np.histogram(finite, bins=int(bins))
    return {
        "counts": [int(c) for c in counts],
        "edges": [float(e) for e in edges],
        "sampled": sampled,
    }


def heatmap_payload(arr: np.ndarray, axis0: int = 0, max_side: int = _MAX_HEAT) -> Dict[str, Any]:
    plane = _plane(arr, axis0)
    if plane.size == 0 or not _numeric(plane):
        return {"values": [], "rows": 0, "cols": 0, "min": None, "max": None}
    grid = _block_mean(np.asarray(plane, dtype=np.float64), max_side)
    finite = grid[np.isfinite(grid)]
    lo = float(finite.min()) if finite.size else 0.0
    hi = float(finite.max()) if finite.size else 0.0
    return {
        "values": [[None if not np.isfinite(v) else float(v) for v in row] for row in grid],
        "rows": int(grid.shape[0]),
        "cols": int(grid.shape[1]),
        "min": lo,
        "max": hi,
        "source_shape": [int(s) for s in plane.shape],
    }


def plane_axes(key: str, plane: np.ndarray) -> Dict[str, Any]:
    """Row/column unit names for a 2-D weight plane."""
    if plane.ndim == 1:
        plane = plane.reshape(-1, 1)
    if plane.ndim == 0:
        return {"rows": 1, "cols": 1, "row_kind": "row", "col_kind": "col"}
    height, width = int(plane.shape[0]), int(plane.shape[1])
    leaf = classify_key(key).get("leaf") or ""
    row_kind, col_kind = "row", "col"
    if leaf == "token_embedding":
        row_kind, col_kind = "token", "residual"
    elif leaf == "position_embedding":
        row_kind, col_kind = "position", "residual"
    elif leaf == "mlp_expand":
        row_kind, col_kind = "residual", "mlp hidden"
    elif leaf == "mlp_contract":
        row_kind, col_kind = "mlp hidden", "residual"
    elif leaf == "qkv_proj":
        row_kind, col_kind = "residual", "qkv"
    elif leaf in ("attn_out_proj", "attn_out_bias"):
        row_kind, col_kind = "residual", "residual"
    elif leaf == "lm_head":
        row_kind, col_kind = "residual", "token"
    elif leaf.endswith("_gamma") or leaf.endswith("_beta"):
        row_kind, col_kind = leaf.split("_")[-1], "scale"
    return {
        "rows": height,
        "cols": width,
        "row_kind": row_kind,
        "col_kind": col_kind,
    }


def unit_label(kind: str, index: int, *, width: int = 0) -> str:
    if kind == "qkv" and width >= 3:
        c = width // 3
        if c and 0 <= index < 3 * c:
            part = "Q" if index < c else ("K" if index < 2 * c else "V")
            return f"{part}[{index % c}]"
    return f"{kind} {int(index)}"


def neuron_partners(
    path: Path,
    key: str,
    *,
    axis: str = "row",
    index: int = 0,
    k: int = 12,
    axis0: int = 0,
) -> Dict[str, Any]:
    """Top-|weight| partners of one row or column (mmap, no Metal)."""
    axis = str(axis or "row").strip().lower()
    if axis in ("0", "r"):
        axis = "row"
    if axis in ("1", "c"):
        axis = "col"
    if axis not in ("row", "col"):
        raise ValueError("axis must be row or col")
    k = max(1, min(int(k), _MAX_PARTNERS))
    with open_weight_file(path) as archive:
        if key not in archive.arrays:
            raise KeyError(key)
        arr = archive.arrays[key]
        role = classify_key(key)
        plane = _plane(arr, axis0)
        if plane.ndim == 0:
            raise ValueError("scalar has no neuron partners")
        if plane.ndim == 1:
            plane = plane.reshape(-1, 1)
        axes = plane_axes(key, plane)
        height, width = axes["rows"], axes["cols"]
        if axis == "row":
            idx = max(0, min(int(index), max(0, height - 1)))
            vec = np.asarray(plane[idx], dtype=np.float64).ravel()
            partner_kind = axes["col_kind"]
            partner_axis = "col"
            picked_kind = axes["row_kind"]
            partner_width = width
        else:
            idx = max(0, min(int(index), max(0, width - 1)))
            vec = np.asarray(plane[:, idx], dtype=np.float64).ravel()
            partner_kind = axes["row_kind"]
            partner_axis = "row"
            picked_kind = axes["col_kind"]
            partner_width = height
        order = np.argsort(-np.abs(np.where(np.isfinite(vec), vec, 0.0)))
        partners: List[Dict[str, Any]] = []
        for raw in order:
            if len(partners) >= k:
                break
            weight = float(vec[int(raw)])
            if not np.isfinite(weight):
                continue
            partners.append(
                {
                    "index": int(raw),
                    "axis": partner_axis,
                    "label": unit_label(partner_kind, int(raw), width=int(partner_width)),
                    "weight": weight,
                    "abs": abs(weight),
                }
            )
        return {
            "key": key,
            **role,
            "axes": axes,
            "axis": axis,
            "index": idx,
            "label": unit_label(picked_kind, idx, width=int(width if axis == "col" else height)),
            "kind": picked_kind,
            "k": k,
            "partners": partners,
        }


def table_payload(
    arr: np.ndarray,
    row0: int = 0,
    col0: int = 0,
    rows: int = 16,
    cols: int = 8,
    axis0: int = 0,
) -> Dict[str, Any]:
    plane = _plane(arr, axis0)
    if plane.ndim == 0:
        return {
            "values": [[_json_num(plane)]],
            "row0": 0,
            "col0": 0,
            "rows": 1,
            "cols": 1,
            "shape": [],
        }
    if plane.ndim == 1:
        plane = plane.reshape(-1, 1)
    height, width = int(plane.shape[0]), int(plane.shape[1])
    r0 = max(0, min(int(row0), max(0, height - 1)))
    c0 = max(0, min(int(col0), max(0, width - 1)))
    rh = max(1, min(int(rows), _MAX_TABLE, height - r0))
    cw = max(1, min(int(cols), _MAX_TABLE, width - c0))
    sl = np.asarray(plane[r0 : r0 + rh, c0 : c0 + cw])
    return {
        "values": [[_json_num(v) for v in row] for row in sl],
        "row0": r0,
        "col0": c0,
        "rows": rh,
        "cols": cw,
        "shape": [height, width],
    }


def _plane(arr: np.ndarray, axis0: int) -> np.ndarray:
    view = _finite_view(arr)
    if view.ndim <= 2:
        return view
    idx = int(axis0)
    limit = int(view.shape[0])
    if limit == 0:
        return view.reshape(0, -1)
    idx = max(0, min(idx, limit - 1))
    rest = view[idx]
    if rest.ndim > 2:
        return rest.reshape(rest.shape[0], -1)
    return rest


def _block_mean(plane: np.ndarray, max_side: int) -> np.ndarray:
    if plane.ndim == 1:
        plane = plane.reshape(-1, 1)
    h, w = plane.shape[:2]
    row_step = max(1, int(np.ceil(h / max_side)))
    col_step = max(1, int(np.ceil(w / max_side)))
    out_h = int(np.ceil(h / row_step))
    out_w = int(np.ceil(w / col_step))
    grid = np.full((out_h, out_w), np.nan, dtype=np.float64)
    for i in range(out_h):
        r0, r1 = i * row_step, min(h, (i + 1) * row_step)
        for j in range(out_w):
            c0, c1 = j * col_step, min(w, (j + 1) * col_step)
            block = plane[r0:r1, c0:c1]
            finite = block[np.isfinite(block)]
            if finite.size:
                grid[i, j] = float(finite.mean())
    return grid


def _json_num(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return str(value)


def archive_health(tensors: List[Dict[str, Any]], architecture: Dict[str, Any]) -> Dict[str, Any]:
    flags = sorted({flag for row in tensors for flag in row.get("flags") or []})
    dead_layers = []
    by_layer: Dict[int, List[str]] = {}
    for row in tensors:
        layer = row.get("layer")
        if layer is None:
            continue
        by_layer.setdefault(int(layer), [])
        if "dead" in (row.get("flags") or []):
            by_layer[int(layer)].append(row["key"])
    for layer, keys in by_layer.items():
        layer_keys = [row["key"] for row in tensors if row.get("layer") == layer]
        if layer_keys and len(keys) == len(layer_keys):
            dead_layers.append(layer)
    return {
        "flags": flags,
        "dead_layers": dead_layers,
        "ok": not any(flag in flags for flag in ("nan", "inf", "dead", "exploding")),
        "kind": architecture.get("kind"),
    }


def manifest(path: Path) -> Dict[str, Any]:
    path = Path(path)
    with open_weight_file(path) as archive:
        shapes = {name: tuple(int(s) for s in arr.shape) for name, arr in archive.arrays.items()}
        architecture = infer_architecture(shapes)
        tensors: List[Dict[str, Any]] = []
        total_params = 0
        total_bytes = 0
        for name in sorted(archive.arrays, key=_tensor_sort):
            arr = archive.arrays[name]
            role = classify_key(name)
            stats = tensor_stats(arr)
            flags = health_flags(name, stats, role)
            total_params += int(stats["size"])
            total_bytes += int(stats["nbytes"])
            tensors.append(
                {
                    "key": name,
                    "family": role["family"],
                    "title": role["title"],
                    "blurb": role["blurb"],
                    "layer": role["layer"],
                    "leaf": role["leaf"],
                    "flags": flags,
                    **stats,
                }
            )
        layers = layer_summaries(archive)
        rel = str(path)
        try:
            rel = str(path.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            pass
        sidecar = sidecar_notes(path)
        return {
            "path": rel,
            "name": path.name,
            "format": archive.format,
            "bytes": int(path.stat().st_size),
            "params": total_params,
            "tensor_bytes": total_bytes,
            "architecture": architecture,
            "sidecar": sidecar,
            "mismatches": sidecar_mismatches(architecture, sidecar),
            "layers": layers,
            "tensors": tensors,
            "health": archive_health(tensors, architecture),
        }


def sidecar_mismatches(architecture: Dict[str, Any], sidecar: Dict[str, Any]) -> List[str]:
    cfg = sidecar.get("config") or {}
    labels = (
        ("vocab_size", "V"),
        ("embedding_dim", "C"),
        ("num_layers", "L"),
        ("norm_type", "norm"),
        ("pos_encoding", "position"),
        ("tie_embeddings", "tied"),
    )
    found: List[str] = []
    for key, label in labels:
        expected = cfg.get(key)
        got = architecture.get(key)
        if expected is None or got is None or expected == got:
            continue
        found.append(f"{label}: weights {got} vs config {expected}")
    return found


def tensor_detail(
    path: Path,
    key: str,
    row0: int = 0,
    col0: int = 0,
    rows: int = 16,
    cols: int = 8,
    axis0: int = 0,
) -> Dict[str, Any]:
    with open_weight_file(path) as archive:
        if key not in archive.arrays:
            raise KeyError(key)
        arr = archive.arrays[key]
        role = classify_key(key)
        stats = tensor_stats(arr)
        plane = _plane(arr, axis0)
        return {
            "key": key,
            **role,
            "stats": stats,
            "flags": health_flags(key, stats, role),
            "axes": plane_axes(key, plane),
            "histogram": histogram_payload(arr),
            "heatmap": heatmap_payload(arr, axis0=axis0),
            "table": table_payload(arr, row0=row0, col0=col0, rows=rows, cols=cols, axis0=axis0),
            "axis0": int(axis0),
            "axis0_max": int(arr.shape[0] - 1) if arr.ndim >= 3 and arr.shape[0] else 0,
        }


def _tensor_sort(name: str) -> Tuple[int, int, str]:
    role = classify_key(name)
    family_order = {"embed": 0, "attn": 1, "mlp": 2, "norm": 3, "head": 4, "other": 5}
    layer = role["layer"] if role["layer"] is not None else -1
    return (layer, family_order.get(role["family"], 9), name)
