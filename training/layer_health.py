"""Per-layer gradient norms from the explicit-VJP grads dict."""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional

import numpy as np

_ATTN_PARTS = ("qkv", "attn_out", "ln1")
_MLP_PARTS = ("mlp", "ln2")


def _is_device(g) -> bool:
    return hasattr(g, "gpudata") or type(g).__name__ in ("DeviceArray", "GPUArray")


def _subset_norm(grads: Mapping[str, object], keys: List[str]) -> float:
    subset = {k: grads[k] for k in keys if k in grads and grads[k] is not None}
    if not subset:
        return 0.0
    if all(_is_device(g) for g in subset.values()):
        from model.cuda import ops as cuda_ops

        return float(np.sqrt(cuda_ops.grad_global_norm_sq(subset)))
    total = 0.0
    for g in subset.values():
        arr = np.asarray(g, dtype=np.float64)
        total += float(np.sum(arr * arr))
    return float(np.sqrt(total))


def _keys_with(names: List[str], parts: tuple) -> List[str]:
    return [k for k in names if any(p in k for p in parts)]


def summarize_layer_grad_norms(
    grads: Mapping[str, object],
    num_layers: int,
) -> Dict:
    """L2 norms per transformer block (after clip, before/with the update)."""
    names = [k for k, g in grads.items() if g is not None]
    layers: List[Dict] = []
    for i in range(int(num_layers)):
        prefix = f"layer_{i}."
        keys = [k for k in names if k.startswith(prefix)]
        layers.append({
            "layer": i,
            "total": _subset_norm(grads, keys),
            "attn": _subset_norm(grads, _keys_with(keys, _ATTN_PARTS)),
            "mlp": _subset_norm(grads, _keys_with(keys, _MLP_PARTS)),
        })

    totals = [row["total"] for row in layers]
    positive = [t for t in totals if t > 1e-12]
    ratio = (max(positive) / min(positive)) if len(positive) >= 2 else 1.0
    early = totals[0] if totals else 0.0
    late = totals[-1] if totals else 0.0
    dead = [i for i, t in enumerate(totals) if t <= 1e-10]
    healthy = bool(positive) and ratio <= 20.0 and not dead

    embed_keys = [k for k in names if k in ("token_embedding", "position_embedding")]
    final_keys = [k for k in names if k in ("final_ln_gamma", "final_ln_beta", "lm_head", "lm_head_bias")]
    return {
        "layers": layers,
        "totals": totals,
        "ratio": float(ratio),
        "early": float(early),
        "late": float(late),
        "embed": _subset_norm(grads, embed_keys),
        "final": _subset_norm(grads, final_keys),
        "dead": dead,
        "healthy": healthy,
    }


def format_layer_grad_line(report: Dict, *, step: Optional[int] = None) -> str:
    bits = []
    if step is not None:
        bits.append(f"step={step}")
    parts = []
    for row in report["layers"]:
        parts.append(f"L{row['layer']}={row['total']:.4g}(a={row['attn']:.3g}/m={row['mlp']:.3g})")
    bits.append(" ".join(parts))
    bits.append(f"embed={report['embed']:.4g}")
    bits.append(f"final={report['final']:.4g}")
    bits.append(f"early={report['early']:.4g}")
    bits.append(f"late={report['late']:.4g}")
    bits.append(f"ratio={report['ratio']:.2f}")
    if report["dead"]:
        bits.append("dead=" + ",".join(str(i) for i in report["dead"]))
    if not report["healthy"]:
        bits.append("UNHEALTHY")
    return "[layer_grads] " + " ".join(bits)
