"""Refuse train recipes that cannot fit the process budget.

MLX ``set_memory_limit`` can lag; a 25M / 32-layer step already peaked at ~5 GB
on the 8 GB Air before ``check_memory`` ran. Estimate resident + VJP cache
before allocating weights.
"""

from __future__ import annotations

from typing import Mapping, Optional

from logging_config import logger
from model.cuda.env import PROCESS_BUDGET_BYTES, MemoryBudgetError, process_budget_label
from training.memory_controller import estimate_train_bytes


def estimate_train_step_bytes(
    *,
    n_params: int,
    batch_size: int,
    max_len: int,
    embedding_dim: int,
    num_heads: int,
    num_layers: int,
    gradient_checkpointing: bool = False,
    grad_accum: int = 1,
) -> int:
    """Conservative float32 bytes for one optimizer step (weights + VJP cache)."""
    return estimate_train_bytes(
        n_params=n_params,
        batch_size=batch_size,
        max_len=max_len,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        gradient_checkpointing=gradient_checkpointing,
        grad_accum=grad_accum,
    ).total


def n_params_from_gpt_config(gpt_config) -> int:
    """Parameter count from config (no weight allocation)."""
    from setup.model_config import estimate_vram_footprint

    return int(estimate_vram_footprint({
        "vocab_size": gpt_config.vocab_size,
        "max_len": gpt_config.max_len,
        "embedding_dim": gpt_config.embedding_dim,
        "num_heads": gpt_config.num_heads,
        "num_layers": gpt_config.num_layers,
        "tie_embeddings": gpt_config.tie_embeddings,
        "norm_type": gpt_config.norm_type,
        "pos_encoding": gpt_config.pos_encoding,
    })["total_params"])


def assert_gpt_train_fits_budget(gpt_config, hyperparams: Mapping, n_params: int) -> int:
    """Convenience wrapper around GPTConfig + training hyperparams."""
    return assert_train_fits_budget(
        n_params=int(n_params),
        batch_size=int(hyperparams.get("batch_size", 1)),
        max_len=int(gpt_config.max_len),
        embedding_dim=int(gpt_config.embedding_dim),
        num_heads=int(gpt_config.num_heads),
        num_layers=int(gpt_config.num_layers),
        gradient_checkpointing=bool(getattr(gpt_config, "gradient_checkpointing", False)),
        grad_accum=int(hyperparams.get("gradient_accumulation_steps", 1)),
    )


def assert_train_fits_budget(
    *,
    n_params: int,
    batch_size: int,
    max_len: int,
    embedding_dim: int,
    num_heads: int,
    num_layers: int,
    gradient_checkpointing: bool = False,
    grad_accum: int = 1,
    where: str = "train preflight",
    extra: Optional[Mapping[str, object]] = None,
) -> int:
    """Raise MemoryBudgetError if the estimated step cannot fit the process budget."""
    estimated = estimate_train_step_bytes(
        n_params=n_params,
        batch_size=batch_size,
        max_len=max_len,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        gradient_checkpointing=gradient_checkpointing,
        grad_accum=grad_accum,
    )
    budget_mb = PROCESS_BUDGET_BYTES / (1024 ** 2)
    est_mb = estimated / (1024 ** 2)
    logger.info(
        "Train memory preflight%s: estimate=%.0f MB budget=%.0f MB "
        "params=%s C=%s L=%s H=%s T=%s batch=%s accum=%s checkpoint=%s",
        f" ({where})" if where else "",
        est_mb, budget_mb, f"{n_params:,}", embedding_dim, num_layers,
        num_heads, max_len, batch_size, grad_accum, gradient_checkpointing,
    )
    if estimated > PROCESS_BUDGET_BYTES:
        hint = (
            "Disable --no-autoscale so the memory controller can shrink batch/context "
            "and enable checkpointing. Architecture C/L/H is never changed. "
            f"If weights+Adam alone exceed {process_budget_label()}, this recipe cannot run."
        )
        extras = ""
        if extra:
            extras = " " + " ".join(f"{k}={v}" for k, v in extra.items())
        raise MemoryBudgetError(
            f"process unified memory estimated over {process_budget_label()} budget ({where}): "
            f"estimate={est_mb:.0f} MB budget={budget_mb:.0f} MB "
            f"params={n_params:,} C={embedding_dim} L={num_layers} H={num_heads} "
            f"T={max_len} batch={batch_size} accum={grad_accum} "
            f"checkpoint={gradient_checkpointing}.{extras} {hint}"
        )
    return estimated
