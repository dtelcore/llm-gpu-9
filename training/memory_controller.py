"""Size batch, context, and activations to the hardcoded process cap.

Architecture (C, L, H, V) is never changed. The live path keeps host NumPy
weights, CUDA mirrors, Adam m/v, grads, and explicit VJP caches in the same
process; ``PROCESS_BUDGET_BYTES`` in ``model.cuda.env`` is the limit (5 GB).
This module does not raise that cap.

Knobs, cheapest quality impact first:
  1. Realize the MLX graph after each layer (drop lazy intermediates)
  2. Gradient checkpointing (recompute attn/MLP in backward)
  3. FP16 storage for kept activations (compute stays FP32)
  4. Sequential layer streaming (one block + Adam on Metal; idle m/v on host)
  5. Smaller micro-batch, more grad-accum (same tokens/step)
  6. Shorter context T (last resort)

If always-resident weights + one block still overflow the usable Metal budget,
refuse. ``--no-layer-stream`` keeps the pre-stream refuse. ``--layer-stream``
forces streaming (L=6 bring-up). Residual checkpoints ``h_in`` stay on device.

Layer streaming: ``layer_strategy='stream'`` pages idle ``device_weights[layer_i]``
and Adam m/v so peak Metal is O(1) in depth. Generate prefill/decode uses the
same load/unload. This module does not spawn workers or shard layers.

"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from logging_config import logger
from model.cuda.env import PROCESS_BUDGET_BYTES, MemoryBudgetError, process_budget_label

_F32 = 4
_DEFAULT_HEADROOM = 0.15
_MIN_BATCH = 1
_MIN_CONTEXT = 32
_MAX_ACCUM = 64
_ACT_SAFETY = 1.20


def process_budget_bytes() -> int:
    """Hardcoded process cap. Not configurable here on purpose."""
    return int(PROCESS_BUDGET_BYTES)


def usable_bytes(headroom: float = _DEFAULT_HEADROOM) -> int:
    """Plan against this many bytes so compile/scratch still fit under the process cap."""
    h = min(0.45, max(0.0, float(headroom)))
    return int(process_budget_bytes() * (1.0 - h))


def _align_context(t: int) -> int:
    t = max(_MIN_CONTEXT, int(t))
    if t <= _MIN_CONTEXT:
        return _MIN_CONTEXT
    # Keep even lengths; prefer multiples of 32 when already large.
    if t >= 64:
        return max(_MIN_CONTEXT, (t // 32) * 32)
    return t if t % 2 == 0 else t - 1


@dataclass
class TrainEstimate:
    param_host_bytes: int
    param_device_bytes: int
    optimizer_bytes: int
    grad_bytes: int
    activation_bytes: int
    logits_bytes: int
    workspace_bytes: int
    total: int

    def as_mb(self) -> Dict[str, float]:
        return {k: round(v / (1024 ** 2), 1) for k, v in asdict(self).items()}


@dataclass
class MemoryPlan:
    mode: str
    batch_size: int
    max_len: int
    grad_accum: int
    gradient_checkpointing: bool
    fp16_activations: bool
    eval_per_layer: bool
    estimated_bytes: int
    usable_bytes: int
    budget_bytes: int
    requested_batch: int
    requested_max_len: int
    requested_accum: int
    requested_checkpoint: bool
    layer_strategy: str = "resident"
    actions: List[str] = field(default_factory=list)
    breakdown_mb: Dict[str, float] = field(default_factory=dict)
    max_new_tokens: Optional[int] = None
    fits: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["budget_gb"] = round(self.budget_bytes / (1024 ** 3), 3)
        return d

    def summary_line(self) -> str:
        changed = "; ".join(self.actions) if self.actions else "no changes (already fits)"
        return (
            f"budget={self.budget_bytes / (1024 ** 2):.0f}MB "
            f"usable={self.usable_bytes / (1024 ** 2):.0f}MB "
            f"estimate={self.estimated_bytes / (1024 ** 2):.0f}MB "
            f"B={self.batch_size} accum={self.grad_accum} T={self.max_len} "
            f"ckpt={int(self.gradient_checkpointing)} fp16={int(self.fp16_activations)} "
            f"eval_per_layer={int(self.eval_per_layer)} "
            f"layers={self.layer_strategy} | {changed}"
        )


def estimate_train_bytes(
    *,
    n_params: int,
    batch_size: int,
    max_len: int,
    embedding_dim: int,
    num_heads: int,
    num_layers: int,
    vocab_size: int = 4112,
    gradient_checkpointing: bool = False,
    fp16_activations: bool = False,
    eval_per_layer: bool = False,
    grad_accum: int = 1,
    layer_strategy: str = "resident",
) -> TrainEstimate:
    """Conservative float32 bytes for one optimizer step on the live MLX path."""
    B = max(1, int(batch_size))
    T = max(1, int(max_len))
    C = max(1, int(embedding_dim))
    H = max(1, int(num_heads))
    L = max(1, int(num_layers))
    V = max(1, int(vocab_size))
    accum = max(1, int(grad_accum))
    n_params = max(0, int(n_params))
    stream = str(layer_strategy).strip().lower() == "stream"

    param_b = n_params * _F32
    param_host = param_b
    param_device = param_b
    optimizer = 2 * param_b  # Adam m, v
    grads = param_b
    if accum > 1:
        grads += param_b  # accumulation buffer

    btc = B * T * C * _F32
    attn = B * H * T * T * _F32
    qkv = 3 * btc
    mlp = B * T * (4 * C) * _F32
    # RMSNorm xhat/inv/out for ln1+ln2, plus residual stream.
    ln_stored = 6 * btc
    kv_stored = 2 * btc
    stored_per_layer = ln_stored + kv_stored
    working = attn + qkv + 2 * mlp

    if stream:
        # Host holds all weights + idle Adam; Metal sees embed + one block.
        resident_n = min(n_params, V * C + 2 * C + V)
        block_n = max(1, (n_params - resident_n) // L)
        param_device = (resident_n + block_n) * _F32
        optimizer = 2 * param_b  # idle layer m/v stay on host
        grads = param_device
        if accum > 1:
            grads += param_b  # host accum of the full grad dict
        # All residual checkpoints on device + one live attn/MLP working set.
        stored = L * btc + stored_per_layer
        live_working = working
        activations = int(_ACT_SAFETY * (stored + live_working))
    elif gradient_checkpointing:
        stored = L * stored_per_layer
        live_working = working if eval_per_layer else L * working
        activations = int(_ACT_SAFETY * (stored + live_working))
        if fp16_activations:
            activations = int(activations * 0.72)
    elif eval_per_layer:
        stored = L * (stored_per_layer + working)
        live_working = working
        activations = int(_ACT_SAFETY * (stored + live_working))
        if fp16_activations:
            activations = int(activations * 0.72)
    else:
        stored = L * (stored_per_layer + working)
        live_working = L * working
        activations = int(_ACT_SAFETY * (stored + live_working))
        if fp16_activations:
            activations = int(activations * 0.72)

    logits = B * T * V * _F32
    workspace = int(1.35 * max(attn, mlp, btc) * 3)
    total = (
        param_host + param_device + optimizer + grads
        + activations + logits + workspace
    )
    return TrainEstimate(
        param_host_bytes=param_host,
        param_device_bytes=param_device,
        optimizer_bytes=optimizer,
        grad_bytes=grads,
        activation_bytes=activations,
        logits_bytes=logits,
        workspace_bytes=workspace,
        total=int(total),
    )


def estimate_generate_bytes(
    *,
    n_params: int,
    max_len: int,
    embedding_dim: int,
    num_heads: int,
    num_layers: int,
    vocab_size: int = 4112,
    prompt_len: int = 1,
    max_new_tokens: int = 80,
    use_kv_cache: bool = True,
    eval_per_layer: bool = True,
    layer_strategy: str = "resident",
) -> int:
    """Prefill + KV arenas + one decode step. No Adam."""
    T = max(1, min(int(max_len), int(prompt_len) + max(1, int(max_new_tokens))))
    est = estimate_train_bytes(
        n_params=n_params,
        batch_size=1,
        max_len=T,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        vocab_size=vocab_size,
        gradient_checkpointing=True,
        fp16_activations=False,
        eval_per_layer=eval_per_layer,
        grad_accum=1,
        layer_strategy=layer_strategy,
    )
    # Generate does not keep Adam / grad / host-train accum.
    # Do not subtract optimizer_bytes: they are not in this sum. (Resident
    # accidentally cancelled param_host+param_device == 2P against optimizer
    # 2P; stream has param_device << P and the subtraction went negative.)
    without_train = (
        est.param_host_bytes + est.param_device_bytes
        + est.activation_bytes + est.logits_bytes + est.workspace_bytes
    )
    L = max(1, int(num_layers))
    C = max(1, int(embedding_dim))
    if use_kv_cache:
        kv_arenas = L * 2 * int(max_len) * C * _F32
    else:
        kv_arenas = 0
    return int(without_train + kv_arenas)


def _plan_fits(est: TrainEstimate, usable: int) -> bool:
    return int(est.total) <= int(usable)


def plan_train(
    *,
    n_params: int,
    batch_size: int,
    max_len: int,
    embedding_dim: int,
    num_heads: int,
    num_layers: int,
    vocab_size: int,
    grad_accum: int = 1,
    gradient_checkpointing: bool = False,
    autoscale: bool = True,
    headroom: float = _DEFAULT_HEADROOM,
    allow_checkpoint: bool = True,
    allow_fp16: bool = True,
    layer_strategy: str = "resident",
    allow_stream: bool = True,
) -> MemoryPlan:
    """Return a plan that fits the process cap, or ``fits=False`` if impossible."""
    usable = usable_bytes(headroom)
    budget = process_budget_bytes()
    requested_batch = max(1, int(batch_size))
    requested_t = max(1, int(max_len))
    requested_accum = max(1, int(grad_accum))
    requested_ckpt = bool(gradient_checkpointing)

    B = requested_batch
    T = requested_t
    accum = requested_accum
    ckpt = requested_ckpt
    fp16 = False
    stream = str(layer_strategy).strip().lower() == "stream"
    if not allow_stream:
        stream = False
    eval_pl = bool(stream)
    actions: List[str] = []
    effective = B * accum

    def _strategy() -> str:
        return "stream" if stream else "resident"

    def _est() -> TrainEstimate:
        return estimate_train_bytes(
            n_params=n_params,
            batch_size=B,
            max_len=T,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            vocab_size=vocab_size,
            gradient_checkpointing=ckpt,
            fp16_activations=fp16,
            eval_per_layer=eval_pl,
            grad_accum=accum,
            layer_strategy=_strategy(),
        )

    est = _est()
    if _plan_fits(est, usable) or not autoscale:
        return MemoryPlan(
            mode="train",
            batch_size=B,
            max_len=T,
            grad_accum=accum,
            gradient_checkpointing=ckpt,
            fp16_activations=fp16,
            eval_per_layer=eval_pl,
            estimated_bytes=est.total,
            usable_bytes=usable,
            budget_bytes=budget,
            requested_batch=requested_batch,
            requested_max_len=requested_t,
            requested_accum=requested_accum,
            requested_checkpoint=requested_ckpt,
            layer_strategy=_strategy(),
            actions=actions if autoscale else (["autoscale disabled"] if not _plan_fits(est, usable) else []),
            breakdown_mb=est.as_mb(),
            fits=_plan_fits(est, usable),
        )

    # 1. Materialize per layer so the lazy graph cannot retain all L blocks.
    eval_pl = True
    actions.append("eval_per_layer")
    est = _est()
    if _plan_fits(est, usable):
        return _finish_plan(
            B, T, accum, ckpt, fp16, eval_pl, est, usable, budget,
            requested_batch, requested_t, requested_accum, requested_ckpt, actions,
            layer_strategy=_strategy(),
        )

    # 2. Checkpoint attn/MLP activations.
    if allow_checkpoint and not ckpt:
        ckpt = True
        actions.append("gradient_checkpointing")
        est = _est()
        if _plan_fits(est, usable):
            return _finish_plan(
                B, T, accum, ckpt, fp16, eval_pl, est, usable, budget,
                requested_batch, requested_t, requested_accum, requested_ckpt, actions,
                layer_strategy=_strategy(),
            )

    # 3. FP16 kept caches.
    if allow_fp16 and not fp16:
        fp16 = True
        actions.append("fp16_activations")
        est = _est()
        if _plan_fits(est, usable):
            return _finish_plan(
                B, T, accum, ckpt, fp16, eval_pl, est, usable, budget,
                requested_batch, requested_t, requested_accum, requested_ckpt, actions,
                layer_strategy=_strategy(),
            )

    # 4. Halve micro-batch; raise accum to keep tokens/optimizer-step.
    while B > _MIN_BATCH:
        B = max(_MIN_BATCH, B // 2)
        accum = min(_MAX_ACCUM, max(accum, (effective + B - 1) // B))
        actions.append(f"batch {requested_batch}->{B} (accum {requested_accum}->{accum})")
        est = _est()
        if _plan_fits(est, usable):
            return _finish_plan(
                B, T, accum, ckpt, fp16, eval_pl, est, usable, budget,
                requested_batch, requested_t, requested_accum, requested_ckpt, actions,
                layer_strategy=_strategy(),
            )

    # 5. Sequential layer streaming before shrinking context.
    # Enable even if this B/T is still over so later T shrinks use the stream estimate.
    if allow_stream and not stream:
        stream = True
        eval_pl = True
        actions.append("layer_strategy=stream")
        if "eval_per_layer" not in actions:
            actions.append("eval_per_layer")
        est = _est()
        if _plan_fits(est, usable):
            return _finish_plan(
                B, T, accum, ckpt, fp16, eval_pl, est, usable, budget,
                requested_batch, requested_t, requested_accum, requested_ckpt, actions,
                layer_strategy=_strategy(),
            )

    # 6. Shrink context. Effective batch already at B=1.
    while T > _MIN_CONTEXT:
        nxt = _align_context(T // 2)
        if nxt >= T:
            nxt = max(_MIN_CONTEXT, T - 32)
        if nxt >= T:
            break
        T = nxt
        actions.append(f"max_len {requested_t}->{T}")
        est = _est()
        if _plan_fits(est, usable):
            return _finish_plan(
                B, T, accum, ckpt, fp16, eval_pl, est, usable, budget,
                requested_batch, requested_t, requested_accum, requested_ckpt, actions,
                layer_strategy=_strategy(),
            )

    est = _est()
    return MemoryPlan(
        mode="train",
        batch_size=B,
        max_len=T,
        grad_accum=accum,
        gradient_checkpointing=ckpt,
        fp16_activations=fp16,
        eval_per_layer=eval_pl,
        estimated_bytes=est.total,
        usable_bytes=usable,
        budget_bytes=budget,
        requested_batch=requested_batch,
        requested_max_len=requested_t,
        requested_accum=requested_accum,
        requested_checkpoint=requested_ckpt,
        layer_strategy=_strategy(),
        actions=actions,
        breakdown_mb=est.as_mb(),
        fits=_plan_fits(est, usable),
    )


def _finish_plan(
    B, T, accum, ckpt, fp16, eval_pl, est, usable, budget,
    requested_batch, requested_t, requested_accum, requested_ckpt, actions,
    layer_strategy: str = "resident",
) -> MemoryPlan:
    return MemoryPlan(
        mode="train",
        batch_size=B,
        max_len=T,
        grad_accum=accum,
        gradient_checkpointing=ckpt,
        fp16_activations=fp16,
        eval_per_layer=eval_pl,
        estimated_bytes=est.total,
        usable_bytes=usable,
        budget_bytes=budget,
        requested_batch=requested_batch,
        requested_max_len=requested_t,
        requested_accum=requested_accum,
        requested_checkpoint=requested_ckpt,
        layer_strategy=layer_strategy,
        actions=actions,
        breakdown_mb=est.as_mb(),
        fits=True,
    )


def plan_generate(
    *,
    n_params: int,
    max_len: int,
    embedding_dim: int,
    num_heads: int,
    num_layers: int,
    vocab_size: int,
    prompt_len: int,
    max_new_tokens: int,
    use_kv_cache: bool = True,
    headroom: float = _DEFAULT_HEADROOM,
    layer_strategy: str = "resident",
) -> MemoryPlan:
    usable = usable_bytes(headroom)
    budget = process_budget_bytes()
    T = max(1, int(max_len))
    new_toks = max(1, int(max_new_tokens))
    prompt_len = max(0, int(prompt_len))
    actions: List[str] = []
    eval_pl = True

    room = max(1, T - min(prompt_len, T))
    if new_toks > room:
        actions.append(f"max_new_tokens {new_toks}->{room} (context T={T})")
        new_toks = room

    def _bytes(nt: int) -> int:
        return estimate_generate_bytes(
            n_params=n_params,
            max_len=T,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            vocab_size=vocab_size,
            prompt_len=prompt_len,
            max_new_tokens=nt,
            use_kv_cache=use_kv_cache,
            eval_per_layer=eval_pl,
            layer_strategy=layer_strategy,
        )

    est_b = _bytes(new_toks)
    while est_b > usable and new_toks > 1:
        new_toks = max(1, new_toks // 2)
        actions.append(f"max_new_tokens -> {new_toks} (generate budget)")
        est_b = _bytes(new_toks)

    return MemoryPlan(
        mode="generate",
        batch_size=1,
        max_len=T,
        grad_accum=1,
        gradient_checkpointing=True,
        fp16_activations=False,
        eval_per_layer=eval_pl,
        estimated_bytes=est_b,
        usable_bytes=usable,
        budget_bytes=budget,
        requested_batch=1,
        requested_max_len=T,
        requested_accum=1,
        requested_checkpoint=True,
        layer_strategy=str(layer_strategy),
        actions=actions,
        breakdown_mb={"total": round(est_b / (1024 ** 2), 1)},
        max_new_tokens=new_toks,
        fits=est_b <= usable,
    )


def apply_train_plan(
    gpt_config,
    hyperparams: Mapping[str, Any],
    n_params: int,
    *,
    model_dict: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    autoscale: bool = True,
    headroom: float = _DEFAULT_HEADROOM,
    allow_checkpoint: bool = True,
    allow_fp16: bool = True,
    allow_stream: bool = True,
    force_stream: bool = False,
) -> MemoryPlan:
    """Mutate config/hyperparams to the plan. Raises if the process cap cannot be met."""
    strategy = str(getattr(gpt_config, "layer_strategy", "resident"))
    if force_stream:
        strategy = "stream"
    if not allow_stream:
        strategy = "resident"
    plan = plan_train(
        n_params=int(n_params),
        batch_size=int(hyperparams.get("batch_size", 1)),
        max_len=int(gpt_config.max_len),
        embedding_dim=int(gpt_config.embedding_dim),
        num_heads=int(gpt_config.num_heads),
        num_layers=int(gpt_config.num_layers),
        vocab_size=int(gpt_config.vocab_size),
        grad_accum=int(hyperparams.get("gradient_accumulation_steps", 1)),
        gradient_checkpointing=bool(getattr(gpt_config, "gradient_checkpointing", False)),
        autoscale=autoscale,
        headroom=headroom,
        allow_checkpoint=allow_checkpoint,
        allow_fp16=allow_fp16,
        layer_strategy=strategy,
        allow_stream=allow_stream,
    )
    if not plan.fits:
        raise MemoryBudgetError(
            f"process unified memory cannot fit this architecture under {process_budget_label()} "
            f"(C={gpt_config.embedding_dim} L={gpt_config.num_layers} "
            f"H={gpt_config.num_heads} V={gpt_config.vocab_size} params={n_params:,}): "
            f"estimate={plan.estimated_bytes / (1024 ** 2):.0f} MB "
            f"usable={plan.usable_bytes / (1024 ** 2):.0f} MB "
            f"after autoscale B={plan.batch_size} T={plan.max_len} "
            f"ckpt={plan.gradient_checkpointing} layers={plan.layer_strategy}."
        )

    gpt_config.max_len = int(plan.max_len)
    gpt_config.gradient_checkpointing = bool(plan.gradient_checkpointing)
    gpt_config.eval_per_layer = bool(plan.eval_per_layer)
    gpt_config.layer_strategy = str(plan.layer_strategy)
    if model_dict is not None:
        model_dict["max_len"] = int(plan.max_len)
        model_dict["gradient_checkpointing"] = bool(plan.gradient_checkpointing)
        model_dict["layer_strategy"] = str(plan.layer_strategy)
    if not isinstance(hyperparams, dict):
        raise TypeError("hyperparams must be a mutable dict")
    hyperparams["batch_size"] = int(plan.batch_size)
    hyperparams["gradient_accumulation_steps"] = int(plan.grad_accum)
    stride = int(hyperparams.get("window_stride", 1) or 1)
    if stride >= plan.max_len:
        hyperparams["window_stride"] = max(1, plan.max_len // 2)
        plan.actions.append(f"window_stride {stride}->{hyperparams['window_stride']}")

    from model.cuda.fp16_storage import set_fp16_activation_storage

    set_fp16_activation_storage(bool(plan.fp16_activations))

    if config is not None:
        config["memory_plan"] = plan.to_dict()

    logger.info("[memory] %s breakdown=%s", plan.summary_line(), plan.breakdown_mb)
    print(f"[memory] {plan.summary_line()}")
    return plan


def apply_generate_plan(
    gpt_config,
    n_params: int,
    *,
    prompt_len: int,
    max_new_tokens: int,
    use_kv_cache: bool = True,
    headroom: float = _DEFAULT_HEADROOM,
) -> MemoryPlan:
    gpt_config.eval_per_layer = True
    strategy = str(getattr(gpt_config, "layer_strategy", "resident"))
    plan = plan_generate(
        n_params=int(n_params),
        max_len=int(gpt_config.max_len),
        embedding_dim=int(gpt_config.embedding_dim),
        num_heads=int(gpt_config.num_heads),
        num_layers=int(gpt_config.num_layers),
        vocab_size=int(gpt_config.vocab_size),
        prompt_len=int(prompt_len),
        max_new_tokens=int(max_new_tokens),
        use_kv_cache=use_kv_cache,
        headroom=headroom,
        layer_strategy=strategy,
    )
    if not plan.fits:
        raise MemoryBudgetError(
            f"generate cannot fit under {process_budget_label()}: estimate="
            f"{plan.estimated_bytes / (1024 ** 2):.0f} MB "
            f"usable={plan.usable_bytes / (1024 ** 2):.0f} MB "
            f"T={plan.max_len} L={gpt_config.num_layers} C={gpt_config.embedding_dim}"
        )
    logger.info("[memory] generate %s", plan.summary_line())
    if plan.actions:
        print(f"[memory] {plan.summary_line()}")
    return plan
