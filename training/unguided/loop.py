"""train_segment: run a fixed number of optimizer steps using train.py primitives."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("llm_gpu.unguided")


@dataclass
class TrainMetrics:
    steps_done: int
    avg_loss: float
    ppl: float
    grad_norm: float | None
    nan_detected: bool
    tokens_per_s: float | None = None


def train_segment(session: Any, steps: int) -> TrainMetrics:
    """Execute `steps` optimizer steps on the already-built session. No input()."""
    import numpy as np

    from train import _accumulate_grads_, _iter_batches_forever, _scale_grads_
    from training.checkpoint import save_checkpoint
    from training.eval import perplexity_from_loss
    from training.loss import softmax_cross_entropy_batch, softmax_cross_entropy_batch_gpu

    if steps <= 0:
        return TrainMetrics(steps_done=0, avg_loss=0.0, ppl=1.0, grad_norm=None, nan_detected=False)

    if session.batch_iter is None:
        session.batch_iter = _iter_batches_forever(session.train_dataset, session.rng)

    losses: list[float] = []
    last_gnorm: float | None = None
    nan = False
    t0 = time.time()
    token_count = 0
    accum_grads = None
    accum_loss_sum = 0.0
    accum_count = 0
    optimizer_steps = 0
    grad_accum = max(1, int(session.grad_accum))

    while optimizer_steps < steps:
        batch, epoch = next(session.batch_iter)
        session.epoch = epoch
        xs = np.stack([x for x, _ in batch])
        ys = np.stack([y for _, y in batch])
        logits, cache = session.model.forward_batch(xs, need_host_logits=False)
        if cache.get("gpu"):
            loss, dlogits_d = softmax_cross_entropy_batch_gpu(cache["logits_d"], ys)
            batch_grads = session.model.backward_batch_gpu(
                cache, dlogits_d.reshape(-1, session.model.config.vocab_size),
            )
        else:
            loss, dlogits = softmax_cross_entropy_batch(logits, ys)
            batch_grads = session.model.backward_batch(cache, dlogits)
        batch_loss = float(loss)
        if not math.isfinite(batch_loss):
            nan = True
            break
        accum_grads = _accumulate_grads_(accum_grads, batch_grads)
        accum_loss_sum += batch_loss
        accum_count += 1
        will_step = accum_count >= grad_accum or (optimizer_steps + 1) >= steps
        if not will_step:
            continue
        _scale_grads_(accum_grads, 1.0 / float(accum_count))
        mean_loss = accum_loss_sum / float(accum_count)
        last_gnorm = float(session.optimizer.clip_grads_(accum_grads))
        session.optimizer.step(accum_grads)
        session.step += 1
        optimizer_steps += 1
        losses.append(mean_loss)
        token_count += int(xs.size)
        accum_grads = None
        accum_loss_sum = 0.0
        accum_count = 0
        if not math.isfinite(mean_loss):
            nan = True
            break
        log_every = max(1, int(getattr(session.args, "log_every", 10) or 10))
        total = int(getattr(session.args, "steps", 0) or session.step)
        if session.step % log_every == 0 or optimizer_steps >= steps:
            ppl_now = perplexity_from_loss(mean_loss)
            lr = 0.0
            try:
                lr = float(session.optimizer.current_lr())
            except Exception:
                pass
            logger.info(
                f"[train] step={session.step}/{total} epoch={int(session.epoch or 1)} "
                f"loss={mean_loss:.4f} ppl={ppl_now:.4f} "
                f"tok_s={float(token_count) / max(time.time() - t0, 1e-6):.0f} "
                f"lr={lr:.6g} grad_norm={last_gnorm or 0.0:.4f}"
            )

    avg_loss = float(sum(losses) / len(losses)) if losses else float("nan")
    if not math.isfinite(avg_loss):
        nan = True
    elapsed = max(time.time() - t0, 1e-6)
    tokens_per_s = token_count / elapsed if token_count else None

    try:
        session.optimizer.sync_host_weights()
        save_checkpoint(
            str(session.run_dir),
            session.params,
            session.tokenizer,
            session.config,
            step=session.step,
            epoch=int(session.epoch or 1),
            metrics={
                "step": session.step,
                "loss": avg_loss if math.isfinite(avg_loss) else None,
                "ppl": perplexity_from_loss(avg_loss) if math.isfinite(avg_loss) else None,
            },
        )
    except Exception as exc:
        logger.warning("latest checkpoint save failed: %s", exc)

    return TrainMetrics(
        steps_done=optimizer_steps,
        avg_loss=avg_loss,
        ppl=perplexity_from_loss(avg_loss) if math.isfinite(avg_loss) else float("nan"),
        grad_norm=last_gnorm,
        nan_detected=nan,
        tokens_per_s=tokens_per_s,
    )
