"""TrainSession wrapper around existing train.py helpers.

Forces no_prompt=True and refuses interactive / cross-BPE resume paths.
Does not copy the train() loop.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("llm_gpu.unguided")

_FORBIDDEN_DIR_MARKERS = ("chat_facts_v6", "chat_facts_v4", "v6", "v4")


@dataclass
class TrainSession:
    """Opaque handle holding tokenizer, model, optimizer, datasets, and run state."""

    args: Any
    tokenizer: Any
    model: Any
    optimizer: Any
    params: Any
    config: dict
    gpt_config: Any
    train_dataset: Any
    val_dataset: Any
    checkpoint_dir: Path
    run_dir: Path
    vocab_fingerprint: str
    dataset_path: str
    grad_accum: int
    rng: Any
    step: int = 0
    epoch: int = 0
    best_val_loss: float | None = None
    recent_val_losses: list[float] = field(default_factory=list)
    no_improvement_count: int = 0
    start_wall: float = 0.0
    batch_iter: Any = None

    def wall_s(self) -> float:
        return time.time() - self.start_wall


def fingerprint_vocab(tokenizer) -> str:
    """Stable hash of vocab so we can refuse cross-BPE resume."""
    try:
        if hasattr(tokenizer, "get_vocab"):
            vocab = tokenizer.get_vocab()
            raw = json.dumps(sorted(vocab.items()), sort_keys=True).encode()
        elif hasattr(tokenizer, "vocab"):
            vocab = tokenizer.vocab
            if isinstance(vocab, dict):
                raw = json.dumps(sorted(vocab.items()), sort_keys=True).encode()
            else:
                raw = json.dumps(list(vocab), sort_keys=True).encode()
        else:
            raw = str(type(tokenizer)).encode()
        return hashlib.sha256(raw).hexdigest()[:16]
    except Exception:
        return "unknown"


def refuse_checkpoint_dir(checkpoint_dir: Path, policy: dict, *, existing_ok: bool = False) -> None:
    """Hard refusals that --unguarded cannot skip."""
    hard = policy.get("hard_limits") or {}
    name = checkpoint_dir.name.lower()
    if hard.get("refuse_existing_v6_v4_dir", True):
        if any(marker in name for marker in _FORBIDDEN_DIR_MARKERS):
            raise RuntimeError(
                f"Refusing checkpoint dir that looks like a v6/v4 resume: {checkpoint_dir}. "
                "Unguided runs must start in a fresh directory."
            )
    weights = checkpoint_dir / "weights.npz"
    if weights.exists() and not existing_ok:
        raise RuntimeError(
            f"Refusing to train into an existing checkpoint ({weights}). "
            "Unguided runs need a fresh directory and a fresh BPE; do not --resume."
        )


def refuse_vocab_mismatch(checkpoint_dir: Path, vocab_fp: str, policy: dict) -> None:
    hard = policy.get("hard_limits") or {}
    if not hard.get("refuse_resume_on_vocab_mismatch", True):
        return
    meta = checkpoint_dir / "unguided_meta.json"
    if not meta.exists():
        return
    try:
        prev = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    prev_fp = str(prev.get("vocab_fingerprint") or "")
    if prev_fp and prev_fp != vocab_fp:
        raise RuntimeError(
            f"Vocab fingerprint mismatch: checkpoint has {prev_fp}, "
            f"current recipe has {vocab_fp}. Refusing --resume across a BPE change."
        )


def make_train_args(
    config_path: Path,
    checkpoint_dir: Path,
    policy: dict,
    *,
    unguarded: bool = False,
) -> Any:
    """Build a train.py argparse namespace. Always forces --no-prompt."""
    import train as train_mod

    max_steps = int(policy.get("max_steps", 500))
    eval_every = int(policy.get("eval_every", 50))
    log_every = max(1, int(policy.get("log_every", 10)))
    argv = [
        "--config", str(config_path),
        "--checkpoint", str(checkpoint_dir),
        "--no-prompt",
        "--steps", str(max_steps),
        "--no-generate-probe",
        "--log-every", str(log_every),
        "--checkpoint-every", str(max(1, eval_every)),
        "--run-budget", str(max_steps),
    ]
    args = train_mod.parse_args(argv)
    args.no_prompt = True
    args.menu = False
    args.resume = False
    args.plot = False
    args.unguarded = bool(unguarded)
    args.quality_trial = False
    return args


def build_train_session(args, policy: dict) -> TrainSession:
    """Construct tokenizer / model / optimizer / datasets without calling input()."""
    if hasattr(args, "no_prompt"):
        args.no_prompt = True
    if hasattr(args, "menu"):
        args.menu = False
    if hasattr(args, "resume"):
        args.resume = False

    from logging_config import setup_logging
    from model.gpt import GPTModel
    from model.weights import ModelParameters
    from paths import ensure_output_dirs, run_root_for_checkpoint
    from train import _build_prebuilt_windowed_dataset, _build_windowed_dataset, build_tokenizer_and_config
    import cli_common
    from training.eval import ensure_train_val_split
    from training.gpu_optimizer import AdamWGPU
    from training.memory_controller import apply_train_plan
    from training.memory_preflight import n_params_from_gpt_config
    from training.quality import apply_chat_quality_defaults

    ensure_output_dirs()
    checkpoint_dir = Path(policy.get("checkpoint_dir") or args.checkpoint)
    refuse_checkpoint_dir(checkpoint_dir, policy)
    run_dir = run_root_for_checkpoint(str(checkpoint_dir))
    args.checkpoint = str(run_dir)
    setup_logging(log_filename=f"unguided_{run_dir.name}")

    config = cli_common.load_config(args.config)
    hyperparams = config["hyperparameters"]
    apply_chat_quality_defaults(args, config)
    cli_common.prompt_model_hyperparams(args, config["model"], hyperparams)
    tokenizer, gpt_config = build_tokenizer_and_config(config, args)
    apply_train_plan(
        gpt_config,
        hyperparams,
        n_params_from_gpt_config(gpt_config),
        model_dict=config["model"],
        config=config,
        autoscale=not getattr(args, "no_autoscale", False),
        headroom=float(getattr(args, "memory_headroom", None) or 0.15),
        allow_checkpoint=not getattr(args, "no_grad_checkpoint", False),
        allow_stream=not getattr(args, "no_layer_stream", False),
        force_stream=bool(getattr(args, "layer_stream", False)),
    )
    vocab_fp = fingerprint_vocab(tokenizer)
    refuse_vocab_mismatch(run_dir, vocab_fp, policy)

    cli_common.prompt_training_length_and_lr(args, hyperparams)
    if args.learning_rate is not None:
        hyperparams["learning_rate"] = args.learning_rate
    if args.steps is None and args.epochs is None:
        args.steps = int(policy.get("max_steps", 500))

    from training.tinystories_tokens import dataset_uses_prebuilt_tokens

    prebuilt_tokens = dataset_uses_prebuilt_tokens(config.get("dataset"))
    if prebuilt_tokens:
        train_corpus, val_corpus = [], []
    else:
        train_corpus, val_corpus = ensure_train_val_split(config, seed=args.seed)
    grad_accum = max(1, int(hyperparams.get("gradient_accumulation_steps", 1)))
    hyperparams["gradient_accumulation_steps"] = grad_accum
    if getattr(args, "min_lr_ratio", None) is not None:
        hyperparams["min_lr_ratio"] = float(args.min_lr_ratio)
    elif "min_lr_ratio" not in hyperparams:
        hyperparams["min_lr_ratio"] = 0.1
    window_stride = (
        int(args.window_stride)
        if getattr(args, "window_stride", None) is not None
        else int(hyperparams.get("window_stride", 1))
    )
    if window_stride >= int(gpt_config.max_len):
        window_stride = max(1, int(gpt_config.max_len) // 2)
    hyperparams["window_stride"] = window_stride

    params = ModelParameters(
        gpt_config,
        init_scales=config.get("weight_initialization", {}),
        seed=args.seed,
    )
    model = GPTModel(gpt_config, params)
    if prebuilt_tokens:
        token_dir = str((config.get("dataset") or {}).get("token_dir"))
        dataset = _build_prebuilt_windowed_dataset(
            tokenizer, gpt_config.max_len, hyperparams["batch_size"],
            window_stride, token_dir, "train",
        )
        val_dataset = None
        try:
            val_dataset = _build_prebuilt_windowed_dataset(
                tokenizer, gpt_config.max_len, hyperparams["batch_size"],
                window_stride, token_dir, "valid",
            )
        except (ValueError, FileNotFoundError) as exc:
            logger.warning("Prebuilt val tokens skipped: %s", exc)
            val_dataset = None
    else:
        dataset = _build_windowed_dataset(
            train_corpus, tokenizer, gpt_config.max_len, hyperparams["batch_size"],
            window_stride, run_dir, "train",
        )
        val_dataset = None
        if val_corpus:
            try:
                val_dataset = _build_windowed_dataset(
                    val_corpus, tokenizer, gpt_config.max_len, hyperparams["batch_size"],
                    window_stride, run_dir, "val",
                )
            except ValueError as exc:
                logger.warning("Val dataset too small for windows; skipping val eval: %s", exc)
                val_dataset = None

    total_steps = int(args.steps or policy.get("max_steps", 500))
    optimizer = AdamWGPU(
        params,
        learning_rate=hyperparams["learning_rate"],
        weight_decay=hyperparams.get("weight_decay", 0.01),
        beta1=hyperparams.get("beta1", 0.9),
        beta2=hyperparams.get("beta2", 0.999),
        epsilon=hyperparams.get("epsilon", 1e-8),
        warmup_steps=hyperparams.get("warmup_steps", 0),
        gradient_clip=hyperparams.get("gradient_clip", 1.0),
        total_steps=total_steps,
        min_lr_ratio=float(hyperparams.get("min_lr_ratio", 0.1)),
    )

    import numpy as np
    from model.cuda import ops as cuda_ops

    cuda_ops.reset_memory_baseline()
    dataset_path = str((config.get("dataset") or {}).get("path") or getattr(args, "config", "") or "")

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "unguided_meta.json").write_text(
        json.dumps(
            {
                "vocab_fingerprint": vocab_fp,
                "dataset_path": dataset_path,
                "policy_recipe": policy.get("recipe"),
                "run_name": policy.get("run_name"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return TrainSession(
        args=args,
        tokenizer=tokenizer,
        model=model,
        optimizer=optimizer,
        params=params,
        config=config,
        gpt_config=gpt_config,
        train_dataset=dataset,
        val_dataset=val_dataset,
        checkpoint_dir=run_dir,
        run_dir=run_dir,
        vocab_fingerprint=vocab_fp,
        dataset_path=dataset_path,
        grad_accum=grad_accum,
        rng=np.random.default_rng(args.seed),
        start_wall=time.time(),
    )
