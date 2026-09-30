#!/usr/bin/env python3
"""0.0.9 Unguided (non-interactive) trainer kernel.

Never calls input(). Remix aborts with NEXT_MIX.json + ABORT_REASON.
Does not --resume across a BPE change.

Usage:
  python unguided_trainer.py --config setup/chat_facts_v7_config.json --policy setup/unguided_v7_policy.json
  python unguided_trainer.py --config ... --policy ... --dry-run
  python unguided_trainer.py --config ... --policy ... --unguarded
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from paths import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("llm_gpu.unguided")


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _append_decision(run_dir: Path, record: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "decisions.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_abort(run_dir: Path, reason: str, next_mix: dict | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ABORT_REASON").write_text(reason + "\n", encoding="utf-8")
    if next_mix is not None:
        (run_dir / "NEXT_MIX.json").write_text(json.dumps(next_mix, indent=2) + "\n", encoding="utf-8")


def _write_summary(run_dir: Path, payload: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (run_dir / "eval_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _finalize_stop(
    session,
    policy: dict,
    run_dir: Path,
    last_eval: dict | None,
    action: str,
    reason: str,
    *,
    max_steps: int,
    skip_probe: bool,
) -> None:
    """Write stop artifacts. On a clean stop, generate-probe then decide the next step."""
    payload = {
        **(last_eval or {}),
        "final_step": session.step,
        "best_val_loss": session.best_val_loss,
        "action": action,
        "reason": reason,
    }
    if not skip_probe:
        try:
            from training.unguided.decide import decide_next_step
            from training.unguided.prober import next_step_context, run_session_probe, write_session_stop_reports

            logger.info("stop probe | step=%s n=%s", session.step, policy.get("probe_n", 50))
            report = run_session_probe(session, policy)
            if last_eval and last_eval.get("cabinet_exact_match") is not None:
                report["teacher_forced"] = last_eval.get("cabinet_exact_match")
            verdict = decide_next_step(
                next_step_context(
                    step=session.step,
                    max_steps=max_steps,
                    policy=policy,
                    last_eval=last_eval,
                    report=report,
                )
            )
            md = write_session_stop_reports(run_dir, report, verdict)
            cabinet_report = report.get("cabinet_report") if isinstance(report.get("cabinet_report"), dict) else report
            payload["generate_exact_rate"] = (cabinet_report or {}).get("exact_rate")
            payload["generate_swap_rate"] = (cabinet_report or {}).get("swap_rate")
            payload["next_step"] = verdict.primary
            payload["understands"] = verdict.understands
            payload["recitation_mode"] = verdict.mode
            logger.info(
                "stop probe mode=%s exact=%s swap=%s next=%s md=%s",
                policy.get("probe_mode", "cabinet"),
                (cabinet_report or {}).get("exact_rate"),
                (cabinet_report or {}).get("swap_rate"),
                verdict.primary,
                md,
            )
        except Exception as exc:
            logger.warning("stop probe failed: %s", exc)
            payload["probe_error"] = str(exc)
    _write_abort(run_dir, reason)
    _write_summary(run_dir, payload)


def _dry_run_param_estimate(recipe: dict) -> dict:
    from setup.model_config import estimate_vram_footprint

    model = dict(recipe.get("model") or {})
    dataset = recipe.get("dataset") or {}
    vocab = model.get("vocab_size")
    if not vocab:
        vocab = int(dataset.get("bpe_merges") or 4000) + 256
    model["vocab_size"] = int(vocab)
    try:
        return estimate_vram_footprint(model)
    except Exception:
        return {"total_params": None, "vocab_size": vocab}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="0.0.9 Unguided trainer (no input())")
    parser.add_argument("--config", required=True, help="Recipe JSON (setup/*.json)")
    parser.add_argument("--policy", required=True, help="Policy JSON (setup/unguided_*_policy.json)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit 0, no Metal")
    parser.add_argument(
        "--unguarded",
        action="store_true",
        help="Relax data hygiene only; hard limits still enforced",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Override policy max_steps")
    parser.add_argument("--eval-every", type=int, default=None, help="Override policy eval_every")
    parser.add_argument("--log-every", type=int, default=None, help="Override policy log_every")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Fresh run + checkpoint basename (output/runs/<name>, output/checkpoints/<name>)",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Override policy run_name (output/runs/<name>)")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override policy checkpoint_dir (must be empty / no weights.npz)",
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip the generate prober at max_steps / early_stop / wall",
    )
    return parser.parse_args(argv)


def _safe_basename(raw: str, flag: str) -> str:
    text = str(raw).strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise SystemExit(f"{flag} must be a single directory name, not a path: {raw!r}")
    return text


def _apply_location_overrides(args: argparse.Namespace, policy: dict) -> None:
    if args.name:
        name = _safe_basename(args.name, "--name")
        policy["run_name"] = name
        policy["checkpoint_dir"] = f"output/checkpoints/{name}"
    if args.run_name:
        policy["run_name"] = _safe_basename(args.run_name, "--run-name")
    if args.checkpoint_dir:
        policy["checkpoint_dir"] = str(args.checkpoint_dir).strip()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config)
    policy_path = Path(args.policy)

    if not config_path.is_file():
        logger.error("Recipe not found: %s", config_path)
        return 2
    if not policy_path.is_file():
        logger.error("Policy not found: %s", policy_path)
        return 2

    recipe = _load_json(config_path)
    policy = _load_json(policy_path)
    if args.max_steps is not None:
        policy["max_steps"] = args.max_steps
    if args.eval_every is not None:
        policy["eval_every"] = args.eval_every
    if args.log_every is not None:
        policy["log_every"] = args.log_every
    if args.unguarded:
        policy["unguarded"] = True
    _apply_location_overrides(args, policy)

    run_name = policy.get("run_name", "unguided_v7")
    run_dir = Path("output/runs") / run_name
    max_steps = int(policy.get("max_steps", 500))
    eval_every = int(policy.get("eval_every", 50))
    log_every = max(1, int(policy.get("log_every", 10)))
    max_wall_s = float(policy.get("max_wall_s", 7200))
    checkpoint_dir = Path(policy.get("checkpoint_dir") or f"output/checkpoints/{run_name}")

    if args.dry_run:
        estimate = _dry_run_param_estimate(recipe)
        model = recipe.get("model") or {}
        print("=== UNGUIDED DRY-RUN ===")
        print(f"recipe:        {config_path}")
        print(f"policy:        {policy_path}")
        print(f"run_name:      {run_name}")
        print(f"run_dir:       {run_dir}")
        print(f"checkpoint:    {checkpoint_dir}")
        print(f"dataset:       {(recipe.get('dataset') or {}).get('path')}")
        print(
            f"arch:          C={model.get('embedding_dim')} L={model.get('num_layers')} "
            f"T={model.get('max_len')} H={model.get('num_heads')}"
        )
        print(f"param_est:     {estimate.get('total_params')} (vocab placeholder until BPE)")
        print(f"max_steps:     {max_steps}")
        print(f"eval_every:    {eval_every}")
        print(f"log_every:     {log_every}")
        print(f"max_wall_s:    {max_wall_s}")
        print(f"early_stop:    patience={policy.get('early_stop_patience')}")
        print(f"loss_spike:    ratio={policy.get('loss_spike_ratio')}")
        print(f"remix_if:      {policy.get('remix_if')}")
        print(f"probe_on_stop: {bool(policy.get('probe_on_stop', True)) and not args.no_probe}")
        print(f"probe_mode:    {policy.get('probe_mode', 'cabinet')}")
        print(f"probe_every:   {int(policy.get('probe_every') or 0)}")
        print(f"probe_n:       {int(policy.get('probe_n', 50))}")
        print(f"unguarded:     {bool(args.unguarded)}")
        print("No Metal init. Exiting 0.")
        return 0

    from training.checkpoint import promote_best
    from training.unguided.decide import Decision, DecideContext, decide
    from training.unguided.eval_suite import run_eval_suite
    from training.unguided.loop import train_segment
    from training.unguided.session import build_train_session, make_train_args

    skip_probe = bool(args.no_probe) or not bool(policy.get("probe_on_stop", True))

    train_args = make_train_args(
        config_path, checkpoint_dir, policy, unguarded=bool(args.unguarded),
    )
    try:
        session = build_train_session(train_args, policy)
    except RuntimeError as exc:
        logger.error("Session build refused: %s", exc)
        _write_abort(run_dir, str(exc))
        return 1

    logger.info(
        "Unguided session ready | run=%s | vocab_fp=%s | ckpt=%s",
        run_name, session.vocab_fingerprint, session.checkpoint_dir,
    )

    history: list[float] = []
    start = time.time()
    last_eval = None

    while True:
        remaining = max_steps - session.step
        if remaining <= 0:
            result_action = Decision.STOP_LIMIT
            result_reason = f"max_steps={max_steps}"
            rec = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "step": session.step,
                "action": result_action.value,
                "reason": result_reason,
                "promote": False,
            }
            _append_decision(run_dir, rec)
            _finalize_stop(
                session,
                policy,
                run_dir,
                last_eval,
                result_action.value,
                result_reason,
                max_steps=max_steps,
                skip_probe=skip_probe,
            )
            return 0

        metrics = train_segment(session, min(eval_every, remaining))
        logger.info(
            "segment done | step=%d loss=%.4f nan=%s",
            session.step, metrics.avg_loss, metrics.nan_detected,
        )

        eval_res = run_eval_suite(session, policy)
        if eval_res.val_loss is not None:
            history.append(eval_res.val_loss)
            window = int(policy.get("loss_median_window", 5))
            session.recent_val_losses = history[-window:]

        ctx = DecideContext(
            step=session.step,
            max_steps=max_steps,
            wall_s=time.time() - start,
            max_wall_s=max_wall_s,
            val_loss=eval_res.val_loss,
            best_val_loss=session.best_val_loss,
            recent_val_losses=session.recent_val_losses,
            nan_detected=metrics.nan_detected or eval_res.nan_detected,
            cabinet_exact_match=eval_res.cabinet_exact_match,
            policy=policy,
            no_improvement_count=session.no_improvement_count,
        )
        result = decide(ctx)

        if eval_res.val_loss is not None:
            if session.best_val_loss is None or eval_res.val_loss < session.best_val_loss:
                session.best_val_loss = eval_res.val_loss
                session.no_improvement_count = 0
            else:
                session.no_improvement_count += 1

        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "step": session.step,
            "action": result.action.value,
            "reason": result.reason,
            "val_loss": eval_res.val_loss,
            "cabinet_exact_match": eval_res.cabinet_exact_match,
            "promote": result.promote,
        }
        _append_decision(run_dir, rec)
        logger.info("decision=%s reason=%s", result.action.value, result.reason)

        last_eval = {
            "val_loss": eval_res.val_loss,
            "val_ppl": eval_res.val_ppl,
            "cabinet_exact_match": eval_res.cabinet_exact_match,
            "router_precision": eval_res.router_precision,
            "router_recall": eval_res.router_recall,
            "anchor_hits": eval_res.anchor_hits,
            "harvested_entity_swaps": eval_res.harvested_entity_swaps,
            "harvested_exact": eval_res.harvested_exact,
            "best_val_loss": session.best_val_loss,
            "step": session.step,
        }
        _write_summary(run_dir, {**last_eval, "action": result.action.value, "reason": result.reason})

        if result.promote:
            try:
                promote_best(
                    session.run_dir,
                    session.run_dir,
                    meta={
                        "metric": "val_loss",
                        "value": session.best_val_loss,
                        "step": session.step,
                    },
                )
            except Exception as exc:
                logger.warning("promote_best failed: %s", exc)

        if result.action in (Decision.CONTINUE, Decision.PROMOTE):
            probe_every = int(policy.get("probe_every") or 0)
            if (
                not skip_probe
                and probe_every > 0
                and session.step > 0
                and session.step % probe_every == 0
            ):
                try:
                    from training.unguided.decide import decide_next_step
                    from training.unguided.prober import (
                        next_step_context,
                        run_session_probe,
                        write_session_stop_reports,
                    )

                    report = run_session_probe(session, policy)
                    verdict = decide_next_step(
                        next_step_context(
                            step=session.step,
                            max_steps=max_steps,
                            policy=policy,
                            last_eval=last_eval,
                            report=report,
                        )
                    )
                    md = write_session_stop_reports(run_dir, report, verdict)
                    logger.info("mid-train probe step=%s md=%s", session.step, md)
                except Exception as exc:
                    logger.warning("mid-train probe failed: %s", exc)
            continue

        if result.action == Decision.ABORT_REMIX:
            _write_abort(run_dir, result.reason, result.next_mix_recipe)
            logger.warning("ABORT_REMIX: %s", result.reason)
            return 1

        if result.action == Decision.ABORT_SPIKE:
            _write_abort(run_dir, result.reason)
            logger.error("ABORT_SPIKE: %s", result.reason)
            return 1

        if result.action in (Decision.EARLY_STOP, Decision.STOP_LIMIT):
            logger.info("Stopping: %s", result.reason)
            _finalize_stop(
                session,
                policy,
                run_dir,
                last_eval,
                result.action.value,
                result.reason,
                max_steps=max_steps,
                skip_probe=skip_probe,
            )
            return 0

        _write_abort(run_dir, f"unknown_action:{result.action}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
