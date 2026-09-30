#!/usr/bin/env python3
"""Generate probe for an unguided checkpoint. Writes generate_probe.md.

Uses the same cabinet generate path as App chat. Stop the trainer and App
before this process loads Metal. The trainer also runs this in-process at
max_steps so a finished run already has the report.

Usage:
  python unguided_prober.py --name Unguarded-Initialv7-Run-2
  python unguided_prober.py --checkpoint output/checkpoints/Unguarded-Initialv7-Run-2 --n 50
  python unguided_prober.py --name Unguarded-Initialv7-Run-2 --dry-run
  python unguided_prober.py --checkpoint output/checkpoints/English-Phase1 --probe-mode inject
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from paths import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("llm_gpu.unguided")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unguided generate prober (no stdin)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint dir with weights.npz")
    parser.add_argument("--name", type=str, default=None, help="Basename under output/checkpoints and output/runs")
    parser.add_argument("--facts", type=str, default=None, help="Trained cabinet JSONL (default: recipe or v7)")
    parser.add_argument("--config", type=str, default=None, help="Recipe JSON; used for dataset.path")
    parser.add_argument("--out-dir", type=str, default=None, help="Write generate_probe.md here")
    parser.add_argument("--n", type=int, default=50, help="Cabinet generate count")
    parser.add_argument("--max-steps", type=int, default=None, help="Budget for next-step wording (default: 1500)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-ood", action="store_true", help="Skip out-of-mix prompts")
    parser.add_argument("--dry-run", action="store_true", help="Select prompts and print plan; no Metal")
    parser.add_argument(
        "--probe-mode",
        type=str,
        default="cabinet",
        choices=("cabinet", "english", "inject", "tinystories"),
        help="cabinet (stored User:), english (legacy Phase 1 OOD), inject (legacy v10), tinystories (160-token open English)",
    )
    return parser.parse_args(argv)


def _step_from_checkpoint(checkpoint: Path) -> int | None:
    state = checkpoint / "state.json"
    if not state.is_file():
        return None
    try:
        return int((json.loads(state.read_text(encoding="utf-8")) or {}).get("step"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _last_eval(run_dir: Path) -> dict:
    path = run_dir / "eval_summary.json"
    if not path.is_file():
        return {}
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return rec if isinstance(rec, dict) else {}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from training.cabinet_index import load_cabinet
    from training.unguided.decide import decide_next_step
    from training.unguided.prober import (
        PHASE1_OOD_PROMPTS,
        PHASE2_INJECT_PROMPTS,
        TINYSTORIES_PROMPTS,
        resolve_checkpoint,
        resolve_facts,
        resolve_run_dir,
        select_prompts,
        next_step_context,
        write_inject_probe_reports,
        write_probe_reports,
    )

    probe_mode = str(args.probe_mode or "cabinet").strip().lower()
    inject = probe_mode == "inject"
    english = probe_mode == "english"
    tinystories = probe_mode == "tinystories"

    try:
        checkpoint = resolve_checkpoint(args.name, args.checkpoint) if (args.name or args.checkpoint) else None
    except ValueError as exc:
        if args.dry_run:
            checkpoint = None
        else:
            logger.error("%s", exc)
            return 2

    facts = None
    if not inject and not english and not tinystories:
        facts = resolve_facts(args.facts, args.config)
        if not facts.is_file():
            logger.error("Facts not found: %s", facts)
            return 2
    run_dir = resolve_run_dir(args.name, checkpoint or Path("probe"), args.out_dir)
    last_eval = _last_eval(run_dir)
    step = _step_from_checkpoint(checkpoint) if checkpoint is not None else last_eval.get("step")
    max_steps = int(args.max_steps or last_eval.get("max_steps") or 1500)

    if args.dry_run:
        print("=== UNGUIDED PROBER DRY-RUN ===")
        print(f"checkpoint:    {checkpoint}")
        print(f"probe_mode:    {probe_mode}")
        print(f"out_dir:       {run_dir}")
        if inject:
            print("ood prompts:")
            for prompt in PHASE1_OOD_PROMPTS:
                print(f"  - {prompt}")
            print("inject frames:")
            for prompt in PHASE2_INJECT_PROMPTS:
                print(f"  - {prompt}")
        elif english:
            print("ood prompts:")
            for prompt in PHASE1_OOD_PROMPTS:
                print(f"  - {prompt}")
        elif tinystories:
            print("tinystories prompts:")
            for prompt in TINYSTORIES_PROMPTS:
                print(f"  - {prompt}")
        else:
            index = load_cabinet(facts)
            picked = select_prompts(index, int(args.n), seed=int(args.seed))
            print(f"facts:         {facts}")
            print(f"n:             {args.n}")
            print(f"ood:           {not args.no_ood}")
            print(f"selected:      {len(picked)}")
            for fact in picked[:12]:
                print(f"  - {fact.user}")
            if len(picked) > 12:
                print(f"  … {len(picked) - 12} more")
        print("No Metal init. Exiting 0.")
        return 0

    if checkpoint is None or not (checkpoint / "weights.npz").is_file():
        logger.error("Need a checkpoint with weights.npz")
        return 2

    from model.gpt import GPTModel
    from training.checkpoint import load_checkpoint
    from training.unguided.prober import (
        run_english_stop_probe,
        run_generate_probe,
        run_inject_probe,
        run_tinystories_stop_probe,
    )

    logger.info("loading checkpoint %s", checkpoint)
    gpt_config, params, tokenizer, _, _ = load_checkpoint(str(checkpoint))
    model = GPTModel(gpt_config, params)
    policy = {"probe_mode": probe_mode} if (inject or english or tinystories) else {}
    if inject:
        report = run_inject_probe(
            model=model,
            tokenizer=tokenizer,
            step=int(step) if step is not None else None,
            checkpoint=str(checkpoint),
            seed=int(args.seed),
        )
    elif english:
        report = run_english_stop_probe(
            model=model,
            tokenizer=tokenizer,
            step=int(step) if step is not None else None,
            checkpoint=str(checkpoint),
            seed=int(args.seed),
        )
    elif tinystories:
        report = run_tinystories_stop_probe(
            model=model,
            tokenizer=tokenizer,
            step=int(step) if step is not None else None,
            checkpoint=str(checkpoint),
            seed=int(args.seed),
        )
    else:
        report = run_generate_probe(
            model=model,
            tokenizer=tokenizer,
            facts_path=facts,
            n=int(args.n),
            seed=int(args.seed),
            include_ood=not args.no_ood,
            step=int(step) if step is not None else None,
            checkpoint=str(checkpoint),
        )
        if last_eval.get("cabinet_exact_match") is not None:
            report["teacher_forced"] = last_eval.get("cabinet_exact_match")
    verdict = decide_next_step(
        next_step_context(
            step=int(step or 0),
            max_steps=int(max_steps),
            policy=policy,
            last_eval=last_eval,
            report=report,
        )
    )
    if inject:
        md = write_inject_probe_reports(run_dir, report, verdict)
        print(f"inject frames {len(report.get('inject') or [])}  ood dumps {report['ood_mix_copies']}/{report['ood_n']}")
    else:
        md = write_probe_reports(run_dir, report, verdict)
        if not english and not tinystories:
            print(f"exact {report['n_match']}/{report['n_cabinet']} = {report['exact_rate']:.1%}")
            print(f"swap  {report['n_swap']}/{report['n_cabinet']} = {report['swap_rate']:.1%}")
    print(f"mode  {verdict.mode}  understands={verdict.understands}  next={verdict.primary}")
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
