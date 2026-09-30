#!/usr/bin/env python3
"""0.0.9 Autotrainer daemon.

Polls cabinet_retrain.jsonl, builds a bounded mix (queued entities +
Paris/Cubitt/streaming/how-to anchors), launches unguided_trainer.py as an
isolated Metal subprocess, gates, promotes, and writes status for the UI.

Never calls input(). Never imports model.gpt. Do not run beside a live App.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from paths import DATA_DIR, OUTPUT_ROOT, PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.checkpoint import promote_best
from training.unguided.containment import (
    harvest_keys,
    load_state,
    note_train_started,
    on_crash,
    on_gate_fail,
    on_promoted,
    refuse_train_reason,
    resolve_log_cursor,
    save_state,
)
from training.unguided.gate import gate_eval
from training.unguided.harvest import (
    ANCHOR_QUESTIONS,
    build_bounded_mix,
    harvest_retrain_log,
    write_fresh_recipe,
)
from training.unguided.supervisor import (
    refuse_spawn_reason,
    spawn_kernel,
    write_status,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("llm_gpu.autotrainer_daemon")

DEFAULT_MANIFEST = PROJECT_ROOT / "legacy" / "setup" / "autotrainer_config.json"


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="0.0.9 Autotrainer daemon")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--retrain-log", default=None)
    parser.add_argument("--status", default=None)
    parser.add_argument("--state", default=None)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--recipe", default=None)
    parser.add_argument("--poll-s", type=float, default=None)
    parser.add_argument("--once", action="store_true", help="Single harvest cycle then exit")
    parser.add_argument("--unguarded", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-idle-app", dest="require_idle_app", action="store_true")
    parser.add_argument("--no-require-idle-app", dest="require_idle_app", action="store_false")
    parser.set_defaults(require_idle_app=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = _load_json(Path(args.manifest)) if args.manifest else {}

    retrain_log = Path(args.retrain_log or manifest.get("retrain_log") or OUTPUT_ROOT / "cabinet_retrain.jsonl")
    status_path = Path(args.status or manifest.get("status") or OUTPUT_ROOT / "autotrainer_status.json")
    policy_path = Path(args.policy or manifest.get("policy") or PROJECT_ROOT / "legacy" / "setup" / "unguided_v7_policy.json")
    recipe_path = Path(args.recipe or manifest.get("recipe") or PROJECT_ROOT / "legacy" / "setup" / "chat_facts_v7_config.json")
    holder_path = Path(manifest.get("metal_holder") or OUTPUT_ROOT / "metal_holder.json")
    if args.state:
        state_path = Path(args.state)
    elif args.status:
        state_path = Path(args.status).with_name("autotrainer_state.json")
    else:
        state_path = Path(manifest.get("state") or OUTPUT_ROOT / "autotrainer_state.json")
    poll_s = float(args.poll_s if args.poll_s is not None else manifest.get("poll_s", 30))
    min_queue = int(manifest.get("min_retrain_queue_size", 5))
    max_queue = int(manifest.get("max_retrain_queue_size", 40))
    max_trains = int(manifest.get("max_trains_per_hour", 2))
    backoff_base = float(manifest.get("crash_backoff_s", 300))
    backoff_max = float(manifest.get("backoff_max_s", 3600))
    max_gate_retries = int(manifest.get("max_gate_retries", 2))
    state = load_state(state_path)
    require_idle = (
        bool(args.require_idle_app)
        if args.require_idle_app is not None
        else bool(manifest.get("require_idle_app", True))
    )
    mix_jsonl = Path(manifest.get("mix_jsonl") or DATA_DIR / "chat_facts_unguided.jsonl")
    mix_txt = Path(manifest.get("mix_txt") or DATA_DIR / "chat_facts_unguided.txt")
    anchors = list(manifest.get("gate_anchors") or ANCHOR_QUESTIONS)
    cycle = 0

    while True:
        cycle += 1
        cursor = resolve_log_cursor(retrain_log, state)
        harvested = harvest_retrain_log(
            retrain_log,
            cursor,
            min_queue_size=min_queue,
            max_queue_size=max_queue,
            skip_keys=state.skip_keys(),
        )
        if not harvested.ready:
            state.byte_offset = harvested.byte_offset
            save_state(state_path, state)
        logger.info(
            "Cycle %d | harvested=%d ready=%s offset=%d poison=%d",
            cycle, len(harvested.rows), harvested.ready, state.byte_offset, len(state.poison),
        )

        if not harvested.ready:
            write_status(status_path, {
                "state": "idle",
                "cycle": cycle,
                "harvested": len(harvested.rows),
                "min_retrain_queue_size": min_queue,
                "needs_chat_restart": False,
                "metal_busy": False,
                "backoff_until": state.backoff_until,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            if args.dry_run:
                print(f"dry-run: queue {len(harvested.rows)}/{min_queue}, no kernel")
                return 0
            if args.once:
                return 0
            time.sleep(poll_s)
            continue

        run_name = f"daemon_cycle_{cycle:04d}"
        run_dir = Path("output/runs") / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        mix_stats = build_bounded_mix(
            harvested.rows,
            broad_jsonl=DATA_DIR / "chat_facts_v7.jsonl",
            user_facts=DATA_DIR / "user_facts.txt",
            out_txt=mix_txt,
            out_jsonl=mix_jsonl,
            retrain_weight=int(manifest.get("retrain_queue_weight", 4)),
            anchor_weight=int(manifest.get("clean_anchor_weight", 3)),
            broad_weight=int(manifest.get("broad_corpus_weight", 1)),
            drop_quarantine=not bool(args.unguarded),
        )
        fresh_recipe = write_fresh_recipe(recipe_path, mix_jsonl, run_dir / "recipe.json")
        policy = _load_json(policy_path)
        policy["run_name"] = run_name
        policy["checkpoint_dir"] = f"output/checkpoints/{run_name}"
        policy["recipe"] = str(fresh_recipe)
        policy["gate_anchors"] = anchors
        policy["harvested_questions"] = harvested.rows
        if args.max_steps is not None:
            policy["max_steps"] = args.max_steps
        cycle_policy = run_dir / "policy.json"
        cycle_policy.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")

        if args.dry_run:
            print(f"dry-run: would launch kernel for {run_name}")
            print(f"  mix recipe → {fresh_recipe}")
            print(f"  harvested  → {len(harvested.rows)}")
            print(f"  mix rows   → {mix_stats.get('n')}")
            return 0

        contained = refuse_train_reason(state, max_trains_per_hour=max_trains)
        refused = contained or refuse_spawn_reason(
            status_path=status_path,
            holder_path=holder_path,
            require_idle_app=require_idle,
        )
        if refused:
            write_status(status_path, {
                "state": "idle",
                "cycle": cycle,
                "harvested": len(harvested.rows),
                "error": refused,
                "needs_chat_restart": False,
                "metal_busy": False,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            logger.warning("Refusing spawn: %s", refused)
            if args.once:
                return 1
            time.sleep(poll_s)
            continue

        write_status(status_path, {
            "state": "training",
            "cycle": cycle,
            "harvested": len(harvested.rows),
            "run_name": run_name,
            "mix": mix_stats,
            "needs_chat_restart": False,
            "metal_busy": True,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

        note_train_started(state)
        save_state(state_path, state)
        rc = spawn_kernel(
            config=fresh_recipe,
            policy=cycle_policy,
            unguarded=bool(args.unguarded),
            max_steps=args.max_steps,
            max_wall_s=float(policy.get("max_wall_s", 7200)),
        )

        child_summary = _load_json(run_dir / "summary.json")
        if not child_summary:
            child_summary = _load_json(run_dir / "eval_summary.json")
        gate = gate_eval(child_summary, anchors=anchors)
        ckpt_dir = Path(policy["checkpoint_dir"])
        promoted = ""
        cycle_state = "aborted"
        keys = harvest_keys(harvested.rows)
        abort_reason = ""
        abort_path = run_dir / "ABORT_REASON"
        if abort_path.is_file():
            abort_reason = abort_path.read_text(encoding="utf-8").strip()
        crash_reason = abort_reason or f"exit_{rc}"

        if rc != 0:
            on_crash(
                state, keys, harvested.byte_offset,
                reason=crash_reason, base_s=backoff_base, max_s=backoff_max,
            )
            cycle_state = "aborted"
        elif gate.passed:
            source = ckpt_dir / "best" if (ckpt_dir / "best" / "config.json").exists() else ckpt_dir
            if (source / "config.json").exists():
                promote_best(
                    ckpt_dir,
                    source,
                    meta={"gate": gate.reason, "cycle": cycle, "run_name": run_name},
                )
                promoted = str(ckpt_dir / "best")
                cycle_state = "promoted"
                on_promoted(state, keys, harvested.byte_offset)
            else:
                cycle_state = "aborted"
                gate.passed = False
                gate.reason = "missing_checkpoint"
                on_crash(
                    state, keys, harvested.byte_offset,
                    reason="missing_checkpoint", base_s=backoff_base, max_s=backoff_max,
                )
        else:
            kept = on_gate_fail(
                state, keys, harvested.byte_offset,
                reason=gate.reason, max_retries=max_gate_retries,
                base_s=backoff_base, max_s=backoff_max,
            )
            cycle_state = "gate_failed"
            logger.warning(
                "Gate failed: %s missing=%s kept=%s",
                gate.reason, gate.missing_anchors, kept,
            )

        save_state(state_path, state)
        write_status(status_path, {
            "state": cycle_state,
            "cycle": cycle,
            "harvested": len(harvested.rows),
            "run_name": run_name,
            "exit_code": rc,
            "gate": gate.reason,
            "checkpoint": promoted,
            "weights": f"{promoted}/weights.npz" if promoted else "",
            "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) if promoted else "",
            "needs_chat_restart": bool(promoted),
            "metal_busy": False,
            "abort_reason": abort_reason,
            "poison": len(state.poison),
            "backoff_until": state.backoff_until,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        logger.info("Kernel finished rc=%d state=%s gate=%s", rc, cycle_state, gate.reason)

        if args.once:
            return 0 if cycle_state == "promoted" else 1
        time.sleep(poll_s)


if __name__ == "__main__":
    sys.exit(main())
