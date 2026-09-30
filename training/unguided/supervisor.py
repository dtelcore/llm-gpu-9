"""Isolated Metal subprocess launcher. Parent never imports model.gpt."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from paths import OUTPUT_ROOT, PROJECT_ROOT

logger = logging.getLogger("llm_gpu.autotrainer_daemon")

DEFAULT_STATUS = OUTPUT_ROOT / "autotrainer_status.json"
DEFAULT_HOLDER = OUTPUT_ROOT / "metal_holder.json"
BUSY_STATES = frozenset({"training", "metal_busy"})


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def write_metal_holder(path: Path, *, holder: str, checkpoint: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "holder": holder,
                "pid": os.getpid(),
                "checkpoint": checkpoint,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def holder_is_busy(path: Path) -> bool:
    rec = read_json(path)
    pid = int(rec.get("pid") or 0)
    return bool(rec.get("holder")) and _pid_alive(pid)


def refuse_spawn_reason(
    *,
    status_path: Path,
    holder_path: Path,
    require_idle_app: bool = True,
) -> Optional[str]:
    """Return a refusal string, or None if the kernel may launch."""
    status = read_json(status_path)
    if str(status.get("state") or "") in BUSY_STATES or bool(status.get("metal_busy")):
        return "metal_busy_status"
    if require_idle_app and holder_is_busy(holder_path):
        return "app_holds_metal"
    return None


def spawn_kernel(
    *,
    config: Path,
    policy: Path,
    unguarded: bool = False,
    max_steps: Optional[int] = None,
    max_wall_s: Optional[float] = None,
    cwd: Optional[Path] = None,
    executable: Optional[str] = None,
) -> int:
    """Launch unguided_trainer.py as a subprocess. Metal dies with the child."""
    root = cwd or PROJECT_ROOT
    cmd = [
        executable or sys.executable,
        str(root / "unguided_trainer.py"),
        "--config",
        str(config),
        "--policy",
        str(policy),
    ]
    if unguarded:
        cmd.append("--unguarded")
    if max_steps is not None:
        cmd.extend(["--max-steps", str(int(max_steps))])

    logger.info("Launching kernel: %s", " ".join(cmd))
    env = os.environ.copy()
    env["LLM_NO_PROMPT"] = "1"
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            env=env,
            timeout=float(max_wall_s) if max_wall_s else None,
        )
        return int(proc.returncode)
    except subprocess.TimeoutExpired:
        logger.error("Kernel wall-clock timeout (%s s)", max_wall_s)
        return 124
