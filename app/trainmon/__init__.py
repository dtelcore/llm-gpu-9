"""Host-only training monitor. Reads logs; never imports model.gpt."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

from flask import Blueprint, Flask, jsonify, render_template, request

from paths import OUTPUT_LOGS, OUTPUT_ROOT, PROJECT_ROOT

_APP_DIR = Path(__file__).resolve().parent.parent


def _empty_series(name: str = "") -> dict:
    return {
        "name": name,
        "log": name,
        "steps": [],
        "loss": [],
        "ppl": [],
        "tok_s": [],
        "val_loss": [],
        "volatility": [],
        "total_steps": 0,
        "last": {},
        "decisions": [],
        "source": "",
    }


def _safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _carry_forward(values: List[Optional[float]]) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    prev: Optional[float] = None
    for value in values:
        if value is not None:
            prev = value
        out.append(prev)
    return out


_RUN_LOG_LIMIT = 5
_LIVE_AGE_S = 60.0
_LIVE_CLOCK_SLACK_S = 90.0


def _is_run_log(name: str) -> bool:
    """Session logs only. The aggregate training.log and generate logs are not runs."""
    if name == "training.log":
        return False
    return name.startswith("training_") or name.startswith("unguided_")


def _latest_from_text(log_path: Path) -> dict:
    import training_log_plotter as tlp

    last: dict = {}
    try:
        lines = tlp._read_tail_lines(log_path, 200)
    except OSError:
        return last
    for line in lines:
        parsed = tlp._parse_log_line(line)
        if parsed is None:
            continue
        step, total, row = parsed
        last["step"] = step
        last["total"] = total
        if row.get("loss") is not None:
            last["loss"] = row["loss"]
        if row.get("ppl") is not None:
            last["ppl"] = row["ppl"]
        tok = row.get("tok_s")
        if tok is None:
            tok = row.get("tok/s")
        if tok is not None:
            last["tok_s"] = tok
        if row.get("val_loss") is not None:
            last["val_loss"] = row["val_loss"]
        if row.get("lr") is not None:
            last["lr"] = row["lr"]
        if row.get("grad_norm") is not None:
            last["grad_norm"] = row["grad_norm"]
        if row.get("eta_s") is not None:
            last["eta_s"] = row["eta_s"]
        if row.get("elapsed_s") is not None:
            last["elapsed_s"] = row["elapsed_s"]
        if row.get("step_ms") is not None:
            last["step_ms"] = row["step_ms"]
        if row.get("epoch") is not None:
            last["epoch"] = row["epoch"]
    return last


def _finite_stats(values: List[Optional[float]]) -> dict:
    nums = [v for v in values if v is not None]
    if not nums:
        return {}
    return {
        "min": min(nums),
        "max": max(nums),
        "mean": sum(nums) / len(nums),
        "first": nums[0],
        "last": nums[-1],
    }


def _stamped_train_rows(log_path: Path, tail: int = 120) -> list:
    """(timestamp, step, total, kv) for train lines in the file tail."""
    import training_log_plotter as tlp

    try:
        lines = tlp._read_tail_lines(log_path, tail)
    except OSError:
        return []
    stamped = []
    for line in lines:
        match = _TS_RE.match(line)
        parsed = tlp._parse_log_line(line)
        if not match or parsed is None or parsed[2].get("loss") is None:
            continue
        try:
            ts = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        step, total, row = parsed
        stamped.append((ts, step, total, row))
    return stamped


def _latest_segment(stamped: list) -> list:
    """Trailing rows whose step counter did not reset."""
    segment: list = []
    for item in stamped:
        if segment and item[1] < segment[-1][1]:
            segment = [item]
        else:
            segment.append(item)
    return segment


def _clocks_from_log(log_path: Path) -> dict:
    """Elapsed and ETA for the current session.

    The trainer already writes elapsed_s and eta_s on each [train] line, measured
    from this process start. Prefer those. A resumed log file still begins with
    older sessions, so wall time back to the first line in the file is not the
    run clock. While the file is still being written, advance the last line's
    clocks by the seconds since that line.
    """
    segment = _latest_segment(_stamped_train_rows(log_path))
    if not segment:
        return {}
    last_ts, last_step, last_total, last_row = segment[-1]
    elapsed = _safe_float(last_row.get("elapsed_s"))
    eta = _safe_float(last_row.get("eta_s"))
    step_ms = _safe_float(last_row.get("step_ms"))
    age = (datetime.now() - last_ts).total_seconds()
    try:
        fresh = (time.time() - log_path.stat().st_mtime) < _LIVE_AGE_S
    except OSError:
        fresh = False
    out: dict = {"started_at": segment[0][0].isoformat(sep=" ", timespec="seconds")}
    if elapsed is not None:
        if fresh and 0.0 <= age <= _LIVE_CLOCK_SLACK_S:
            elapsed += age
            if eta is not None:
                eta = max(0.0, eta - age)
        out["elapsed_s"] = elapsed
        if eta is not None:
            out["eta_s"] = eta
        if step_ms is not None and step_ms > 0:
            out["sec_per_step"] = step_ms / 1000.0
        elif eta is not None and last_total > last_step:
            out["sec_per_step"] = eta / float(last_total - last_step)
        return out
    span = max(0.0, (last_ts - segment[0][0]).total_seconds())
    steps = last_step - segment[0][1]
    out["elapsed_s"] = span
    if span > 0 and steps > 0:
        sec = span / float(steps)
        out["sec_per_step"] = sec
        out["eta_s"] = max(0, last_total - last_step) * sec
    return out


def _kind(name: str) -> str:
    if name.startswith("unguided_"):
        return "unguided"
    if name.startswith("training"):
        return "training"
    return "other"


def _pretty_log_name(name: str) -> str:
    stem = Path(name).stem
    if stem.startswith("unguided_"):
        return stem[len("unguided_") :]
    if stem.startswith("training_"):
        return stem[len("training_") :]
    return stem


def _rolling_std(values: List[Optional[float]], window: int = 8) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for i, raw in enumerate(values):
        if raw is None:
            out.append(None)
            continue
        start = max(0, i - max(2, window) + 1)
        chunk = [v for v in values[start : i + 1] if v is not None]
        if len(chunk) < 2:
            out.append(0.0)
            continue
        mean = sum(chunk) / len(chunk)
        var = sum((x - mean) ** 2 for x in chunk) / len(chunk)
        out.append(var ** 0.5)
    return out


def _run_name_from_log(log_path: Path) -> str:
    stem = log_path.stem
    if stem.startswith("unguided_"):
        return stem[len("unguided_") :]
    if stem.startswith("training_"):
        return stem[len("training_") :]
    return stem


def _list_logs(log_dir: Path) -> List[dict]:
    """The five newest training sessions. A live file is first so Follow live lands on it."""
    if not log_dir.is_dir():
        return []
    now = time.time()
    rows = []
    for path in log_dir.glob("*.log"):
        if not _is_run_log(path.name):
            continue
        st = path.stat()
        kind = _kind(path.name)
        age = max(0.0, now - st.st_mtime)
        rows.append(
            {
                "name": path.name,
                "path": path.name,
                "label": _pretty_log_name(path.name),
                "kind": kind,
                "live": age < _LIVE_AGE_S,
                "age_s": age,
                "bytes": st.st_size,
                "mtime": st.st_mtime,
            }
        )
    rows.sort(key=lambda r: (0 if r["live"] else 1, -r["mtime"]))
    return rows[:_RUN_LOG_LIMIT]


def _resolve_in_log_dir(wanted: str, logs: Path) -> Optional[Path]:
    logs = Path(logs).resolve()
    name = Path(wanted).name
    if not name or name.startswith("."):
        return None
    path = (logs / name).resolve()
    try:
        path.relative_to(logs)
    except ValueError:
        return None
    return path


def _decisions_for_run(run_name: str, *, runs_root: Optional[Path] = None) -> List[dict]:
    root = Path(runs_root) if runs_root is not None else OUTPUT_ROOT / "runs"
    path = root / run_name / "decisions.jsonl"
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _series_from_decisions(name: str, records: List[dict]) -> dict:
    steps: List[int] = []
    vals: List[Optional[float]] = []
    for rec in records:
        if rec.get("step") is None:
            continue
        try:
            steps.append(int(rec["step"]))
        except (TypeError, ValueError):
            continue
        vals.append(_safe_float(rec.get("val_loss")))
    last_i = len(steps) - 1
    last = {}
    if last_i >= 0:
        last = {
            "step": steps[last_i],
            "total": steps[last_i],
            "loss": vals[last_i] if last_i < len(vals) else None,
            "ppl": None,
            "tok_s": None,
            "val_loss": vals[last_i] if last_i < len(vals) else None,
        }
    return {
        "name": name,
        "log": name,
        "steps": steps,
        "loss": vals,
        "ppl": [],
        "tok_s": [],
        "val_loss": vals,
        "volatility": _rolling_std(vals),
        "total_steps": steps[-1] if steps else 0,
        "last": last,
        "decisions": records,
        "source": "decisions",
    }


def series_payload(
    log_path: Path,
    *,
    tail: int = 1000,
    keep_short: bool = True,
    runs_root: Optional[Path] = None,
) -> dict:
    import training_log_plotter as tlp

    run_name = _run_name_from_log(log_path)
    decisions = _decisions_for_run(run_name, runs_root=runs_root)
    empty = _empty_series(log_path.name)
    empty["decisions"] = decisions
    if not log_path.is_file():
        return _series_from_decisions(log_path.name, decisions) if decisions else empty

    min_points = 2 if keep_short else tlp.DEFAULT_MIN_POINTS
    try:
        runs = tlp._load_runs(
            [log_path],
            all_runs=False,
            use_cache=False,
            min_points=min_points,
            tail_lines=int(tail),
        )
    except (OSError, ValueError):
        runs = []
    if not runs:
        return _series_from_decisions(log_path.name, decisions) if decisions else empty
    run = runs[0]
    loss = _carry_forward(list(run.metrics.get("loss") or run.metrics.get("avg_loss") or []))
    ppl = _carry_forward(list(run.metrics.get("ppl") or []))
    tok = _carry_forward(list(run.metrics.get("tok_s") or run.metrics.get("tok/s") or []))
    val = list(run.metrics.get("val_loss") or [])
    last_i = len(run.steps) - 1
    last = {
        "step": run.steps[last_i] if last_i >= 0 else 0,
        "total": run.total_steps,
        "loss": loss[last_i] if last_i < len(loss) else None,
        "ppl": ppl[last_i] if last_i < len(ppl) else None,
        "tok_s": tok[last_i] if last_i < len(tok) else None,
        "val_loss": val[last_i] if last_i < len(val) else None,
    }
    last.update(_latest_from_text(log_path))
    if decisions and last.get("val_loss") is None:
        for rec in reversed(decisions):
            parsed = _safe_float(rec.get("val_loss"))
            if parsed is not None:
                last["val_loss"] = parsed
                break
    first_loss = next((v for v in loss if v is not None), None)
    prev_loss = next((v for v in reversed(loss[:-1]) if v is not None), None) if len(loss) > 1 else None
    if last.get("loss") is not None and first_loss is not None:
        last["delta_loss"] = last["loss"] - first_loss
    if last.get("loss") is not None and prev_loss is not None:
        last["dloss"] = last["loss"] - prev_loss
    last["n_points"] = len(run.steps)
    last["loss_stats"] = _finite_stats(loss)
    last["tok_stats"] = _finite_stats(tok)
    last["ppl_stats"] = _finite_stats(ppl)
    last.update(_clocks_from_log(log_path))
    if decisions:
        last["decision"] = decisions[-1]
    return {
        "name": run.name,
        "log": log_path.name,
        "label": _pretty_log_name(log_path.name),
        "steps": run.steps,
        "loss": loss,
        "ppl": ppl,
        "tok_s": tok,
        "val_loss": val,
        "volatility": _rolling_std(loss),
        "total_steps": run.total_steps,
        "last": last,
        "decisions": decisions,
        "source": "log",
    }


def create_blueprint(
    *,
    log_dir: Optional[Path] = None,
    runs_root: Optional[Path] = None,
    api_base: str = "",
    url_prefix: str = "",
    name: str = "trainmon",
) -> Blueprint:
    bp = Blueprint(name, __name__, url_prefix=url_prefix)
    logs = Path(log_dir) if log_dir is not None else OUTPUT_LOGS

    @bp.get("/")
    def index():
        return render_template("trainmon/index.html", api_base=api_base)

    @bp.get("/api/logs")
    def api_logs():
        available = _list_logs(logs)
        return jsonify({
            "logs": available,
            "preferred": available[0]["name"] if available else "",
        })

    @bp.get("/api/series")
    def api_series():
        wanted = (request.args.get("log") or "").strip()
        available = _list_logs(logs)
        if wanted:
            path = _resolve_in_log_dir(wanted, logs)
            if path is None:
                return jsonify({"error": "log outside log dir"}), 400
        elif available:
            path = logs / available[0]["name"]
        else:
            return jsonify(_empty_series())
        if not path.is_file():
            return jsonify({"error": f"missing log: {path.name}"}), 404
        tail = int(request.args.get("tail") or 1000)
        return jsonify(series_payload(path, tail=tail, keep_short=True, runs_root=runs_root))

    return bp


def create_app(*, log_dir: Optional[Path] = None, runs_root: Optional[Path] = None) -> Flask:
    app = Flask(
        "trainmon",
        template_folder=str(_APP_DIR),
        static_folder=None,
    )
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.register_blueprint(create_blueprint(log_dir=log_dir, runs_root=runs_root, api_base=""))
    return app
