"""Unified Apple MLX app: model selector + chat + npzviewer + train monitor.

Chat loads one checkpoint onto Metal (2 GB). Viewer mmaps the same
weights.npz. Train reads output/logs only (no Metal). Switching models
restarts this process — no second net.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from argparse import Namespace
from pathlib import Path
from typing import Callable, Optional

from flask import Flask, jsonify, render_template, request

from app.models import list_models, resolve_model
from app.npzviewer import create_blueprint as viewer_blueprint
from app.trainmon import create_blueprint as trainmon_blueprint
from app.webui import create_blueprint as chat_blueprint
from paths import OUTPUT_CHECKPOINTS, OUTPUT_LOGS, OUTPUT_ROOT, PROJECT_ROOT

_APP_DIR = Path(__file__).resolve().parent
_APP_SCRIPT = PROJECT_ROOT / "App.py"


class AppState:
    def __init__(self) -> None:
        self.session = None
        self.selected = None
        self.args: Optional[Namespace] = None


def load_chat_session(args: Namespace):
    from training.chat_session import ChatSession
    from training.unguided.supervisor import write_metal_holder

    args._entry = getattr(args, "_entry", None) or "app"
    session = ChatSession.from_args(args, configure_logging=True)
    write_metal_holder(
        OUTPUT_ROOT / "metal_holder.json",
        holder="app",
        checkpoint=str(getattr(args, "checkpoint", "") or ""),
    )
    return session


def restart_argv(args: Namespace, checkpoint: str, facts: str) -> list:
    cmd = [
        sys.executable,
        str(_APP_SCRIPT),
        "--checkpoint",
        checkpoint,
        "--facts",
        facts,
        "--host",
        str(getattr(args, "host", "127.0.0.1")),
        "--port",
        str(getattr(args, "port", 7860)),
    ]
    if getattr(args, "chat", True) and not getattr(args, "no_chat", False):
        cmd.append("--chat")
    if getattr(args, "no_search", False):
        cmd.append("--no-search")
    if getattr(args, "router", None) is False:
        cmd.append("--no-router")
    elif getattr(args, "router", None) is True:
        cmd.append("--router")
    if getattr(args, "system", None):
        cmd.extend(["--system", str(args.system)])
    return cmd


def _same_checkpoint(current: Optional[str], wanted: str) -> bool:
    if not current:
        return False
    left = Path(str(current)).as_posix().rstrip("/")
    right = Path(wanted).as_posix().rstrip("/")
    return left == right or Path(left).name == Path(right).name


def _selected_from_session(session) -> dict:
    ckpt = str(session.status().get("checkpoint") or "")
    return resolve_model(ckpt) or {
        "id": ckpt,
        "name": Path(ckpt).name,
        "title": session.status().get("model") or "",
        "checkpoint": ckpt,
        "weights": session.status().get("weights") or "",
        "facts": "",
        "bytes": 0,
        "chat": True,
    }


def create_app(
    *,
    session=None,
    selected=None,
    args: Optional[Namespace] = None,
    root: Path = PROJECT_ROOT,
    models_root: Optional[Path] = None,
    load_session: Optional[Callable] = None,
    restart_fn: Optional[Callable] = None,
    autotrainer_status: Optional[Path] = None,
    log_dir: Optional[Path] = None,
) -> Flask:
    state = AppState()
    state.session = session
    state.args = args or Namespace(
        host="127.0.0.1",
        port=7860,
        chat=True,
        no_chat=False,
        no_search=False,
        router=None,
        system=None,
        checkpoint="",
        facts="",
    )
    if selected is not None:
        state.selected = selected
    elif session is not None:
        state.selected = _selected_from_session(session)

    app = Flask(
        "mlx_app",
        template_folder=str(_APP_DIR),
        static_folder=None,
    )
    catalog = Path(models_root) if models_root is not None else OUTPUT_CHECKPOINTS
    app.config["APP_STATE"] = state
    app.config["MODELS_ROOT"] = catalog
    app.config["AUTOTRAINER_STATUS"] = Path(autotrainer_status) if autotrainer_status is not None else OUTPUT_ROOT / "autotrainer_status.json"
    app.url_map.strict_slashes = False

    def get_session():
        return state.session

    def get_initial() -> str:
        if state.selected:
            return str(state.selected.get("weights") or "")
        return ""

    app.register_blueprint(
        chat_blueprint(
            get_session,
            viewer_url="/weights",
            api_base="/chat",
            url_prefix="/chat",
        )
    )
    app.register_blueprint(
        viewer_blueprint(
            root=root,
            get_initial=get_initial,
            chat_url="/chat",
            api_base="/weights",
            url_prefix="/weights",
        )
    )
    app.register_blueprint(
        trainmon_blueprint(
            log_dir=log_dir if log_dir is not None else OUTPUT_LOGS,
            api_base="/train",
            url_prefix="/train",
        )
    )

    @app.get("/")
    def shell():
        return render_template(
            "templates/shell.html",
            models=list_models(catalog),
            selected=state.selected,
            loaded=state.session is not None,
            view=request.args.get("view") or "chat",
        )

    @app.get("/api/models")
    def api_models():
        sel = state.selected or {}
        return jsonify(
            {
                "models": list_models(catalog),
                "selected": sel.get("checkpoint") or "",
                "loaded": state.session is not None,
                "weights": sel.get("weights") or "",
                "facts": sel.get("facts") or "",
            }
        )

    @app.post("/api/select")
    def api_select():
        body = request.get_json(silent=True) or {}
        wanted = str(body.get("checkpoint") or body.get("id") or "").strip()
        if not wanted:
            return jsonify({"error": "missing checkpoint"}), 400
        model = resolve_model(wanted, root=catalog)
        if model is None:
            return jsonify({"error": f"unknown checkpoint: {wanted}"}), 404
        current = ""
        if state.session is not None:
            current = str(state.session.status().get("checkpoint") or "")
        if state.session is None:
            loader = load_session or load_chat_session
            try:
                state.args.checkpoint = model["checkpoint"]
                state.args.facts = model["facts"]
                if model.get("chat") and not getattr(state.args, "no_chat", False):
                    state.args.chat = True
                state.session = loader(state.args)
            except Exception as exc:
                return jsonify({"error": str(exc)}), 500
            state.selected = model
            return jsonify({"ok": True, "loaded": True, "restart": False, "model": model})
        if _same_checkpoint(current, str(model["checkpoint"])):
            state.selected = model
            return jsonify({"ok": True, "loaded": True, "restart": False, "model": model})
        argv = restart_argv(state.args, str(model["checkpoint"]), str(model["facts"]))
        restarter = restart_fn or _default_restart
        restarter(argv)
        return jsonify({"ok": True, "loaded": False, "restart": True, "model": model, "argv": argv})

    @app.get("/api/autotrainer")
    def api_autotrainer():
        path = Path(app.config["AUTOTRAINER_STATUS"])
        if not path.is_file():
            return jsonify({
                "state": "idle",
                "needs_chat_restart": False,
                "checkpoint": "",
                "weights": "",
            })
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return jsonify({"state": "unknown", "needs_chat_restart": False})
        if not isinstance(data, dict):
            return jsonify({"state": "unknown", "needs_chat_restart": False})
        return jsonify(data)

    return app


def _default_restart(argv: list) -> None:
    def _go() -> None:
        os.execv(argv[0], argv)

    threading.Timer(0.25, _go).start()
