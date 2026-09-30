"""Weight inspector routes (standalone at / or mounted at /weights)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from flask import Blueprint, Flask, jsonify, render_template, request

from paths import (
    DEFAULT_CHAT_URL,
    OUTPUT_CACHE,
    PROJECT_ROOT,
    chat_open_url,
    ensure_output_dirs,
)
from tools.npz_inspect import (
    WEIGHT_SUFFIXES,
    discover_weight_files,
    manifest,
    neuron_partners,
    resolve_weight_path,
    tensor_detail,
)

_APP_DIR = Path(__file__).resolve().parent.parent
_UPLOAD_DIR = OUTPUT_CACHE / "npzviewer"


def _json_error(message: str, status: int):
    return jsonify({"error": message}), status


def create_blueprint(
    *,
    root: Path = PROJECT_ROOT,
    get_initial: Optional[Callable[[], str]] = None,
    initial: str = "",
    chat_url: str = DEFAULT_CHAT_URL,
    api_base: str = "",
    url_prefix: str = "",
    name: str = "npzviewer",
) -> Blueprint:
    bp = Blueprint(name, __name__, url_prefix=url_prefix)
    npz_root = Path(root).resolve()

    def _start_path() -> str:
        qpath = (request.args.get("path") or "").strip()
        if qpath:
            return qpath
        if get_initial is not None:
            return (get_initial() or "").strip()
        return initial

    @bp.get("/")
    def index():
        start = _start_path()
        chat_base = chat_url.rstrip("/")
        return render_template(
            "npzviewer/index.html",
            initial=start,
            chat_base=chat_base,
            chat_url=chat_open_url(chat_base, start),
            api_base=api_base,
        )

    @bp.get("/api/files")
    def api_files():
        files = discover_weight_files()
        return jsonify({"files": files, "initial": _start_path()})

    @bp.get("/api/manifest")
    def api_manifest():
        try:
            path = resolve_weight_path(str(request.args.get("path") or ""), npz_root)
            return jsonify(manifest(path))
        except FileNotFoundError as exc:
            return _json_error(str(exc), 404)
        except (ValueError, OSError, KeyError) as exc:
            return _json_error(str(exc), 400)

    @bp.get("/api/tensor")
    def api_tensor():
        try:
            path = resolve_weight_path(str(request.args.get("path") or ""), npz_root)
            key = str(request.args.get("key") or "").strip()
            if not key:
                return _json_error("missing key", 400)
            payload = tensor_detail(
                path,
                key,
                row0=int(request.args.get("row0") or 0),
                col0=int(request.args.get("col0") or 0),
                rows=int(request.args.get("rows") or 16),
                cols=int(request.args.get("cols") or 8),
                axis0=int(request.args.get("axis0") or 0),
            )
            return jsonify(payload)
        except FileNotFoundError as exc:
            return _json_error(str(exc), 404)
        except KeyError as exc:
            return _json_error(f"unknown key: {exc}", 404)
        except (ValueError, OSError) as exc:
            return _json_error(str(exc), 400)

    @bp.get("/api/neuron")
    def api_neuron():
        try:
            path = resolve_weight_path(str(request.args.get("path") or ""), npz_root)
            key = str(request.args.get("key") or "").strip()
            if not key:
                return _json_error("missing key", 400)
            payload = neuron_partners(
                path,
                key,
                axis=str(request.args.get("axis") or "row"),
                index=int(request.args.get("index") or 0),
                k=int(request.args.get("k") or 12),
                axis0=int(request.args.get("axis0") or 0),
            )
            return jsonify(payload)
        except FileNotFoundError as exc:
            return _json_error(str(exc), 404)
        except KeyError as exc:
            return _json_error(f"unknown key: {exc}", 404)
        except (ValueError, OSError) as exc:
            return _json_error(str(exc), 400)

    @bp.post("/api/upload")
    def api_upload():
        incoming = request.files.get("file")
        if incoming is None or not incoming.filename:
            return _json_error("missing file", 400)
        name = Path(incoming.filename).name
        suffix = Path(name).suffix.lower()
        if suffix not in WEIGHT_SUFFIXES:
            return _json_error("expected .npz, .npy, or .npx", 400)
        ensure_output_dirs()
        _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dest = _UPLOAD_DIR / name
        incoming.save(str(dest))
        try:
            path = resolve_weight_path(str(dest.relative_to(PROJECT_ROOT)), npz_root)
        except ValueError as exc:
            return _json_error(str(exc), 400)
        return jsonify({"path": str(path.relative_to(PROJECT_ROOT)), "name": name})

    return bp


def create_app(
    root: Path = PROJECT_ROOT,
    initial: str = "",
    *,
    chat_url: str = DEFAULT_CHAT_URL,
) -> Flask:
    app = Flask(
        "npzviewer",
        template_folder=str(_APP_DIR),
        static_folder=None,
    )
    app.register_blueprint(
        create_blueprint(root=root, initial=initial, chat_url=chat_url, api_base="")
    )
    return app
