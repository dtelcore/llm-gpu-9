"""Chat UI routes (standalone at / or mounted at /chat)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from flask import Blueprint, Flask, jsonify, render_template, request

from paths import DEFAULT_VIEWER_URL, checkpoint_weights_relpath, viewer_open_url

_APP_DIR = Path(__file__).resolve().parent.parent
_EMPTY_STATUS = {
    "checkpoint": "",
    "weights": "",
    "model": "",
    "router": False,
    "search": False,
    "cabinet": 0,
    "viewer_url": DEFAULT_VIEWER_URL + "/",
}


def create_blueprint(
    get_session: Callable[[], object],
    *,
    viewer_url: str = DEFAULT_VIEWER_URL,
    api_base: str = "",
    url_prefix: str = "",
    name: str = "webui",
) -> Blueprint:
    bp = Blueprint(name, __name__, url_prefix=url_prefix)

    def _status() -> dict:
        session = get_session()
        if session is None:
            st = dict(_EMPTY_STATUS)
            st["viewer_url"] = viewer_url.rstrip("/") + "/"
            return st
        st = dict(session.status())
        ckpt = st.get("checkpoint") or ""
        st["weights"] = st.get("weights") or checkpoint_weights_relpath(ckpt)
        st["viewer_url"] = viewer_open_url(viewer_url, ckpt)
        return st

    @bp.get("/")
    def index():
        return render_template(
            "webui/index.html",
            status=_status(),
            api_base=api_base,
        )

    @bp.get("/api/status")
    def api_status():
        return jsonify(_status())

    @bp.post("/api/chat")
    def api_chat():
        session = get_session()
        if session is None:
            return jsonify({"error": "no model loaded"}), 503
        body = request.get_json(silent=True) or {}
        message = str(body.get("message") or body.get("prompt") or "").strip()
        if not message:
            return jsonify({"error": "empty message"}), 400
        result = session.turn(message)
        if result.quit:
            return jsonify({"error": "quit is a CLI command"}), 400
        return jsonify(
            {
                "reply": result.text,
                "kind": result.kind,
                "detail": result.detail,
                "learned_added": result.learned_added,
                "related": list(result.related or []),
                "match_type": getattr(result, "match_type", "") or "",
                "canonical": getattr(result, "canonical", "") or "",
                "classification": getattr(result, "classification", "") or "",
            }
        )

    @bp.post("/api/clear")
    def api_clear():
        session = get_session()
        if session is None:
            return jsonify({"error": "no model loaded"}), 503
        session.clear()
        return jsonify({"ok": True})

    @bp.post("/v1/chat/completions")
    def openai_chat():
        session = get_session()
        if session is None:
            return jsonify({"error": {"message": "no model loaded"}}), 503
        body = request.get_json(silent=True) or {}
        messages = body.get("messages") or []
        content = ""
        if messages:
            content = str((messages[-1] or {}).get("content") or "")
        elif body.get("prompt"):
            content = str(body.get("prompt"))
        content = content.strip()
        if not content:
            return jsonify({"error": {"message": "empty messages"}}), 400
        result = session.turn(content)
        return jsonify(
            {
                "id": "chatcmpl-mlx",
                "object": "chat.completion",
                "model": session.status()["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result.text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        )

    return bp


def create_app(session, *, viewer_url: str = DEFAULT_VIEWER_URL) -> Flask:
    app = Flask(
        "webui",
        template_folder=str(_APP_DIR),
        static_folder=None,
    )
    app.register_blueprint(
        create_blueprint(lambda: session, viewer_url=viewer_url, api_base="")
    )
    return app
