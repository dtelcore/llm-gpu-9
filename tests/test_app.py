"""App.py shell: model selector loads the same checkpoint into chat + viewer."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app import create_app, restart_argv
from app.models import describe_checkpoint, infer_facts, list_models


class FakeSession:
    def __init__(self, checkpoint="output/checkpoints/ck"):
        self.checkpoint = checkpoint
        self.cleared = False

    def status(self):
        return {
            "checkpoint": self.checkpoint,
            "weights": self.checkpoint.rstrip("/") + "/weights.npz",
            "model": "Chat C=8 L=1 T=8",
            "vocab_size": 10,
            "max_len": 8,
            "chat": True,
            "router": True,
            "search": False,
            "cabinet": 0,
            "system": "sys",
            "learned": "l",
        }

    def turn(self, message):
        return type("Turn", (), {"text": "4", "kind": "calc", "detail": "calc", "quit": False, "learned_added": 0, "related": []})()

    def clear(self):
        self.cleared = True


def _checkpoint(dirpath: Path, name: str, facts_rel: str = "") -> Path:
    ckpt = Path(dirpath) / name
    ckpt.mkdir(parents=True, exist_ok=True)
    np.savez(ckpt / "weights.npz", token_embedding=np.zeros((2, 2), dtype=np.float32))
    (ckpt / "config.json").write_text(
        json.dumps(
            {
                "model": {"name": "Toy " + name},
                "dataset": {"path": facts_rel or f"data/{name}.jsonl"},
            }
        ),
        encoding="utf-8",
    )
    return ckpt


class ModelCatalogTests(unittest.TestCase):
    def test_list_and_infer_facts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            mix = data / "demo.jsonl"
            mix.write_text("{}\n", encoding="utf-8")
            ckpt = _checkpoint(root / "ckpts", "demo", facts_rel=str(mix))
            rec = describe_checkpoint(ckpt)
            self.assertIsNotNone(rec)
            self.assertEqual(rec["name"], "demo")
            self.assertTrue(str(rec["weights"]).endswith("demo/weights.npz"))
            self.assertEqual(infer_facts(ckpt), str(mix))
            listed = list_models(root / "ckpts")
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0]["title"], "Toy demo")


class AppShellTests(unittest.TestCase):
    def test_selector_loads_chat_and_viewer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = _checkpoint(root, "chat_facts_v9")
            loaded = []

            def loader(args):
                loaded.append(args.checkpoint)
                return FakeSession(str(args.checkpoint))

            restarts = []
            app = create_app(root=root, models_root=root, load_session=loader, restart_fn=restarts.append)
            client = app.test_client()
            home = client.get("/")
            self.assertEqual(home.status_code, 200)
            self.assertIn(b"Select a checkpoint", home.data)
            self.assertIn(b"chat_facts_v9", home.data)
            self.assertIn(b'id="tab-train"', home.data)
            listing = client.get("/api/models")
            self.assertEqual(listing.status_code, 200)
            self.assertFalse(listing.get_json()["loaded"])
            picked = client.post("/api/select", json={"checkpoint": str(ckpt)})
            self.assertEqual(picked.status_code, 200)
            body = picked.get_json()
            self.assertTrue(body["ok"])
            self.assertTrue(body["loaded"])
            self.assertFalse(body["restart"])
            self.assertEqual(len(loaded), 1)
            chat = client.get("/chat/")
            self.assertEqual(chat.status_code, 200)
            self.assertIn(b"Apple MLX", chat.data)
            ping = client.post("/chat/api/chat", json={"message": "2+2"})
            self.assertEqual(ping.status_code, 200)
            self.assertEqual(ping.get_json()["reply"], "4")
            self.assertIn("classification", ping.get_json())
            weights = str(body["model"]["weights"])
            viewer = client.get("/weights/", query_string={"path": Path(ckpt.name) / "weights.npz"})
            self.assertEqual(viewer.status_code, 200)
            self.assertIn(b"NPZ Viewer", viewer.data)
            self.assertIn(weights.encode() if weights.startswith(ckpt.name) else ckpt.name.encode(), viewer.data)

    def test_switch_restarts_instead_of_second_metal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = _checkpoint(root, "one")
            second = _checkpoint(root, "two")
            restarts = []
            app = create_app(
                session=FakeSession(str(first)),
                selected=describe_checkpoint(first),
                args=Namespace(host="127.0.0.1", port=7860, chat=True, no_chat=False, no_search=False, router=None, system=None),
                root=root,
                models_root=root,
                restart_fn=restarts.append,
            )
            client = app.test_client()
            out = client.post("/api/select", json={"checkpoint": str(second)})
            self.assertEqual(out.status_code, 200)
            body = out.get_json()
            self.assertTrue(body["restart"])
            self.assertEqual(len(restarts), 1)
            self.assertIn(str(describe_checkpoint(second)["checkpoint"]), restarts[0])
            argv = restart_argv(Namespace(host="127.0.0.1", port=7860, chat=True, no_chat=False, no_search=False, router=None, system=None), "output/checkpoints/x", "data/x.jsonl")
            self.assertEqual(argv[0], sys.executable)
            self.assertIn("--checkpoint", argv)

    def test_autotrainer_status_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status = root / "autotrainer_status.json"
            status.write_text(
                json.dumps({
                    "state": "promoted",
                    "needs_chat_restart": True,
                    "checkpoint": "output/checkpoints/daemon_cycle_0001/best",
                }),
                encoding="utf-8",
            )
            app = create_app(root=root, models_root=root, autotrainer_status=status)
            client = app.test_client()
            out = client.get("/api/autotrainer")
            self.assertEqual(out.status_code, 200)
            body = out.get_json()
            self.assertTrue(body["needs_chat_restart"])
            self.assertEqual(body["state"], "promoted")


if __name__ == "__main__":
    unittest.main()
