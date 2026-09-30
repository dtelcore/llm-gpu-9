"""Supervisor refuses spawn when Metal is already held. No model.gpt import."""

from __future__ import annotations

import inspect
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import training.unguided.supervisor as supervisor


class SupervisorTests(unittest.TestCase):
    def test_source_does_not_import_gpt(self):
        source = inspect.getsource(supervisor)
        self.assertNotIn("from model.gpt", source)
        self.assertNotIn("import model.gpt", source)
        self.assertNotIn("from model ", source)

    def test_refuse_when_status_busy(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = Path(tmp) / "status.json"
            holder = Path(tmp) / "holder.json"
            status.write_text(json.dumps({"state": "training", "metal_busy": True}), encoding="utf-8")
            reason = supervisor.refuse_spawn_reason(
                status_path=status, holder_path=holder, require_idle_app=True,
            )
            self.assertEqual(reason, "metal_busy_status")

    def test_refuse_when_app_holder_alive(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = Path(tmp) / "status.json"
            holder = Path(tmp) / "holder.json"
            status.write_text(json.dumps({"state": "idle"}), encoding="utf-8")
            holder.write_text(
                json.dumps({"holder": "app", "pid": os.getpid(), "checkpoint": "x"}),
                encoding="utf-8",
            )
            reason = supervisor.refuse_spawn_reason(
                status_path=status, holder_path=holder, require_idle_app=True,
            )
            self.assertEqual(reason, "app_holds_metal")

    def test_allow_when_idle(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = Path(tmp) / "status.json"
            holder = Path(tmp) / "holder.json"
            status.write_text(json.dumps({"state": "idle"}), encoding="utf-8")
            holder.write_text(json.dumps({"holder": "app", "pid": 999999999}), encoding="utf-8")
            reason = supervisor.refuse_spawn_reason(
                status_path=status, holder_path=holder, require_idle_app=True,
            )
            self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
