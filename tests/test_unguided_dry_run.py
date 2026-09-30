"""Kernel and daemon --dry-run must not touch Metal."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import autotrainer_daemon
import unguided_trainer
from training.unguided.session import refuse_checkpoint_dir, refuse_vocab_mismatch


class DryRunTests(unittest.TestCase):
    def test_quicktest_policy_dry_run(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = unguided_trainer.main([
                "--config", str(_ROOT / "legacy" / "setup" / "quicktest_config.json"),
                "--policy", str(_ROOT / "legacy" / "setup" / "unguided_quicktest_policy.json"),
                "--dry-run",
            ])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("unguided_quicktest", out)
        self.assertIn("max_steps:     8", out)
        self.assertIn("No Metal init", out)

    def test_kernel_dry_run_prints_plan(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = unguided_trainer.main([
                "--config", str(_ROOT / "legacy" / "setup" / "chat_facts_v7_config.json"),
                "--policy", str(_ROOT / "legacy" / "setup" / "unguided_v7_policy.json"),
                "--dry-run",
            ])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("UNGUIDED DRY-RUN", out)
        self.assertIn("No Metal init", out)
        self.assertIn("eval_every", out)
        self.assertIn("log_every:", out)
        self.assertIn("probe_on_stop:", out)
        self.assertIn("probe_n:", out)
        self.assertIn("No Metal init", out)

    def test_name_flags_override_dirs(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = unguided_trainer.main([
                "--config", str(_ROOT / "legacy" / "setup" / "chat_facts_v7_config.json"),
                "--policy", str(_ROOT / "legacy" / "setup" / "unguided_v7_policy.json"),
                "--name", "unguided_v7_log1",
                "--dry-run",
            ])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("run_name:      unguided_v7_log1", out)
        self.assertIn("output/runs/unguided_v7_log1", out)
        self.assertIn("output/checkpoints/unguided_v7_log1", out)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = unguided_trainer.main([
                "--config", str(_ROOT / "legacy" / "setup" / "chat_facts_v7_config.json"),
                "--policy", str(_ROOT / "legacy" / "setup" / "unguided_v7_policy.json"),
                "--run-name", "run_only",
                "--checkpoint-dir", "output/checkpoints/ckpt_only",
                "--dry-run",
            ])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("run_name:      run_only", out)
        self.assertIn("output/checkpoints/ckpt_only", out)

    def test_log_every_cli_overrides_policy(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = unguided_trainer.main([
                "--config", str(_ROOT / "legacy" / "setup" / "quicktest_config.json"),
                "--policy", str(_ROOT / "legacy" / "setup" / "unguided_quicktest_policy.json"),
                "--log-every", "1",
                "--dry-run",
            ])
        self.assertEqual(rc, 0)
        self.assertIn("log_every:     1", buf.getvalue())

    def test_daemon_dry_run_idle_when_queue_short(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "retrain.jsonl"
            log.write_text("", encoding="utf-8")
            status = Path(tmp) / "status.json"
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = autotrainer_daemon.main([
                    "--retrain-log", str(log),
                    "--status", str(status),
                    "--policy", str(_ROOT / "legacy" / "setup" / "unguided_v7_policy.json"),
                    "--recipe", str(_ROOT / "legacy" / "setup" / "chat_facts_v7_config.json"),
                    "--once",
                    "--dry-run",
                ])
            self.assertEqual(rc, 0)
            self.assertIn("no kernel", buf.getvalue())

    def test_refuse_v6_dir_and_vocab_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            v6 = Path(tmp) / "chat_facts_v6"
            v6.mkdir()
            with self.assertRaises(RuntimeError):
                refuse_checkpoint_dir(v6, {"hard_limits": {"refuse_existing_v6_v4_dir": True}})
            fresh = Path(tmp) / "unguided_v7"
            fresh.mkdir()
            (fresh / "weights.npz").write_bytes(b"x")
            with self.assertRaises(RuntimeError):
                refuse_checkpoint_dir(fresh, {"hard_limits": {}})
            meta = fresh
            (meta / "unguided_meta.json").write_text(
                json.dumps({"vocab_fingerprint": "aaa"}),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                refuse_vocab_mismatch(
                    meta,
                    "bbb",
                    {"hard_limits": {"refuse_resume_on_vocab_mismatch": True}},
                )

    def test_dry_run_does_not_import_mlx(self):
        with mock.patch.dict(sys.modules, {"mlx": None, "mlx.core": None}):
            rc = unguided_trainer.main([
                "--config", str(_ROOT / "legacy" / "setup" / "chat_facts_v7_config.json"),
                "--policy", str(_ROOT / "legacy" / "setup" / "unguided_v7_policy.json"),
                "--dry-run",
            ])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
