"""Host-only train monitor: parse plotter-style logs, no Metal."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app import create_app
from app.trainmon import create_app as create_trainmon, series_payload


def _write_train_log(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "2026-09-13 15:00:00 | INFO | llm_gpu | [train] step=10/500 epoch=1 loss=3.2100 ppl=24.78 tok_s=120 lr=0.0003",
                "2026-09-13 15:00:05 | INFO | llm_gpu | [train] step=20/500 epoch=1 loss=2.9800 ppl=19.69 tok_s=118 lr=0.0003",
                "2026-09-13 15:00:10 | INFO | llm_gpu | [train] step=30/500 epoch=1 loss=2.7400 ppl=15.49 tok_s=121 lr=0.0003",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


class SeriesPayloadTests(unittest.TestCase):
    def test_same_step_val_line_keeps_train_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "training_demo.log"
            log.write_text(
                "\n".join(
                    [
                        "[train] step=10/20 loss=2.1000 ppl=8.2 tok_s=99",
                        "[train] step=20/20 loss=2.0000 ppl=7.4 tok_s=100",
                        "[train] step=20/20 val_loss=2.5000 val_ppl=12.2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            payload = series_payload(log, keep_short=True)
            self.assertEqual(payload["last"]["loss"], 2.0)
            self.assertEqual(payload["last"]["tok_s"], 100.0)
            self.assertEqual(payload["last"]["val_loss"], 2.5)

    def test_plotter_lines_and_volatility(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "unguided_demo.log"
            _write_train_log(log)
            payload = series_payload(log, keep_short=True)
            self.assertEqual(payload["source"], "log")
            self.assertEqual(payload["steps"], [10, 20, 30])
            self.assertEqual(payload["loss"][0], 3.21)
            self.assertEqual(payload["last"]["step"], 30)
            self.assertEqual(payload["last"]["total"], 500)
            self.assertEqual(len(payload["volatility"]), 3)
            self.assertIsNotNone(payload["volatility"][-1])

    def test_decisions_fallback_when_log_has_no_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "logs" / "unguided_demo.log"
            log.parent.mkdir()
            log.write_text("session ready\n", encoding="utf-8")
            decisions = root / "runs" / "demo" / "decisions.jsonl"
            decisions.parent.mkdir(parents=True)
            decisions.write_text(
                json.dumps({"step": 50, "action": "continue", "reason": "ok", "val_loss": 2.4})
                + "\n"
                + json.dumps({"step": 100, "action": "promote", "reason": "best", "val_loss": 2.1})
                + "\n",
                encoding="utf-8",
            )
            payload = series_payload(log, keep_short=True, runs_root=root / "runs")
            self.assertEqual(payload["source"], "decisions")
            self.assertEqual(payload["steps"], [50, 100])
            self.assertEqual(payload["val_loss"][-1], 2.1)
            self.assertEqual(payload["decisions"][-1]["action"], "promote")


class TrainmonAppTests(unittest.TestCase):
    def test_logs_and_series_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / "logs"
            logs.mkdir()
            _write_train_log(logs / "training_demo.log")
            app = create_trainmon(log_dir=logs)
            client = app.test_client()
            home = client.get("/")
            self.assertEqual(home.status_code, 200)
            self.assertIn(b"Train monitor", home.data)
            listing = client.get("/api/logs")
            self.assertEqual(listing.status_code, 200)
            names = [row["name"] for row in listing.get_json()["logs"]]
            self.assertEqual(names, ["training_demo.log"])
            series = client.get("/api/series", query_string={"log": "training_demo.log"})
            self.assertEqual(series.status_code, 200)
            body = series.get_json()
            self.assertEqual(body["steps"], [10, 20, 30])
            self.assertAlmostEqual(body["tok_s"][0], 120.0)

    def test_prefers_the_live_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / "logs"
            logs.mkdir()
            now = time.time()
            old = logs / "unguided_finished.log"
            _write_train_log(old)
            os.utime(old, (now - 3600, now - 3600))
            live = logs / "training_english_tinystories_c512_l6.log"
            _write_train_log(live)
            os.utime(live, (now, now))
            app = create_trainmon(log_dir=logs)
            client = app.test_client()
            listing = client.get("/api/logs").get_json()
            self.assertEqual(listing["preferred"], live.name)
            self.assertTrue(listing["logs"][0]["live"])
            series = client.get("/api/series").get_json()
            self.assertEqual(series["log"], live.name)

    def test_dropdown_keeps_five_newest_run_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / "logs"
            logs.mkdir()
            now = time.time()
            for i in range(7):
                path = logs / f"training_run{i}.log"
                _write_train_log(path)
                os.utime(path, (now - (7 - i) * 120, now - (7 - i) * 120))
            aggregate = logs / "training.log"
            _write_train_log(aggregate)
            os.utime(aggregate, (now, now))
            generate = logs / "generate_story.log"
            _write_train_log(generate)
            os.utime(generate, (now, now))
            live = logs / "training_smoke.log"
            _write_train_log(live)
            os.utime(live, (now, now))
            listing = create_trainmon(log_dir=logs).test_client().get("/api/logs").get_json()
            names = [row["name"] for row in listing["logs"]]
            self.assertEqual(len(names), 5)
            self.assertEqual(names[0], "training_smoke.log")
            self.assertEqual(listing["preferred"], "training_smoke.log")
            self.assertNotIn("training.log", names)
            self.assertNotIn("generate_story.log", names)
            self.assertNotIn("training_run0.log", names)
            self.assertNotIn("training_run1.log", names)

    def test_clock_uses_the_train_line_not_the_file_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "training_smoke.log"
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log.write_text(
                "\n".join(
                    [
                        "2020-01-01 00:00:00 | INFO | llm_gpu | [train] step=400000/400000 epoch=1 loss=1.0000 ppl=2.72 tok_s=11000 elapsed_s=140000 eta_s=0 step_ms=360",
                        f"{stamp} | INFO | llm_gpu | [train] step=19/2000 epoch=1 loss=4.5000 ppl=90 tok_s=1500 elapsed_s=31.00 eta_s=400.00 step_ms=4200",
                        f"{stamp} | INFO | llm_gpu | [train] step=20/2000 epoch=1 loss=4.4000 ppl=81.5 tok_s=1550 lr=4e-05 elapsed_s=42.00 eta_s=380.00 step_ms=4200",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            payload = series_payload(log, keep_short=True)
            self.assertAlmostEqual(payload["last"]["elapsed_s"], 42.0, delta=5)
            self.assertAlmostEqual(payload["last"]["eta_s"], 380.0, delta=5)
            self.assertAlmostEqual(payload["last"]["sec_per_step"], 4.2, delta=0.05)
            self.assertEqual(payload["last"]["step"], 20)
            self.assertEqual(payload["steps"], [19, 20])

    def test_rejects_path_outside_log_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / "logs"
            logs.mkdir()
            _write_train_log(logs / "ok.log")
            app = create_trainmon(log_dir=logs)
            client = app.test_client()
            escaped = client.get("/api/series", query_string={"log": "../ok.log"})
            # basename is still ok.log inside the log dir
            self.assertEqual(escaped.status_code, 200)
            missing = client.get("/api/series", query_string={"log": "nope.log"})
            self.assertEqual(missing.status_code, 404)


class AppTrainTabTests(unittest.TestCase):
    def test_shell_exposes_train_tab(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / "logs"
            logs.mkdir()
            _write_train_log(logs / "live.log")
            app = create_app(root=Path(tmp), models_root=Path(tmp), log_dir=logs)
            client = app.test_client()
            home = client.get("/?view=train")
            self.assertEqual(home.status_code, 200)
            self.assertIn(b'id="tab-train"', home.data)
            pane = client.get("/train/")
            self.assertEqual(pane.status_code, 200)
            self.assertIn(b"Train monitor", pane.data)
            self.assertIn(b"Follow live", pane.data)
            series = client.get("/train/api/series", query_string={"log": "live.log"})
            self.assertEqual(series.status_code, 200)
            self.assertEqual(series.get_json()["last"]["step"], 30)


if __name__ == "__main__":
    unittest.main()
