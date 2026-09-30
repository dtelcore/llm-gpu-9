"""ChatSession router turns without loading a checkpoint."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.cabinet_index import CabinetIndex
from training.chat_session import ChatSession, TurnResult, add_session_args


class FakeTok:
    def __init__(self, reply="FAKE GENERATE"):
        self.reply = reply

    def encode(self, text):
        return [1, 2, 3]

    def decode(self, ids):
        return self.reply


class FakeModel:
    def generate(self, *args, **kwargs):
        raise AssertionError("generate should not run for calc/learned/commands")


def _session(**kwargs):
    defaults = dict(
        model=FakeModel(),
        tokenizer=FakeTok(),
        gpt_config=SimpleNamespace(name="Chat C=512 L=6 T=256", vocab_size=10, max_len=256),
        args=Namespace(checkpoint="ck"),
        chat_mode=True,
        router_on=True,
        search_enabled=False,
        learned_path="unused.jsonl",
        cabinet=CabinetIndex(),
        temperature=0.7,
        top_k=32,
        top_p=0.9,
        system="sys",
        stop_strings=["User:"],
        max_new_tokens=80,
        tracer=SimpleNamespace(any_enabled=False),
        rng=None,
        use_kv_cache=True,
        use_cuda_graph=False,
        trace_enabled=False,
    )
    defaults.update(kwargs)
    return ChatSession(**defaults)


class ChatSessionTests(unittest.TestCase):
    def test_calc_does_not_generate(self):
        s = _session()
        r = s.turn("2+2")
        self.assertEqual(r.kind, "calc")
        self.assertEqual(r.text, "4")

    def test_learned_replay(self):
        cab = CabinetIndex()
        cab.add("tell me about python", "Python is a language.", source="learned")
        s = _session(cabinet=cab)
        r = s.turn("tell me about python")
        self.assertEqual(r.kind, "cabinet")
        self.assertEqual(r.text, "Python is a language.")
        self.assertEqual(len(s.history), 2)

    def test_clear_command(self):
        s = _session()
        s.turn("2+2")
        r = s.turn(":clear")
        self.assertEqual(r.kind, "command")
        self.assertEqual(s.history, [])
        self.assertEqual(s.last_entities, [])

    def test_generate_sets_entities_and_related_miss(self):
        class GenModel:
            def generate(self, prompt_ids, **kwargs):
                return list(prompt_ids) + [9]

        cab = CabinetIndex()
        cab.add("What is the capital of France?", "The capital of France is Paris.")
        s = _session(
            model=GenModel(),
            tokenizer=FakeTok("The capital of France is Paris."),
            cabinet=cab,
            search_enabled=True,
        )
        r = s.turn("What is the capital of France?")
        self.assertEqual(r.kind, "cabinet")
        self.assertEqual(r.detail, "generate")
        self.assertEqual(r.classification, "MATCH")
        self.assertEqual(r.text, "The capital of France is Paris.")
        self.assertIn("france", s.last_entities)
        self.assertTrue(r.related == [] or isinstance(r.related, list))
        miss = s.turn("where is paris?")
        self.assertEqual(miss.kind, "miss")
        self.assertEqual(miss.detail, "related_miss")
        self.assertIn("What is the capital of France?", miss.related)
        rel = s.turn(":related")
        self.assertEqual(rel.kind, "command")
        self.assertIn("France", rel.text)

    def test_generate_mismatch_is_flagged_not_substituted(self):
        class GenModel:
            def generate(self, prompt_ids, **kwargs):
                return list(prompt_ids) + [9]

        cab = CabinetIndex()
        cab.add(
            "What did William Cubitt invent?",
            "William Cubitt is credited with inventing penal treadmill.",
        )
        cab.add(
            "What did Ernesto Schiaparelli invent?",
            "Ernesto Schiaparelli is credited with inventing Plough.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            retrain = Path(tmp) / "retrain.jsonl"
            diag = Path(tmp) / "diag.jsonl"
            s = _session(
                model=GenModel(),
                tokenizer=FakeTok("William Cubitt is credited with inventing penal treadmill."),
                cabinet=cab,
                retrain_path=str(retrain),
                diagnostics_path=str(diag),
            )
            r = s.turn("What did Ernesto Schiaparelli invent?")
            self.assertEqual(r.text, "William Cubitt is credited with inventing penal treadmill.")
            self.assertEqual(r.detail, "generate_target_mismatch")
            self.assertEqual(r.classification, "BINDING_ENTITY_SWAP")
            recs = [json.loads(ln) for ln in retrain.read_text(encoding="utf-8").splitlines() if ln]
            self.assertEqual(recs[0]["detail"], "generate_target_mismatch")
            events = [json.loads(ln) for ln in diag.read_text(encoding="utf-8").splitlines() if ln]
            self.assertEqual(events[0]["diagnostic_classification"], "BINDING_ENTITY_SWAP")

    def test_add_session_args_accepts_facts(self):
        parser = __import__("argparse").ArgumentParser()
        add_session_args(parser)
        ns = parser.parse_args(["--checkpoint", "ck", "--no-search"])
        self.assertTrue(ns.no_search)
        self.assertTrue(ns.cabinet_retrain_log)


class WebuiApiTests(unittest.TestCase):
    def test_chat_and_clear(self):
        from webui import create_app

        class Fake:
            def status(self):
                return {
                    "checkpoint": "ck",
                    "model": "Chat C=512 L=6 T=256",
                    "vocab_size": 10,
                    "max_len": 256,
                    "chat": True,
                    "router": True,
                    "search": False,
                    "cabinet": 0,
                    "system": "sys",
                    "learned": "l",
                }

            def turn(self, message):
                return TurnResult(text="4", kind="calc", detail="calc")

            def clear(self):
                self.cleared = True

        fake = Fake()
        app = create_app(fake)
        client = app.test_client()
        home = client.get("/")
        self.assertEqual(home.status_code, 200)
        self.assertIn(b"Apple MLX", home.data)
        self.assertIn(b">Weights</a>", home.data)
        self.assertIn(b"7861/?path=", home.data)
        self.assertIn(b"ck/weights.npz", home.data)
        chat = client.post("/api/chat", json={"message": "2+2"})
        self.assertEqual(chat.status_code, 200)
        self.assertEqual(chat.get_json()["reply"], "4")
        self.assertEqual(chat.get_json().get("related"), [])
        self.assertIn("classification", chat.get_json())
        self.assertIn("match_type", chat.get_json())
        self.assertEqual(client.post("/api/clear").status_code, 200)
        empty = client.post("/api/chat", json={"message": "  "})
        self.assertEqual(empty.status_code, 400)


if __name__ == "__main__":
    unittest.main()
