"""Host-side npzviewer: .npy / .npz / .npx inspect + Flask API."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from npzviewer import create_app
from paths import chat_open_url, checkpoint_weights_relpath, viewer_open_url
from tools.npz_inspect import (
    classify_key,
    infer_architecture,
    manifest,
    neuron_partners,
    open_weight_file,
    resolve_weight_path,
    sidecar_mismatches,
    tensor_detail,
    tensor_stats,
)


def _gptish(dirpath: Path) -> Path:
    arrays = {
        "token_embedding": np.arange(12, dtype=np.float32).reshape(4, 3),
        "layer_0.qkv_proj": np.ones((3, 9), dtype=np.float32) * 0.1,
        "layer_0.attn_out_proj": np.eye(3, dtype=np.float32) * 0.2,
        "layer_0.ln1_gamma": np.ones((3,), dtype=np.float32),
        "layer_0.mlp_expand": np.ones((3, 12), dtype=np.float32) * 0.05,
        "layer_0.mlp_contract": np.ones((12, 3), dtype=np.float32) * 0.04,
        "lm_head": np.arange(12, dtype=np.float32).reshape(3, 4),
        "lm_head_bias": np.zeros((4,), dtype=np.float32),
    }
    path = Path(dirpath) / "weights.npz"
    np.savez(path, **arrays)
    return path


class InspectTests(unittest.TestCase):
    def test_npy_npz_npx_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npy = root / "tokens.npy"
            np.save(npy, np.arange(6, dtype=np.int32))
            with open_weight_file(npy) as archive:
                self.assertEqual(archive.format, "npy")
                self.assertEqual(list(archive.arrays), ["tokens"])
                self.assertEqual(archive.arrays["tokens"].shape, (6,))

            npz = root / "bundle.npz"
            np.savez(npz, a=np.array([[1.0, 2.0]], dtype=np.float32))
            with open_weight_file(npz) as archive:
                self.assertIn("a", archive.arrays)
                self.assertEqual(archive.arrays["a"].shape, (1, 2))

            npx = root / "bundle.npx"
            npx.write_bytes(npz.read_bytes())
            with open_weight_file(npx) as archive:
                self.assertEqual(archive.format, "npx")
                self.assertEqual(archive.arrays["a"].tolist(), [[1.0, 2.0]])

    def test_classify_and_architecture(self):
        role = classify_key("layer_2.qkv_proj")
        self.assertEqual(role["family"], "attn")
        self.assertEqual(role["layer"], 2)
        self.assertIn("QKV", role["title"])
        arch = infer_architecture(
            {
                "token_embedding": (100, 8),
                "lm_head": (8, 100),
                "layer_0.qkv_proj": (8, 24),
                "layer_1.mlp_expand": (8, 32),
                "layer_0.ln1_gamma": (8,),
                "layer_1.ln1_gamma": (8,),
            }
        )
        self.assertEqual(arch["kind"], "gpt")
        self.assertEqual(arch["vocab_size"], 100)
        self.assertEqual(arch["embedding_dim"], 8)
        self.assertEqual(arch["num_layers"], 2)
        self.assertEqual(arch["norm_type"], "rmsnorm")
        self.assertEqual(arch["pos_encoding"], "rope")
        self.assertTrue(arch["tie_embeddings"])
        self.assertEqual(arch["mlp_multiplier"], 4.0)
        self.assertEqual(
            sidecar_mismatches(arch, {"config": {"num_layers": 6, "embedding_dim": 8}}),
            ["L: weights 2 vs config 6"],
        )

    def test_stats_and_dead_flag(self):
        stats = tensor_stats(np.zeros((4, 4), dtype=np.float32))
        self.assertEqual(stats["std"], 0.0)
        self.assertEqual(stats["abs_max"], 0.0)
        role = classify_key("layer_0.qkv_proj")
        from tools.npz_inspect import health_flags

        self.assertIn("dead", health_flags("layer_0.qkv_proj", stats, role))

    def test_manifest_and_slice(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _gptish(tmp)
            info = manifest(path)
            keys = {row["key"] for row in info["tensors"]}
            self.assertIn("token_embedding", keys)
            self.assertEqual(info["architecture"]["embedding_dim"], 3)
            self.assertEqual(info["architecture"]["num_layers"], 1)
            self.assertTrue(info["architecture"]["tie_embeddings"])
            self.assertEqual(len(info["layers"]), 1)
            detail = tensor_detail(path, "token_embedding", rows=2, cols=2)
            self.assertEqual(detail["table"]["values"][0][0], 0.0)
            self.assertEqual(len(detail["histogram"]["counts"]), 48)
            self.assertEqual(detail["heatmap"]["rows"], 4)
            self.assertEqual(detail["axes"]["row_kind"], "token")
            self.assertEqual(detail["axes"]["col_kind"], "residual")
            partners = neuron_partners(path, "token_embedding", axis="row", index=1, k=2)
            self.assertEqual(partners["kind"], "token")
            self.assertEqual(partners["index"], 1)
            self.assertEqual(partners["partners"][0]["index"], 2)
            self.assertEqual(partners["partners"][0]["weight"], 5.0)

    def test_resolve_rejects_outside_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inside = root / "ok.npy"
            np.save(inside, np.array([1], dtype=np.float32))
            resolved = resolve_weight_path("ok.npy", root)
            self.assertEqual(resolved, inside.resolve())
            with self.assertRaises(ValueError):
                resolve_weight_path("/etc/passwd", root)


class ViewerApiTests(unittest.TestCase):
    def test_manifest_and_tensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = _gptish(root)
            (root / "config.json").write_text(
                json.dumps({"model": {"name": "Toy", "num_heads": 1, "embedding_dim": 3}}),
                encoding="utf-8",
            )
            rel = "weights.npz"
            app = create_app(root=root, initial=rel)
            client = app.test_client()
            home = client.get("/")
            self.assertEqual(home.status_code, 200)
            self.assertIn(b"NPZ Viewer", home.data)
            listed = client.get("/api/files")
            self.assertEqual(listed.status_code, 200)
            opened = client.get("/api/manifest", query_string={"path": rel})
            self.assertEqual(opened.status_code, 200)
            body = opened.get_json()
            self.assertEqual(body["sidecar"]["model_name"], "Toy")
            self.assertEqual(body["architecture"]["vocab_size"], 4)
            neuron = client.get(
                "/api/neuron",
                query_string={"path": rel, "key": "token_embedding", "axis": "row", "index": 0, "k": 2},
            )
            self.assertEqual(neuron.status_code, 200)
            self.assertEqual(neuron.get_json()["partners"][0]["index"], 2)
            tensor = client.get("/api/tensor", query_string={"path": rel, "key": "lm_head_bias"})
            self.assertEqual(tensor.status_code, 200)
            self.assertEqual(tensor.get_json()["family"], "head")
            missing = client.get("/api/tensor", query_string={"path": rel, "key": "nope"})
            self.assertEqual(missing.status_code, 404)
            outside = client.get("/api/manifest", query_string={"path": "../secret.npz"})
            self.assertIn(outside.status_code, (400, 404))
            home_q = client.get("/", query_string={"path": rel})
            self.assertEqual(home_q.status_code, 200)
            self.assertIn(b">Chat</a>", home_q.data)
            self.assertIn(b"7860", home_q.data)
            self.assertIn(rel.encode(), home_q.data)


class PeerNavTests(unittest.TestCase):
    def test_viewer_url_uses_same_weights(self):
        url = viewer_open_url("http://127.0.0.1:7861", "output/checkpoints/chat_facts_v6")
        self.assertTrue(url.startswith("http://127.0.0.1:7861/?path="))
        self.assertIn("chat_facts_v6/weights.npz", url)
        self.assertEqual(
            checkpoint_weights_relpath("output/checkpoints/chat_facts_v6/weights.npz"),
            checkpoint_weights_relpath("output/checkpoints/chat_facts_v6"),
        )
        chat = chat_open_url(
            "http://127.0.0.1:7860",
            "output/checkpoints/chat_facts_v6/weights.npz",
        )
        self.assertTrue(chat.startswith("http://127.0.0.1:7860/?checkpoint="))
        self.assertIn("chat_facts_v6", chat)
        self.assertNotIn("weights.npz", chat)


if __name__ == "__main__":
    unittest.main()
