"""Phase 1 sequential layer streaming: Metal keys, logits/grad parity, smoke step."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.parity._common import B, C, H, T, V, CudaTestCase, assert_close


def _cfg(num_layers: int = 2, **overrides):
    from model.config import GPTConfig

    base = {
        "name": "stream",
        "vocab_size": V,
        "max_len": T,
        "embedding_dim": C,
        "num_heads": H,
        "num_layers": int(num_layers),
        "dropout_prob": 0.0,
        "tie_embeddings": True,
        "norm_type": "rmsnorm",
        "pos_encoding": "rope",
        "gradient_checkpointing": False,
        "layer_strategy": "resident",
        "residual_scale": True,
    }
    base.update(overrides)
    return GPTConfig(base)


def _copy_params(src, dst) -> None:
    for key, val in src.weights.items():
        if key not in dst.weights:
            continue
        if key == "lm_head" and dst.tie_embeddings:
            continue
        dst.weights[key] = np.array(val, copy=True)
    if dst.tie_embeddings:
        dst.weights["lm_head"] = dst.weights["token_embedding"].T
    for key, val in src.biases.items():
        if key in dst.biases:
            dst.biases[key] = np.array(val, copy=True)
    dst.upload_to_device()


def _to_np(tensor) -> np.ndarray:
    if hasattr(tensor, "get"):
        return np.ascontiguousarray(tensor.get(), dtype=np.float32)
    return np.ascontiguousarray(tensor, dtype=np.float32)


class TestLayerStream(CudaTestCase):
    def setUp(self) -> None:
        import model.gpt as gpt_mod

        gpt_mod._GPU_TRAINING = True
        gpt_mod._USE_GPU_ATTENTION = True

    def test_load_unload_metal_keys(self) -> None:
        from model.cuda.env import check_memory
        from model.weights import ModelParameters

        cfg = _cfg(num_layers=2, layer_strategy="stream")
        params = ModelParameters(cfg, seed=5)
        host_tok = np.array(params.weights["token_embedding"], copy=True)
        resident = set(params.always_resident_keys())
        metal = set(params.device_weights) | set(params.device_biases)
        self.assertTrue(metal <= resident | {"lm_head"})
        self.assertFalse(any(k.startswith("layer_0.") for k in params.device_weights))

        params.load_layer(0)
        check_memory("load_layer_0")
        for name in params.layer_keys(0):
            self.assertTrue(
                name in params.device_weights or name in params.device_biases,
                msg=name,
            )
        self.assertFalse(any(k.startswith("layer_1.") for k in params.device_weights))
        self.assertFalse(any(k.startswith("layer_1.") for k in params.device_biases))

        params.unload_layer(0)
        check_memory("unload_layer_0")
        self.assertFalse(any(k.startswith("layer_0.") for k in params.device_weights))
        self.assertFalse(any(k.startswith("layer_0.") for k in params.device_biases))
        np.testing.assert_array_equal(params.weights["token_embedding"], host_tok)
        self.assertIsNone(params._loaded_layer)

        params.load_layer(1)
        check_memory("load_layer_1")
        self.assertTrue(
            any(k.startswith("layer_1.") for k in params.device_weights)
            or any(k.startswith("layer_1.") for k in params.device_biases)
        )
        params.unload_layer(1)
        check_memory("unload_layer_1")

    def test_logits_grads_match_resident_l2(self) -> None:
        self._assert_stream_matches_resident(num_layers=2)

    def test_logits_grads_match_resident_l4(self) -> None:
        self._assert_stream_matches_resident(num_layers=4)

    def _assert_stream_matches_resident(self, num_layers: int) -> None:
        from model.gpt import GPTModel
        from model.weights import ModelParameters
        from training.loss import softmax_cross_entropy_batch_gpu
        cfg_r = _cfg(num_layers=num_layers, layer_strategy="resident")
        cfg_s = _cfg(num_layers=num_layers, layer_strategy="stream")
        params_r = ModelParameters(cfg_r, seed=11)
        params_s = ModelParameters(cfg_s, seed=11)
        _copy_params(params_r, params_s)

        rng = np.random.default_rng(9)
        xs = rng.integers(0, V, size=(B, T), dtype=np.int32)
        ys = rng.integers(0, V, size=(B, T), dtype=np.int32)

        m_r = GPTModel(cfg_r, params_r)
        m_s = GPTModel(cfg_s, params_s)
        logits_r, cache_r = m_r.forward_batch(xs)
        logits_s, cache_s = m_s.forward_batch(xs)
        assert_close("logits", logits_s, logits_r, rtol=1e-4, atol=1e-5)

        _, d_r = softmax_cross_entropy_batch_gpu(cache_r["logits_d"], ys)
        _, d_s = softmax_cross_entropy_batch_gpu(cache_s["logits_d"], ys)
        g_r = m_r.backward_batch_gpu(cache_r, d_r.reshape(-1, V))
        g_s = m_s.backward_batch_gpu(cache_s, d_s.reshape(-1, V))
        self.assertEqual(set(g_r), set(g_s))
        for key in g_r:
            assert_close(f"grad.{key}", _to_np(g_s[key]), _to_np(g_r[key]), rtol=1e-3, atol=1e-4)

    def test_smoke_l8_stream_step_under_cap(self) -> None:
        from model.gpt import GPTModel
        from model.cuda.env import check_memory
        from model.weights import ModelParameters
        from training.gpu_optimizer import AdamWGPU
        from training.loss import softmax_cross_entropy_batch_gpu
        cfg = _cfg(num_layers=8, layer_strategy="stream")
        params = ModelParameters(cfg, seed=3)
        orig_load = params.load_layer
        orig_unload = params.unload_layer

        def load_checked(layer: int) -> None:
            orig_load(layer)
            check_memory(f"load_layer_{layer}")

        def unload_checked(layer=None) -> None:
            orig_unload(layer)
            check_memory(f"unload_layer_{layer}")

        params.load_layer = load_checked
        params.unload_layer = unload_checked

        model = GPTModel(cfg, params)
        rng = np.random.default_rng(4)
        xs = rng.integers(0, V, size=(B, T), dtype=np.int32)
        ys = rng.integers(0, V, size=(B, T), dtype=np.int32)
        opt = AdamWGPU(params, learning_rate=1e-3, gradient_clip=1.0)
        logits, cache = model.forward_batch(xs)
        self.assertTrue(np.isfinite(logits).all())
        _, dlogits = softmax_cross_entropy_batch_gpu(cache["logits_d"], ys)
        grads = model.backward_batch_gpu(cache, dlogits.reshape(-1, V))
        self.assertTrue(all(isinstance(v, np.ndarray) for v in grads.values()))
        opt.clip_grads_(grads)
        opt.step(grads)
        check_memory("after_stream_step")

        ids = model.generate([1, 2, 3], max_new_tokens=2, temperature=0.0, use_kv_cache=True)
        self.assertGreaterEqual(len(ids), 4)
        check_memory("after_stream_generate")


if __name__ == "__main__":
    import unittest
    unittest.main()
