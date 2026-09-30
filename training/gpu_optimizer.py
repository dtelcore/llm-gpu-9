"""
training/gpu_optimizer.py

AdamW on MLX-resident weight mirrors. Host NumPy copies sync only at checkpoint.
Streaming: layer m/v stay on host; device copies exist only while that layer is loaded.
"""

import math
from typing import Dict, Iterable, Optional

import numpy as np

from model.cuda import ops as cuda_ops
from model.weights import ModelParameters


def _is_device(g) -> bool:
    return hasattr(g, "gpudata") or type(g).__name__ in ("DeviceArray", "GPUArray")


class AdamWGPU:
    """AdamW on device weight mirrors (ModelParameters.device_weights/biases)."""

    def __init__(
        self,
        params: ModelParameters,
        learning_rate: float,
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        warmup_steps: int = 0,
        gradient_clip: float = 1.0,
        total_steps: int = 0,
        min_lr_ratio: float = 0.1,
    ) -> None:
        self.params = params
        self.base_lr = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.warmup_steps = max(0, warmup_steps)
        self.gradient_clip = gradient_clip
        self.total_steps = max(0, int(total_steps))
        self.min_lr_ratio = float(min_lr_ratio)
        self.t = 0
        self._streaming = bool(params.streaming())

        all_keys = list(params.trainable_param_names())
        self.m: Dict[str, object] = {}
        self.v: Dict[str, object] = {}
        resident = set(params.always_resident_keys()) if self._streaming else None
        for key in all_keys:
            if key in params.weights:
                shape = params.weights[key].shape
            elif key in params.biases:
                shape = params.biases[key].shape
            else:
                continue
            z = np.zeros(shape, dtype=np.float32)
            on_device = (not self._streaming) or (resident is not None and key in resident)
            if on_device:
                self.m[key] = cuda_ops.to_device(z)
                self.v[key] = cuda_ops.to_device(z)
            else:
                self.m[key] = z
                self.v[key] = z.copy()

        self._batch_keys = [k for k in all_keys if k in self.m]
        self._m_d: Dict[str, object] = {}
        self._v_d: Dict[str, object] = {}
        self._opt_loaded_layer: Optional[int] = None

    def current_lr(self) -> float:
        if self.warmup_steps > 0 and self.t < self.warmup_steps:
            return self.base_lr * (self.t + 1) / self.warmup_steps
        if self.total_steps <= self.warmup_steps:
            return self.base_lr
        min_lr = self.base_lr * self.min_lr_ratio
        denom = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, (self.t - self.warmup_steps) / denom))
        return min_lr + 0.5 * (self.base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    def clip_grads_(self, grads: Dict) -> float:
        if not grads:
            return 0.0
        sample = next(iter(grads.values()))
        if _is_device(sample):
            total_sq = cuda_ops.grad_global_norm_sq(grads)
            global_norm = float(np.sqrt(total_sq))
            if self.gradient_clip and global_norm > self.gradient_clip:
                scale = self.gradient_clip / (global_norm + 1e-6)
                for key in grads:
                    cuda_ops.scal_mul(grads[key], scale)
            return global_norm
        total_sq = sum(float(np.sum(np.asarray(g, dtype=np.float64) ** 2)) for g in grads.values())
        global_norm = float(np.sqrt(total_sq))
        if self.gradient_clip and global_norm > self.gradient_clip:
            scale = self.gradient_clip / (global_norm + 1e-6)
            for g in grads.values():
                g *= np.float32(scale)
        return global_norm

    def _get_weight(self, key: str):
        if key in self.params.device_weights:
            return self.params.device_weights[key]
        return self.params.device_biases[key]

    def begin_step(self) -> None:
        """Increment t once per optimizer step (streaming uses this before step_layer)."""
        self.t += 1

    def _moment_pair(self, key: str):
        m, v = self.m[key], self.v[key]
        if _is_device(m):
            return m, v
        md = self._m_d.get(key)
        vd = self._v_d.get(key)
        if md is None:
            md = cuda_ops.to_device(np.ascontiguousarray(m))
            vd = cuda_ops.to_device(np.ascontiguousarray(v))
            self._m_d[key] = md
            self._v_d[key] = vd
        return md, vd

    def _stash_moments(self, keys: Iterable[str]) -> None:
        for key in keys:
            if key not in self._m_d:
                continue
            host_m = self.m[key]
            host_v = self.v[key]
            if _is_device(host_m):
                continue
            cuda_ops.sync_to_host(self._m_d[key], host_m)
            cuda_ops.sync_to_host(self._v_d[key], host_v)
            self._m_d.pop(key, None)
            self._v_d.pop(key, None)

    def _update_keys(self, keys: Iterable[str], grads: Dict) -> None:
        lr = self.current_lr()
        b1, b2, eps = self.beta1, self.beta2, self.epsilon
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        updated = []
        for key in keys:
            if key not in grads or key not in self.m:
                continue
            g = grads[key]
            if not _is_device(g):
                g = cuda_ops.to_device(np.ascontiguousarray(g, dtype=np.float32))
            w = self._get_weight(key)
            m, v = self._moment_pair(key)
            cuda_ops.adamw_update(
                w, g, m, v,
                lr, self.weight_decay, b1, b2, eps, bc1, bc2,
            )
            updated.append(w)
        if updated:
            import pycuda.driver as cuda
            cuda.Context.synchronize()
        if self.params.tie_embeddings and "token_embedding" in self.params.device_weights:
            self.params.device_weights["lm_head"] = self.params.device_weights["token_embedding"].T

    def step(self, grads: Dict) -> None:
        if self._streaming:
            self.begin_step()
            self.step_resident(grads)
            for i in range(self.params.config.num_layers):
                self.step_layer(i, grads)
            return
        self.t += 1
        self._update_keys(self._batch_keys, grads)

    def step_resident(self, grads: Dict) -> None:
        """Update always-resident keys (embed / final LN / lm_head). t must already be incremented."""
        self._update_keys(self.params.always_resident_keys(), grads)

    def step_layer(self, layer: int, grads: Dict) -> None:
        """Update one transformer block. t must already be incremented via begin_step."""
        keys = self.params.layer_keys(int(layer))
        self.params.load_layer(int(layer))
        self._update_keys(keys, grads)
        self.sync_host_weights(keys)
        self._stash_moments(keys)
        self.params.unload_layer(int(layer))

    def sync_host_weights(self, names: Optional[Iterable[str]] = None) -> None:
        """Pull device mirrors back to host NumPy dicts (checkpoint save only)."""
        if names is not None:
            keys = list(names)
        elif self._streaming:
            keys = list(self.params.always_resident_keys())
            if self.params._loaded_layer is not None:
                keys.extend(self.params.layer_keys(self.params._loaded_layer))
        else:
            keys = list(self.params.trainable_param_names())
        for key in keys:
            if self.params.tie_embeddings and key == "lm_head":
                continue
            if key in self.params.device_weights:
                host = self.params.weights[key]
                if not host.flags.c_contiguous:
                    host = np.ascontiguousarray(host)
                    self.params.weights[key] = host
                cuda_ops.sync_to_host(self.params.device_weights[key], host)
            elif key in self.params.device_biases:
                cuda_ops.sync_to_host(self.params.device_biases[key], self.params.biases[key])
        if self.params.tie_embeddings:
            self.params.weights["lm_head"] = self.params.weights["token_embedding"].T
            if "token_embedding" in self.params.device_weights:
                self.params.device_weights["lm_head"] = self.params.device_weights["token_embedding"].T
