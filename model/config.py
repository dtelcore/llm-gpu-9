"""
model/config.py

Typed view over the `model` section of training_config.json.
"""

import math
from typing import Any, Dict


def parse_residual_scale(raw, num_layers: int) -> float:
    """GPT-2 residual branch scale. Missing/false = 1.0 (legacy checkpoints)."""
    if raw is True:
        return 1.0 / math.sqrt(2.0 * max(1, int(num_layers)))
    if isinstance(raw, str):
        key = raw.strip().lower()
        if key in ("true", "gpt2", "1/sqrt(2l)", "1/sqrt(2*l)"):
            return 1.0 / math.sqrt(2.0 * max(1, int(num_layers)))
        if key in ("", "false", "off", "none"):
            return 1.0
        return float(key)
    if raw is False or raw is None:
        return 1.0
    return float(raw)


class GPTConfig:
    def __init__(self, model_dict: Dict[str, Any]) -> None:
        self.name: str = model_dict.get("name", "Custom")
        self.vocab_size: int = int(model_dict["vocab_size"])
        self.max_len: int = int(model_dict["max_len"])
        self.embedding_dim: int = int(model_dict["embedding_dim"])
        self.num_heads: int = int(model_dict["num_heads"])
        self.num_layers: int = int(model_dict["num_layers"])
        self.dropout_prob: float = float(model_dict.get("dropout_prob", 0.0))
        # Share token_embedding with lm_head (lm_head = embedding.T). Default
        # False so legacy checkpoints without the key stay untied; new presets
        # set tie_embeddings=True explicitly.
        self.tie_embeddings: bool = bool(model_dict.get("tie_embeddings", False))
        # "layernorm" (legacy) | "rmsnorm" (scale-only). Default layernorm for old ckpts.
        norm = str(model_dict.get("norm_type", "layernorm")).lower()
        if norm not in ("layernorm", "rmsnorm"):
            raise ValueError(f"norm_type must be 'layernorm' or 'rmsnorm', got {norm!r}")
        self.norm_type: str = norm
        # "learned" absolute position table | "rope". Default learned for old ckpts.
        pos = str(model_dict.get("pos_encoding", "learned")).lower()
        if pos not in ("learned", "rope"):
            raise ValueError(f"pos_encoding must be 'learned' or 'rope', got {pos!r}")
        self.pos_encoding: str = pos
        self.rope_base: float = float(model_dict.get("rope_base", 10000.0))
        self.gradient_checkpointing: bool = bool(model_dict.get("gradient_checkpointing", False))
        # Default 1.0 so run8+16 / older checkpoints stay bit-identical.
        self.residual_scale: float = parse_residual_scale(
            model_dict.get("residual_scale", False), self.num_layers,
        )
        strategy = str(model_dict.get("layer_strategy", "resident")).strip().lower()
        if strategy not in ("resident", "stream"):
            raise ValueError(f"layer_strategy must be 'resident' or 'stream', got {strategy!r}")
        self.layer_strategy: str = strategy

        assert self.embedding_dim % self.num_heads == 0, (
            f"embedding_dim ({self.embedding_dim}) must be divisible by "
            f"num_heads ({self.num_heads})"
        )
        self.head_dim: int = self.embedding_dim // self.num_heads

    @property
    def use_rmsnorm(self) -> bool:
        return self.norm_type == "rmsnorm"

    @property
    def use_rope(self) -> bool:
        return self.pos_encoding == "rope"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "vocab_size": self.vocab_size,
            "max_len": self.max_len,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dropout_prob": self.dropout_prob,
            "tie_embeddings": self.tie_embeddings,
            "norm_type": self.norm_type,
            "pos_encoding": self.pos_encoding,
            "rope_base": self.rope_base,
            "gradient_checkpointing": self.gradient_checkpointing,
            "residual_scale": self.residual_scale,
            "layer_strategy": self.layer_strategy,
        }

    def __repr__(self) -> str:
        return (
            f"GPTConfig(name={self.name!r}, vocab_size={self.vocab_size}, "
            f"max_len={self.max_len}, embedding_dim={self.embedding_dim}, "
            f"num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"num_layers={self.num_layers}, tie_embeddings={self.tie_embeddings}, "
            f"norm_type={self.norm_type!r}, pos_encoding={self.pos_encoding!r}, "
            f"gradient_checkpointing={self.gradient_checkpointing}, "
            f"residual_scale={self.residual_scale:.6g}, "
            f"layer_strategy={self.layer_strategy!r})"
        )
