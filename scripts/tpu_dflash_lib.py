#!/usr/bin/env python3
"""TPU/JAX DFlash utilities for GPT-OSS (EasyDeL teacher, Flax draft).

This module is intentionally framework-light:
- Teacher model: EasyDeL `AutoEasyDeLModelForCausalLM` (supports GPT-OSS, hidden states).
- Draft model: Flax Linen (small, trainable).

Design goals (quality-first):
- Match DFlash spec-v1 semantics used in SGLang: a fixed-size verify window
  (block_size) where token0 is an *anchor* (the last verified token duplicated
  at the start of the draft block), and tokens 1..B-1 are masked.
- Condition the draft on concatenated intermediate hidden states from a set of
  target layers ("context features"), per token in the prefix.
- Use the teacher's LM head for cross-entropy over the drafted tokens.

We keep the core model/ops here so that both:
  - teacher-cache builder (expensive forward, no grads)
  - draft trainer (cheap, many steps)
can share the same definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def set_shm_caches() -> None:
    import os
    from pathlib import Path

    # This repo carries local EasyDeL source and local shims for older eformer /
    # ejkernel versions on this TPU box. Skip the strict version gate so we can
    # run with the pinned runtime.
    os.environ.setdefault("EASYDEL_SKIP_VERSION_CHECK", "1")

    os.environ.setdefault("HF_HOME", "/dev/shm/hf")
    os.environ.setdefault("HF_HUB_CACHE", "/dev/shm/hf/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/dev/shm/hf/transformers")
    os.environ.setdefault("XDG_CACHE_HOME", "/dev/shm/xdg")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/dev/shm/jax_compilation_cache")
    Path(os.environ["JAX_COMPILATION_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)


def require_hf_token() -> None:
    import os

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
        raise RuntimeError("Missing HF token in env (HF_TOKEN or HUGGINGFACE_HUB_TOKEN).")


def build_target_layer_ids(num_target_layers: int, num_context_features: int) -> list[int]:
    """Mirror the DFlash layer selection heuristic: evenly spaced, skip early/late."""
    if int(num_context_features) <= 0:
        raise ValueError("num_context_features must be positive")
    if int(num_target_layers) <= 0:
        raise ValueError("num_target_layers must be positive")
    if int(num_context_features) == 1:
        return [int(num_target_layers) // 2]
    start = 1
    end = int(num_target_layers) - 3
    span = end - start
    return [
        int(round(start + (i * span) / (int(num_context_features) - 1)))
        for i in range(int(num_context_features))
    ]


def load_json(path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def build_rope(*, cfg: dict, dtype) -> Any:
    """Build EasyDeL RoPE object matching GPT-OSS config."""
    from easydel.layers.rotary_embedding import get_rope

    return get_rope(
        head_size=int(cfg["head_dim"]),
        rotary_dim=int(cfg["head_dim"]),
        max_position=int(cfg["max_position_embeddings"]),
        base=int(cfg["rope_theta"]),
        is_neox_style=True,
        rope_scaling=cfg.get("rope_scaling"),
        dtype=dtype,
    )


@dataclass(frozen=True)
class DFlashDraftConfig:
    hidden_size: int
    num_layers: int
    mlp_ratio: float
    hidden_act: str
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rope_theta: float
    rope_scaling: dict | None
    rms_norm_eps: float
    block_size: int
    num_context_features: int


def _repeat_kv(x, n_rep: int):
    # x: [B, S, kvH, D] -> [B, S, kvH*n_rep, D]
    import jax.numpy as jnp

    if int(n_rep) == 1:
        return x
    b, s, kvh, d = x.shape
    x = x[:, :, None, :, :].repeat(int(n_rep), axis=2)
    return x.reshape((b, s, kvh * int(n_rep), d))


def _split_heads(x, n_heads: int, head_dim: int):
    # x: [B, S, n_heads*D] -> [B, S, n_heads, D]
    return x.reshape((x.shape[0], x.shape[1], int(n_heads), int(head_dim)))


def _merge_heads(x):
    # x: [B, S, H, D] -> [B, S, H*D]
    return x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
