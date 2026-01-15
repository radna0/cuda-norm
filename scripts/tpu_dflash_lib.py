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


def _apply_rope_separate(*, rope, pos_q, q, pos_k, k):
    """Apply RoPE to q and k with potentially different positions.

    EasyDeL's RoPE object takes one `positions` array for both query+key. We call it
    twice (q-only and k-only) using a dummy tensor so we can avoid constructing a
    huge [ctx+block] query for very long contexts.
    """
    q_rot, _ = rope(pos_q, q, q)
    _, k_rot = rope(pos_k, k, k)
    return q_rot, k_rot


def _causal_lm_ce_loss(logits, labels):
    import jax.numpy as jnp
    import optax

    loss = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), labels)
    return jnp.mean(loss)


def make_dflash_draft_module():
    """Define and return a Flax Linen DFlash draft model class."""
    import flax.linen as nn
    import jax
    import jax.numpy as jnp

    class _RMSNorm(nn.Module):
        hidden_size: int
        eps: float = 1e-6

        @nn.compact
        def __call__(self, x):
            w = self.param("weight", nn.initializers.ones, (int(self.hidden_size),), jnp.float32)
            x_f = x.astype(jnp.float32)
            var = jnp.mean(jnp.square(x_f), axis=-1, keepdims=True)
            x_n = x_f * jax.lax.rsqrt(var + float(self.eps))
            y = x_n * w
            return y.astype(x.dtype)

    def _act(name: str):
        n = str(name).lower()
        if n in ("silu", "swish"):
            return jax.nn.silu
        raise ValueError(f"Unsupported activation: {name!r}")

    class _DFlashAttention(nn.Module):
        cfg: DFlashDraftConfig

        @nn.compact
        def __call__(self, *, rope, ctx_hidden, noise_hidden):
            c = self.cfg
            b = noise_hidden.shape[0]
            ctx_len = ctx_hidden.shape[1]
            q_len = noise_hidden.shape[1]

            q_proj = nn.Dense(int(c.num_attention_heads) * int(c.head_dim), use_bias=bool(True), name="q_proj")
            k_proj = nn.Dense(int(c.num_key_value_heads) * int(c.head_dim), use_bias=bool(True), name="k_proj")
            v_proj = nn.Dense(int(c.num_key_value_heads) * int(c.head_dim), use_bias=bool(True), name="v_proj")
            o_proj = nn.Dense(int(c.hidden_size), use_bias=bool(True), name="o_proj")

            q = _split_heads(q_proj(noise_hidden), int(c.num_attention_heads), int(c.head_dim))
            k_ctx = _split_heads(k_proj(ctx_hidden), int(c.num_key_value_heads), int(c.head_dim))
            v_ctx = _split_heads(v_proj(ctx_hidden), int(c.num_key_value_heads), int(c.head_dim))
            k_noise = _split_heads(k_proj(noise_hidden), int(c.num_key_value_heads), int(c.head_dim))
            v_noise = _split_heads(v_proj(noise_hidden), int(c.num_key_value_heads), int(c.head_dim))

            k = jnp.concatenate([k_ctx, k_noise], axis=1)  # [B, ctx+block, kvH, D]
            v = jnp.concatenate([v_ctx, v_noise], axis=1)

            # RoPE positions:
            pos_k = jnp.arange(int(ctx_len + q_len), dtype=jnp.int32)[None, :]
            pos_q = (jnp.arange(int(q_len), dtype=jnp.int32) + int(ctx_len))[None, :]
            q_rope, k_rope = _apply_rope_separate(rope=rope, pos_q=pos_q, q=q, pos_k=pos_k, k=k)

            rep = int(c.num_attention_heads) // int(c.num_key_value_heads)
            k_full = _repeat_kv(k_rope, rep)
            v_full = _repeat_kv(v, rep)

            scale = float(int(c.head_dim) ** -0.5)
            attn = jnp.einsum("bqhd,bkhd->bhqk", q_rope, k_full, precision=jax.lax.Precision.HIGHEST) * scale
            attn = attn - jnp.max(attn, axis=-1, keepdims=True)
            probs = jax.nn.softmax(attn, axis=-1)
            out = jnp.einsum("bhqk,bkhd->bqhd", probs, v_full, precision=jax.lax.Precision.HIGHEST)
            out = _merge_heads(out)
            return o_proj(out)

    class _DFlashMLP(nn.Module):
        cfg: DFlashDraftConfig

        @nn.compact
        def __call__(self, x):
            c = self.cfg
            inter = int(round(float(c.hidden_size) * float(c.mlp_ratio)))
            gate_proj = nn.Dense(inter, use_bias=True, name="gate_proj")
            up_proj = nn.Dense(inter, use_bias=True, name="up_proj")
            down_proj = nn.Dense(int(c.hidden_size), use_bias=True, name="down_proj")
            act = _act(c.hidden_act)
            return down_proj(act(gate_proj(x)) * up_proj(x))

    class _DFlashDecoderLayer(nn.Module):
        cfg: DFlashDraftConfig

        @nn.compact
        def __call__(self, *, rope, ctx_hidden, x):
            c = self.cfg
            x_norm = _RMSNorm(int(c.hidden_size), eps=float(c.rms_norm_eps), name="input_norm")(x)
            x = x + _DFlashAttention(cfg=c, name="self_attn")(rope=rope, ctx_hidden=ctx_hidden, noise_hidden=x_norm)
            x_norm2 = _RMSNorm(int(c.hidden_size), eps=float(c.rms_norm_eps), name="post_attn_norm")(x)
            x = x + _DFlashMLP(cfg=c, name="mlp")(x_norm2)
            return x

    class DFlashDraftModel(nn.Module):
        cfg: DFlashDraftConfig

        def setup(self) -> None:
            c = self.cfg
            self.fc = nn.Dense(int(c.hidden_size), use_bias=False, name="fc")
            self.hidden_norm = _RMSNorm(int(c.hidden_size), eps=float(c.rms_norm_eps), name="hidden_norm")
            self.layers = [_DFlashDecoderLayer(cfg=c, name=f"layers_{i}") for i in range(int(c.num_layers))]
            self.final_norm = _RMSNorm(int(c.hidden_size), eps=float(c.rms_norm_eps), name="norm")
            self.mask_embedding = self.param(
                "mask_embedding",
                nn.initializers.zeros,
                (int(c.hidden_size),),
                jnp.float32,
            )

        def _project_context(self, context_features):
            # context_features: [B, ctx, K*hidden] -> [B, ctx, hidden]
            x = self.fc(context_features)
            return self.hidden_norm(x)

        def __call__(self, *, context_features, anchor_embedding, rope):
            # context_features: [B, ctx, K*hidden]
            # anchor_embedding: [B, hidden]
            # returns: [B, block, hidden]
            c = self.cfg
            ctx_hidden = self._project_context(context_features)

            b = anchor_embedding.shape[0]
            block = int(c.block_size)
            mask = jnp.broadcast_to(self.mask_embedding.astype(anchor_embedding.dtype), (b, block - 1, int(c.hidden_size)))
            noise_hidden = jnp.concatenate([anchor_embedding[:, None, :], mask], axis=1)

            x = noise_hidden
            for layer in self.layers:
                x = layer(rope=rope, ctx_hidden=ctx_hidden, x=x)
            return self.final_norm(x)

    return DFlashDraftModel


__all__ = [
    "DFlashDraftConfig",
    "build_target_layer_ids",
    "build_rope",
    "load_json",
    "make_dflash_draft_module",
    "require_hf_token",
    "set_shm_caches",
    "_causal_lm_ce_loss",
]
