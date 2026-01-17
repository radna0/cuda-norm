from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DraftCtxKVCache:
    """Per-layer ctx KV cache for the DFlash draft model.

    k_full/v_full are lists of arrays shaped:
      - k_full[i]: [B, ctx_len, H, D]
      - v_full[i]: [B, ctx_len, H, D]
    where H = num_attention_heads, D = head_dim.
    """

    k_full: list[Any]
    v_full: list[Any]
    ctx_len: int


def materialize_draft_ctx_kv(*, draft: Any, rope: Any, ctx_hidden: Any) -> DraftCtxKVCache:
    """Materialize ctx KV for all draft layers (one-time for current ctx length)."""
    k_list = []
    v_list = []
    for layer in draft.layers:
        k, v = layer.materialize_ctx_kv(rope=rope, ctx_hidden=ctx_hidden)
        k_list.append(k)
        v_list.append(v)
    return DraftCtxKVCache(k_full=k_list, v_full=v_list, ctx_len=int(ctx_hidden.shape[1]))


def append_draft_ctx_kv(
    *,
    draft: Any,
    rope: Any,
    cache: DraftCtxKVCache,
    new_ctx_hidden: Any,
) -> DraftCtxKVCache:
    """Append new ctx tokens to the existing per-layer ctx KV cache."""
    if int(new_ctx_hidden.shape[1]) <= 0:
        return cache
    k_list = []
    v_list = []
    for layer, k_old, v_old in zip(draft.layers, cache.k_full, cache.v_full):
        k_new, v_new = layer.append_ctx_kv(
            rope=rope,
            ctx_k_full=k_old,
            ctx_v_full=v_old,
            new_ctx_hidden=new_ctx_hidden,
            start_pos=int(cache.ctx_len),
        )
        k_list.append(k_new)
        v_list.append(v_new)
    return DraftCtxKVCache(k_full=k_list, v_full=v_list, ctx_len=int(cache.ctx_len + int(new_ctx_hidden.shape[1])))


def draft_forward_with_ctx_kv(
    *,
    draft: Any,
    rope: Any,
    cache: DraftCtxKVCache,
    anchor_embedding: Any,
    mask_embedding: Any,
    block_size: int,
) -> Any:
    """Run the draft block forward using cached ctx KV (no ctx recomputation).

    Returns hidden states [B, block_size, hidden].
    """
    import jax.numpy as jnp

    b = int(anchor_embedding.shape[0])
    hidden = int(anchor_embedding.shape[-1])
    mask = jnp.broadcast_to(mask_embedding[None, None, :], (b, int(block_size - 1), hidden))
    noise_hidden = jnp.concatenate([anchor_embedding[:, None, :], mask], axis=1)

    x = noise_hidden
    for layer, k_ctx, v_ctx in zip(draft.layers, cache.k_full, cache.v_full):
        x = layer.forward_with_ctx_kv(rope=rope, ctx_k_full=k_ctx, ctx_v_full=v_ctx, noise_hidden=x)
    return draft.final_norm(x)

