#!/usr/bin/env python3
"""JAX test: draft ctx KV cache materialize/append/forward.

This is a *unit* test for the EasyDeL/JAX DFlash draft model cache path.
It runs only when JAX+Flax are installed (TPU env or local jax[cpu]).
"""

from __future__ import annotations


def main() -> None:
    try:
        import jax
        import jax.numpy as jnp
        from flax import nnx
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"SKIP (needs jax+flax): {e}")

    from easydel.inference.speculative import (
        DFlashDraftModel,
        DFlashDraftModelConfig,
        append_draft_ctx_kv,
        draft_forward_with_ctx_kv,
        materialize_draft_ctx_kv,
    )

    def rope(pos, q, k):
        # Identity RoPE for unit testing.
        return q, k

    cfg = DFlashDraftModelConfig(
        hidden_size=32,
        num_layers=2,
        mlp_ratio=2.0,
        hidden_act="silu",
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-5,
        block_size=8,
        num_context_features=2,
        qk_norm=True,
        remat=False,
    )

    rngs = nnx.Rngs(0)
    draft = DFlashDraftModel(cfg, rngs=rngs)

    ctx_len0 = 5
    ctx_hidden0 = jax.random.normal(jax.random.key(0), (1, ctx_len0, cfg.hidden_size), dtype=jnp.bfloat16)
    cache0 = materialize_draft_ctx_kv(draft=draft, rope=rope, ctx_hidden=ctx_hidden0)
    assert cache0.ctx_len == ctx_len0
    assert len(cache0.k_full) == cfg.num_layers
    assert len(cache0.v_full) == cfg.num_layers

    # Shapes: [B, ctx, H, D]
    k0 = cache0.k_full[0]
    v0 = cache0.v_full[0]
    assert k0.shape[:2] == (1, ctx_len0)
    assert v0.shape[:2] == (1, ctx_len0)
    assert k0.shape[2] == cfg.num_attention_heads
    assert k0.shape[3] == cfg.head_dim

    ctx_len_add = 3
    ctx_hidden_add = jax.random.normal(jax.random.key(1), (1, ctx_len_add, cfg.hidden_size), dtype=jnp.bfloat16)
    cache1 = append_draft_ctx_kv(draft=draft, rope=rope, cache=cache0, new_ctx_hidden=ctx_hidden_add)
    assert cache1.ctx_len == ctx_len0 + ctx_len_add
    assert cache1.k_full[0].shape[1] == cache1.ctx_len

    anchor = jax.random.normal(jax.random.key(2), (1, cfg.hidden_size), dtype=jnp.bfloat16)
    out = draft_forward_with_ctx_kv(
        draft=draft,
        rope=rope,
        cache=cache1,
        anchor_embedding=anchor,
        mask_embedding=draft.mask_embedding.value.astype(jnp.bfloat16),
        block_size=cfg.block_size,
    )
    assert out.shape == (1, cfg.block_size, cfg.hidden_size)

    print("[OK] draft ctx-KV cache unit test passed", flush=True)


if __name__ == "__main__":
    main()
