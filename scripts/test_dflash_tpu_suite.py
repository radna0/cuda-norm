from __future__ import annotations


def main() -> None:
    # This suite is meant to run in the TPU environment where JAX is installed.
    try:
        import jax.numpy as jnp  # type: ignore
    except Exception:
        print("skipped (jax not installed)", flush=True)
        return

    import numpy as np

    from easydel.inference.speculative.dflash import (
        dflash_accept_len_and_bonus,
        extract_dflash_context_features_from_hidden_states,
    )

    # ---- accept_len parity (numpy reference)
    def ref(cand: np.ndarray, pred: np.ndarray):
        matches = (cand[:, 1:] == pred[:, :-1]).astype(np.int32)
        accept_len = np.cumprod(matches, axis=1).sum(axis=1).astype(np.int32)
        bonus = pred[np.arange(cand.shape[0]), accept_len].astype(np.int32)
        return accept_len, bonus

    rng = np.random.default_rng(0)
    cand = rng.integers(0, 100, size=(4, 8), dtype=np.int32)
    pred = rng.integers(0, 100, size=(4, 8), dtype=np.int32)
    a_ref, b_ref = ref(cand, pred)
    a, b = dflash_accept_len_and_bonus(candidates=jnp.asarray(cand), target_predict=jnp.asarray(pred))
    assert np.array_equal(np.asarray(a), a_ref)
    assert np.array_equal(np.asarray(b), b_ref)

    # ---- context feature extraction shape test
    batch, seq, hidden = 2, 3, 4
    hs = [jnp.full((batch, seq, hidden), i, dtype=jnp.float32) for i in range(10)]
    out = extract_dflash_context_features_from_hidden_states(
        hidden_states=hs, target_layer_ids=[1, 3, 6], add_one_for_pre_layer_capture=True
    )
    assert out.shape == (batch, seq, 12)

    # ---- draft ctx KV cache sanity (requires flax.nnx)
    try:
        import jax
        from flax import nnx

        from easydel.inference.speculative import (
            DFlashDraftModel,
            DFlashDraftModelConfig,
            append_draft_ctx_kv,
            draft_forward_with_ctx_kv,
            materialize_draft_ctx_kv,
        )

        def rope(pos, q, k):
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
        draft = DFlashDraftModel(cfg, rngs=nnx.Rngs(0))
        ctx_hidden0 = jax.random.normal(jax.random.key(0), (1, 5, cfg.hidden_size), dtype=jnp.bfloat16)
        cache0 = materialize_draft_ctx_kv(draft=draft, rope=rope, ctx_hidden=ctx_hidden0)
        ctx_hidden_add = jax.random.normal(jax.random.key(1), (1, 2, cfg.hidden_size), dtype=jnp.bfloat16)
        cache1 = append_draft_ctx_kv(draft=draft, rope=rope, cache=cache0, new_ctx_hidden=ctx_hidden_add)
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
    except Exception as e:
        raise RuntimeError("draft ctx-KV cache sanity failed") from e

    print("ok", flush=True)


if __name__ == "__main__":
    main()
