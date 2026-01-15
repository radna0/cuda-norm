#!/usr/bin/env python3
"""Evaluate RoPE-K head synthesis impact on layer0 attention output (TPU/JAX).

This is a correctness/quality diagnostic, not full-model PPL.

We compare:
  baseline: attn(q_nope, k_nope_teacher, v_teacher)
  synthK:   attn(q_nope, k_nope_synth,   v_teacher)

Scores use pre-RoPE tensors because dot-product is RoPE-invariant when RoPE is applied
to both Q and K with the same angles (standard GPT-OSS setup).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _set_shm_caches():
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/dev/shm/jax_compilation_cache")


def _cos(a, b, eps=1e-8):
    import jax.numpy as jnp

    num = jnp.sum(a * b, axis=-1)
    den = jnp.sqrt(jnp.sum(a * a, axis=-1) + eps) * jnp.sqrt(jnp.sum(b * b, axis=-1) + eps)
    return num / den


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, required=True, help=".npz from tpu_build_kqkv_cache_layer0.py")
    parser.add_argument("--k-params", type=str, required=True, help=".npz from tpu_train_ropek_head_synth_from_cache.py")
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _set_shm_caches()

    import jax
    import jax.numpy as jnp

    from jax import config as jax_config

    jax_config.update("jax_default_matmul_precision", "highest")

    cache = np.load(args.cache, allow_pickle=True)
    meta = json.loads(str(cache["meta"]))
    x_np = cache["x_attn_in"].astype(np.float32)
    q_np = cache["q_nope"].astype(np.float32)
    k_np = cache["k_nope"].astype(np.float32)
    v_np = cache["v"].astype(np.float32)

    p = np.load(args.k_params, allow_pickle=True)
    w_down = p["w_down"].astype(np.float32)
    w_up = p["w_up"].astype(np.float32)

    q_heads = int(meta["num_attention_heads"])
    kv_heads = int(meta["num_key_value_heads"])
    head_dim = int(meta["head_dim"])
    group = q_heads // kv_heads

    n = int(x_np.shape[0])
    s = min(int(x_np.shape[1]), int(meta["max_position_embeddings"]), args.max_seq_len)

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n, size=(args.num_blocks,))

    device = jax.devices("tpu")[0] if jax.default_backend() == "tpu" else jax.devices()[0]
    x = jax.device_put(jnp.asarray(x_np[idx, :s], dtype=jnp.float32), device)  # [B,S,H]
    q = jax.device_put(jnp.asarray(q_np[idx, :s], dtype=jnp.float32), device)  # [B,S,q,hd]
    k = jax.device_put(jnp.asarray(k_np[idx, :s], dtype=jnp.float32), device)  # [B,S,kv,hd]
    v = jax.device_put(jnp.asarray(v_np[idx, :s], dtype=jnp.float32), device)  # [B,S,kv,hd]
    w_down = jax.device_put(jnp.asarray(w_down, dtype=jnp.float32), device)
    w_up = jax.device_put(jnp.asarray(w_up, dtype=jnp.float32), device)

    z = jnp.einsum("bsh,hr->bsr", x, w_down)  # [B,S,r]
    k_hat = jnp.einsum("bsr,hrd->bshd", z, w_up)  # [B,S,kv,hd]

    # Precompute a causal mask once.
    mask = jnp.tril(jnp.ones((s, s), dtype=jnp.bool_))[None, None, :, :]  # [1,1,S,S]
    scale = 1.0 / np.sqrt(head_dim)

    kv_idx = (jnp.arange(q_heads, dtype=jnp.int32) // group)  # [q]

    @jax.jit
    def attn_out(k_in):
        # k_in: [B,S,kv,hd]
        # Gather per-q-head K/V by GQA group mapping.
        k_gqa = jnp.take(k_in, kv_idx, axis=2)  # [B,S,q,hd]
        v_gqa = jnp.take(v, kv_idx, axis=2)  # [B,S,q,hd]

        # scores: [B,T,q,S]
        scores = jnp.einsum("btqh,bsqh->btqs", q, k_gqa) * scale
        scores = jnp.where(mask, scores, -1e9)
        w = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("btqs,bsqh->btqh", w, v_gqa)  # [B,T,q,hd]
        return out

    out_base = attn_out(k)
    out_synth = attn_out(k_hat)

    diff = (out_synth - out_base).astype(jnp.float32)
    rel_l2 = jnp.sqrt(jnp.mean(jnp.square(diff))) / (jnp.sqrt(jnp.mean(jnp.square(out_base.astype(jnp.float32)))) + 1e-8)
    cos = _cos(out_synth.astype(jnp.float32), out_base.astype(jnp.float32))

    print("[attn_out] rel_l2=", float(rel_l2), "mean_cos=", float(jnp.mean(cos)), "p10_cos=", float(jnp.quantile(cos, 0.1)))


if __name__ == "__main__":
    main()
