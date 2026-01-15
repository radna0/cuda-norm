#!/usr/bin/env python3
"""Verify DeepSeek-style "latent-consume" math on GPT-OSS layer0 (TPU/JAX).

Given:
  z_t = x_t @ W_down
  k_t = z_t @ W_uK[h] (+ bK[h])
  v_t = z_t @ W_uV[h] (+ bV[h])

Stage-1 (explicit) attention scores for a query head i (GQA group -> kv head h):
  score = q_nope_i @ k_nope_h^T
Stage-2 (latent-consume) uses the identity:
  score = (q_nope_i @ W_uK[h]^T) @ z^T

And similarly for values:
  softmax(score) @ (z @ W_uV[h]) == (softmax(score) @ z) @ W_uV[h]

This script checks these equalities numerically on a small cached batch and prints max/mean diffs.
It does not require a FlashInfer/FA3 kernel; it's a correctness-only proof.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _set_shm_caches():
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/dev/shm/jax_compilation_cache")
    Path(os.environ["JAX_COMPILATION_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--params", type=str, required=True, help=".npz from tpu_train_kv_latent_from_cache.py")
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=128, help="Use first S tokens to keep compute small.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _set_shm_caches()

    import jax
    import jax.numpy as jnp

    from jax import config as jax_config

    jax_config.update("jax_default_matmul_precision", "highest")

    cache = np.load(args.cache, allow_pickle=True)
    x_np = cache["x_attn_in"]
    q_nope_np = cache["q_nope"]
    k_nope_np = cache["k_nope"]
    v_np = cache["v"]
    meta = json.loads(str(cache["meta"]))

    p = np.load(args.params, allow_pickle=True)
    w_down = p["w_down"].astype(np.float32)
    w_up_k = p["w_up_k"].astype(np.float32)
    w_up_v = p["w_up_v"].astype(np.float32)
    b_k = p["b_k"].astype(np.float32)
    b_v = p["b_v"].astype(np.float32)

    q_heads = int(meta["num_attention_heads"])
    kv_heads = int(meta["num_key_value_heads"])
    head_dim = int(meta["head_dim"])
    group = q_heads // kv_heads
    if q_heads % kv_heads != 0:
        raise ValueError(f"Expected GQA grouping (q_heads%kv_heads==0), got {q_heads}/{kv_heads}.")

    n = int(x_np.shape[0])
    s = min(int(meta["max_position_embeddings"]), int(x_np.shape[1]), args.max_seq_len)

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n, size=(args.num_blocks,))

    device = jax.devices("tpu")[0] if jax.default_backend() == "tpu" else jax.devices()[0]
    x = jax.device_put(jnp.asarray(x_np[idx, :s], dtype=jnp.float32), device)  # [B,S,H]
    q_nope = jax.device_put(jnp.asarray(q_nope_np[idx, :s], dtype=jnp.float32), device)  # [B,S,q_heads,hd]
    k_nope = jax.device_put(jnp.asarray(k_nope_np[idx, :s], dtype=jnp.float32), device)  # [B,S,kv_heads,hd]
    v = jax.device_put(jnp.asarray(v_np[idx, :s], dtype=jnp.float32), device)  # [B,S,kv_heads,hd]

    w_down = jax.device_put(jnp.asarray(w_down, dtype=jnp.float32), device)
    w_up_k = jax.device_put(jnp.asarray(w_up_k, dtype=jnp.float32), device)
    w_up_v = jax.device_put(jnp.asarray(w_up_v, dtype=jnp.float32), device)
    b_k = jax.device_put(jnp.asarray(b_k, dtype=jnp.float32), device)
    b_v = jax.device_put(jnp.asarray(b_v, dtype=jnp.float32), device)

    z = jnp.einsum("bsh,hr->bsr", x, w_down)  # [B,S,r]
    k_hat = jnp.einsum("bsr,hrd->bshd", z, w_up_k) + b_k[None, None, :, :]
    v_hat = jnp.einsum("bsr,hrd->bshd", z, w_up_v) + b_v[None, None, :, :]

    # Pick a single representative query head to keep compute light.
    qi = 0
    kv = qi // group

    q_i = q_nope[:, :, qi, :]  # [B,S,hd]
    k_i = k_hat[:, :, kv, :]  # [B,S,hd]

    # Stage-1 explicit scores: [B,S,S]
    score_explicit = jnp.einsum("bth,bsh->bts", q_i, k_i)

    # Stage-2 latent scores: q_i @ W_uK[kv]^T -> [B,S,r], then dot with z -> [B,S,S]
    wuk = w_up_k[kv]  # [r,hd]
    q_latent = jnp.einsum("bth,rh->btr", q_i, wuk)
    score_latent = jnp.einsum("btr,bsr->bts", q_latent, z)

    score_diff = (score_explicit - score_latent).astype(jnp.float32)
    print(
        "[scores]",
        "max_abs=",
        float(jnp.max(jnp.abs(score_diff))),
        "mean_abs=",
        float(jnp.mean(jnp.abs(score_diff))),
    )

    # Value latent-consume identity check (per kv head kv):
    # softmax(score) @ (z @ W_uV) == (softmax(score) @ z) @ W_uV
    # Use causal mask on the score.
    scale = 1.0 / np.sqrt(head_dim)
    mask = jnp.tril(jnp.ones((s, s), dtype=jnp.bool_))[None, :, :]
    score = jnp.where(mask, score_explicit * scale, -1e9)
    w = jax.nn.softmax(score, axis=-1)  # [B,S,S]

    v_full = v_hat[:, :, kv, :]  # [B,S,hd]
    ctx1 = jnp.einsum("bts,bsh->bth", w, v_full)  # [B,S,hd]

    wuv = w_up_v[kv]  # [r,hd]
    ctx_latent = jnp.einsum("bts,bsr->btr", w, z)  # [B,S,r]
    ctx2 = jnp.einsum("btr,rh->bth", ctx_latent, wuv)  # [B,S,hd]

    ctx_diff = (ctx1 - ctx2).astype(jnp.float32)
    print(
        "[ctx]",
        "max_abs=",
        float(jnp.max(jnp.abs(ctx_diff))),
        "mean_abs=",
        float(jnp.mean(jnp.abs(ctx_diff))),
    )

    # Sanity: reconstruction vs teacher on this batch.
    k_t = k_nope[:, :, kv, :]
    v_t = v[:, :, kv, :]
    print(
        "[recon_nope]",
        "k_mse=",
        float(jnp.mean(jnp.square((k_i - k_t).astype(jnp.float32)))),
        "v_mse=",
        float(jnp.mean(jnp.square((v_full - v_t).astype(jnp.float32)))),
    )


if __name__ == "__main__":
    main()

