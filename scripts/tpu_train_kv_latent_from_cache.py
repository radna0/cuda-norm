#!/usr/bin/env python3
"""Train a DeepSeek-native KV latent module for GPT-OSS layer0 (TPU/JAX).

This is the next step beyond "RoPE-K head synthesis":
  - Learn a shared latent per token z = x_attn_in @ W_down (rank=r)
  - Reconstruct per-KV-head K_nope and V from z:
      K_hat = z @ W_uK + bK
      V_hat = z @ W_uV + bV

Input cache: built by scripts/tpu_build_kqkv_cache_layer0.py
  - x_attn_in: [N,S,H]  (float16/float32)
  - k_nope:    [N,S,kv_heads,head_dim]
  - k_rope:    [N,S,kv_heads,head_dim]
  - v:         [N,S,kv_heads,head_dim]
  - meta:      JSON

Loss:
  - K loss uses RoPE-applied keys by default: mse( RoPE(K_hat), k_rope_teacher )
    (this matches runtime semantics; RoPE is deterministic and applied after projection)
  - V loss: mse( V_hat, v_teacher )

This is still *layer0 only / clean context*. The goal is to prove we can represent the
non-shareable KV-head geometry without TransMLA RoRoPE/PCA assumptions, and then
reduce rank r with training.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def _set_shm_caches():
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/dev/shm/jax_compilation_cache")
    Path(os.environ["JAX_COMPILATION_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)


def _build_rope_from_meta(meta: dict):
    import jax.numpy as jnp
    from easydel.layers.rotary_embedding import get_rope

    return get_rope(
        head_size=int(meta["head_dim"]),
        rotary_dim=int(meta["head_dim"]),
        max_position=int(meta["max_position_embeddings"]),
        base=int(meta["rope_theta"]),
        is_neox_style=True,
        rope_scaling=meta.get("rope_scaling"),
        dtype=jnp.bfloat16,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--latent-rank", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-loss", type=str, default="rope", choices=("rope", "nope"))
    parser.add_argument("--k-weight", type=float, default=1.0)
    parser.add_argument("--v-weight", type=float, default=1.0)
    parser.add_argument("--out", type=str, default=f"/dev/shm/out/kv_latent_layer0_{time.strftime('%Y%m%d_%H%M%S')}.npz")
    args = parser.parse_args()

    _set_shm_caches()

    import jax
    import jax.numpy as jnp
    import optax

    from jax import config as jax_config

    jax_config.update("jax_default_matmul_precision", "highest")

    cache = np.load(args.cache, allow_pickle=True)
    x_np = cache["x_attn_in"]
    k_nope_np = cache["k_nope"]
    k_rope_np = cache["k_rope"]
    v_np = cache["v"]
    meta = json.loads(str(cache["meta"]))

    n, s, hidden = x_np.shape
    kv_heads = int(meta["num_key_value_heads"])
    head_dim = int(meta["head_dim"])
    kv_dim = kv_heads * head_dim
    if args.latent_rank > kv_dim:
        raise ValueError(f"latent_rank must be <= kv_dim (got {args.latent_rank} > {kv_dim})")

    device = jax.devices("tpu")[0] if jax.default_backend() == "tpu" else jax.devices()[0]
    x = jax.device_put(jnp.asarray(x_np, dtype=jnp.float32), device)
    k_nope = jax.device_put(jnp.asarray(k_nope_np, dtype=jnp.float32), device)
    k_rope = jax.device_put(jnp.asarray(k_rope_np, dtype=jnp.float32), device)
    v = jax.device_put(jnp.asarray(v_np, dtype=jnp.float32), device)

    rope = _build_rope_from_meta(meta)
    pos = jax.device_put(jnp.arange(s, dtype=jnp.int32)[None, :], device)

    key = jax.random.PRNGKey(args.seed)
    key, k1, k2, k3, k4 = jax.random.split(key, 5)
    w_down = jax.random.normal(k1, (hidden, args.latent_rank), dtype=jnp.float32) * (1.0 / np.sqrt(hidden))
    w_up_k = jax.random.normal(k2, (kv_heads, args.latent_rank, head_dim), dtype=jnp.float32) * (
        1.0 / np.sqrt(args.latent_rank)
    )
    w_up_v = jax.random.normal(k3, (kv_heads, args.latent_rank, head_dim), dtype=jnp.float32) * (
        1.0 / np.sqrt(args.latent_rank)
    )
    b_k = jax.random.normal(k4, (kv_heads, head_dim), dtype=jnp.float32) * 0.0
    b_v = jnp.zeros((kv_heads, head_dim), dtype=jnp.float32)
    params = {"w_down": w_down, "w_up_k": w_up_k, "w_up_v": w_up_v, "b_k": b_k, "b_v": b_v}

    opt = optax.adamw(learning_rate=args.lr, b1=0.9, b2=0.999, weight_decay=0.0)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, batch_idx):
        def loss_fn(p):
            xb = x[batch_idx]  # [B,S,H]
            kb_nope = k_nope[batch_idx]  # [B,S,kv_heads,head_dim]
            kb_rope = k_rope[batch_idx]
            vb = v[batch_idx]

            z = jnp.einsum("bsh,hr->bsr", xb, p["w_down"])  # [B,S,r]
            k_hat = jnp.einsum("bsr,hrd->bshd", z, p["w_up_k"]) + p["b_k"][None, None, :, :]
            v_hat = jnp.einsum("bsr,hrd->bshd", z, p["w_up_v"]) + p["b_v"][None, None, :, :]

            if args.k_loss == "rope":
                q0 = jnp.zeros_like(k_hat)
                _, k_hat_rope = rope(pos, q0, k_hat.astype(jnp.bfloat16))
                diff_k = (k_hat_rope.astype(jnp.float32) - kb_rope).astype(jnp.float32)
            else:
                diff_k = (k_hat - kb_nope).astype(jnp.float32)

            diff_v = (v_hat - vb).astype(jnp.float32)

            loss_k = jnp.mean(jnp.square(diff_k))
            loss_v = jnp.mean(jnp.square(diff_v))
            return (args.k_weight * loss_k) + (args.v_weight * loss_v), (loss_k, loss_v)

        (loss, (loss_k, loss_v)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state2 = opt.update(grads, opt_state, params)
        params2 = optax.apply_updates(params, updates)
        return params2, opt_state2, loss, loss_k, loss_v

    losses = []
    losses_k = []
    losses_v = []
    t0 = time.time()
    for t in range(args.steps):
        idx = np.random.randint(0, n, size=(args.batch_size,))
        params, opt_state, loss, loss_k, loss_v = step(params, opt_state, jnp.asarray(idx, dtype=jnp.int32))
        lf = float(loss)
        lk = float(loss_k)
        lv = float(loss_v)
        losses.append(lf)
        losses_k.append(lk)
        losses_v.append(lv)
        if (t + 1) % 20 == 0:
            print(f"[step {t+1:05d}] loss={lf:.6f} (k={lk:.6f} v={lv:.6f})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        w_down=np.array(jax.device_get(params["w_down"])),
        w_up_k=np.array(jax.device_get(params["w_up_k"])),
        w_up_v=np.array(jax.device_get(params["w_up_v"])),
        b_k=np.array(jax.device_get(params["b_k"])),
        b_v=np.array(jax.device_get(params["b_v"])),
        losses=np.array(losses, dtype=np.float32),
        losses_k=np.array(losses_k, dtype=np.float32),
        losses_v=np.array(losses_v, dtype=np.float32),
        cache_path=str(Path(args.cache).resolve()),
        meta=json.dumps(
            {
                "latent_rank": args.latent_rank,
                "k_loss": args.k_loss,
                "k_weight": args.k_weight,
                "v_weight": args.v_weight,
                "steps": args.steps,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "seed": args.seed,
            }
        ),
    )
    print(f"[done] wrote {out_path} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

