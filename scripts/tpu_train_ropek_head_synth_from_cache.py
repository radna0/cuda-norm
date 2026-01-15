#!/usr/bin/env python3
"""Train RoPE‑K head synthesis from a clean-context cache (TPU/JAX).

Input cache .npz from scripts/tpu_build_ropek_cache.py with:
- x_attn_in: float32 [N,S,H]
- k_rope:   float32 [N,S,kv_heads,head_dim]

We train W_down (H->r) and W_up (kv_heads, r->head_dim) so that:
  k_hat_nope = (x_attn_in @ W_down) @ W_up
  k_hat_rope = RoPE(k_hat_nope) matches k_rope.

This is the minimal learned component needed to make "shared latent → per-head RoPE‑K"
possible (DeepSeek-native direction) instead of forcing shareability.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--latent-rank", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=f"/dev/shm/out/ropek_head_synth_from_cache_{time.strftime('%Y%m%d_%H%M%S')}.npz")

    args = parser.parse_args()

    _set_shm_caches()

    import jax
    import jax.numpy as jnp
    import optax

    # Highest numerical precision for matmul/dot.
    from jax import config as jax_config

    jax_config.update("jax_default_matmul_precision", "highest")

    key = jax.random.PRNGKey(args.seed)

    cache = np.load(args.cache, allow_pickle=True)
    x_np = cache["x_attn_in"].astype(np.float32)  # [N,S,H]
    k_np = cache["k_rope"].astype(np.float32)  # [N,S,kv_heads,head_dim]
    meta = json.loads(str(cache["meta"])) if "meta" in cache.files else {}

    n, s, h = x_np.shape
    kv_heads = k_np.shape[2]
    head_dim = k_np.shape[3]
    if args.latent_rank > (kv_heads * head_dim):
        raise ValueError(f"latent_rank must be <= kv_dim (got {args.latent_rank} > {kv_heads*head_dim})")

    device = jax.devices("tpu")[0] if jax.default_backend() == "tpu" else jax.devices()[0]
    x = jax.device_put(jnp.asarray(x_np, dtype=jnp.float32), device)
    k_rope = jax.device_put(jnp.asarray(k_np, dtype=jnp.float32), device)

    if not meta:
        raise RuntimeError("Cache is missing 'meta' needed to build RoPE (use scripts/tpu_build_ropek_cache.py).")

    from easydel.layers.rotary_embedding import get_rope

    rope = get_rope(
        head_size=int(meta["head_dim"]),
        rotary_dim=int(meta["head_dim"]),
        max_position=int(meta["max_position_embeddings"]),
        base=int(meta["rope_theta"]),
        is_neox_style=True,
        rope_scaling=meta.get("rope_scaling"),
        dtype=jnp.bfloat16,
    )
    pos = jax.device_put(jnp.arange(s, dtype=jnp.int32)[None, :], device)

    # Params
    key, k1, k2 = jax.random.split(key, 3)
    w_down = jax.random.normal(k1, (h, args.latent_rank), dtype=jnp.float32) * (1.0 / np.sqrt(h))
    w_up = jax.random.normal(k2, (kv_heads, args.latent_rank, head_dim), dtype=jnp.float32) * (
        1.0 / np.sqrt(args.latent_rank)
    )
    params = {"w_down": w_down, "w_up": w_up}

    opt = optax.adamw(learning_rate=args.lr, b1=0.9, b2=0.999, weight_decay=0.0)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, batch_idx):
        def loss_fn(p):
            xb = x[batch_idx]  # [B,S,H]
            kb = k_rope[batch_idx]  # [B,S,kv_heads,head_dim]
            z = jnp.einsum("bsh,hr->bsr", xb, p["w_down"])  # [B,S,r]
            k_hat_nope = jnp.einsum("bsr,hrd->bshd", z, p["w_up"])  # [B,S,kv_heads,head_dim]
            q0 = jnp.zeros_like(k_hat_nope)
            _, k_hat_rope = rope(pos, q0, k_hat_nope.astype(jnp.bfloat16))
            diff = (k_hat_rope.astype(jnp.float32) - kb).astype(jnp.float32)
            return jnp.mean(jnp.square(diff))

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state2 = opt.update(grads, opt_state, params)
        params2 = optax.apply_updates(params, updates)
        return params2, opt_state2, loss

    losses = []
    t0 = time.time()
    for t in range(args.steps):
        idx = np.random.randint(0, n, size=(args.batch_size,))
        params, opt_state, loss = step(params, opt_state, jnp.asarray(idx, dtype=jnp.int32))
        lf = float(loss)
        losses.append(lf)
        if (t + 1) % 20 == 0:
            print(f"[step {t+1:05d}] loss={lf:.6f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        w_down=np.array(jax.device_get(params["w_down"])),
        w_up=np.array(jax.device_get(params["w_up"])),
        losses=np.array(losses, dtype=np.float32),
        cache_path=str(Path(args.cache).resolve()),
    )
    print(f"[done] wrote {out_path} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
