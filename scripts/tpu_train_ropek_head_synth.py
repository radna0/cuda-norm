#!/usr/bin/env python3
"""Train a tiny "RoPE‑K head synthesis" module on TPU (JAX) for GPT‑OSS.

Goal (quality‑first / DeepSeek‑native direction)
------------------------------------------------
Learn a shared latent per token z_t (rank=r) and per‑KV‑head up‑projection W_uK[h]
such that RoPE‑applied keys K_rope[h] are reconstructed well:

  x_norm = RMSNorm(emb(token_ids))
  K_teacher = RoPE( x_norm @ Wk )
  z = x_norm @ W_down
  K_pred = RoPE( z @ W_uK )

This script starts with *layer 0 only* (clean context), and trains only (W_down, W_uK).
It is a building block for "true DeepSeek‑native MLA" where head‑specific keys are
generated from a compact latent cache, rather than assuming RoPE‑K is shareable.

Notes
-----
- Reads GPT‑OSS weights directly from HF safetensors shards (local snapshot dir).
- Uses the *high‑quality calib packs* dataset (round‑robin across packs) for tokens.
- Uses EasyDeL's YaRN RoPE implementation to match GPT‑OSS RoPE semantics.
- Stores artifacts to /dev/shm by default.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _set_shm_caches():
    os.environ.setdefault("HF_HOME", "/dev/shm/hf")
    os.environ.setdefault("HF_HUB_CACHE", "/dev/shm/hf/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/dev/shm/hf/transformers")
    os.environ.setdefault("XDG_CACHE_HOME", "/dev/shm/xdg")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/dev/shm/jax_compilation_cache")


def _require_token_present():
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
        raise RuntimeError("Missing HF token in env (HF_TOKEN or HUGGINGFACE_HUB_TOKEN).")


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


@dataclass(frozen=True)
class ModelPaths:
    snapshot_dir: Path
    index_path: Path
    config_path: Path


def resolve_model_paths(snapshot_dir: str) -> ModelPaths:
    snapshot = Path(snapshot_dir).resolve()
    if not snapshot.exists():
        raise FileNotFoundError(snapshot)
    index_path = snapshot / "model.safetensors.index.json"
    config_path = snapshot / "config.json"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    return ModelPaths(snapshot_dir=snapshot, index_path=index_path, config_path=config_path)


def _weight_map(index_path: Path) -> dict[str, str]:
    idx = _load_json(index_path)
    return idx["weight_map"]


def _load_safetensor_tensor(snapshot_dir: Path, filename: str, tensor_name: str):
    """Load a single tensor as a JAX array (bf16 supported) via safetensors/flax."""

    from safetensors import safe_open

    path = snapshot_dir / filename
    with safe_open(str(path), framework="flax") as f:
        return f.get_tensor(tensor_name)


def _get_required_tensors(snapshot_dir: Path, weight_map: dict[str, str], layer_idx: int):
    required = {
        "embed": "model.embed_tokens.weight",
        "k_w": f"model.layers.{layer_idx}.self_attn.k_proj.weight",
        "k_b": f"model.layers.{layer_idx}.self_attn.k_proj.bias",
        "in_ln": f"model.layers.{layer_idx}.input_layernorm.weight",
    }
    out = {}
    for key, tensor_name in required.items():
        filename = weight_map.get(tensor_name)
        if filename is None:
            raise KeyError(f"Missing tensor in index: {tensor_name}")
        out[key] = _load_safetensor_tensor(snapshot_dir, filename, tensor_name)
    return out


def _rms_norm(x: "jnp.ndarray", weight: "jnp.ndarray", eps: float) -> "jnp.ndarray":
    import jax.numpy as jnp

    x2 = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    x_norm = x * jnp.reciprocal(jnp.sqrt(x2 + eps))
    return x_norm * weight


def _build_rope(cfg: dict):
    import jax.numpy as jnp
    from easydel.layers.rotary_embedding import get_rope

    return get_rope(
        head_size=int(cfg["head_dim"]),
        rotary_dim=int(cfg["head_dim"]),
        max_position=int(cfg["max_position_embeddings"]),
        base=int(cfg["rope_theta"]),
        is_neox_style=True,
        rope_scaling=cfg.get("rope_scaling"),
        dtype=jnp.bfloat16,
    )


def _load_union_texts(*, repo_id: str, data_files: list[str], max_rows_per_pack: int | None, seed: int) -> list[str]:
    from datasets import load_dataset

    _require_token_present()

    rng = random.Random(seed)
    packs = []
    for f in data_files:
        ds = load_dataset(repo_id, data_files=f, split="train", streaming=False)
        if max_rows_per_pack is not None:
            ds = ds.select(range(min(max_rows_per_pack, len(ds))))
        packs.append([row["text"] for row in ds])

    rr: list[str] = []
    max_len = max(len(p) for p in packs)
    for i in range(max_len):
        for p in packs:
            if i < len(p):
                rr.append(p[i])

    rng.shuffle(rr)
    return rr


def _tokenize_fixed_blocks(*, tokenizer_id_or_path: str, texts: list[str], seq_len: int, max_blocks: int, seed: int) -> np.ndarray:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id_or_path, local_files_only=True, use_fast=True)
    rng = random.Random(seed)
    rng.shuffle(texts)
    texts = texts[:max_blocks]

    toks = tokenizer(
        texts,
        truncation=True,
        max_length=seq_len,
        padding="max_length",
        return_tensors="np",
    )
    return toks["input_ids"].astype(np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-snapshot-dir",
        type=str,
        required=True,
        help="Local HF snapshot dir (must contain config.json + model.safetensors.index.json).",
    )
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--latent-rank", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=f"/dev/shm/out/ropek_head_synth_{_now_tag()}")

    parser.add_argument(
        "--calib-repo-id",
        type=str,
        default="radna0/harmony-qwen3-calib-packs-v2-20260113",
    )
    parser.add_argument(
        "--calib-data-files",
        type=str,
        default="packs/reasoning_style_10k_v2/reasoning_style_10k_v2.parquet,"
        "tool_agentic_10k_v6.parquet,"
        "packs/calib_prompt_10000_v2/calib_prompt_10000_v2.parquet",
    )
    parser.add_argument("--max-rows-per-pack", type=int, default=2000)
    parser.add_argument("--max-blocks", type=int, default=4000)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    _set_shm_caches()
    Path(os.environ["JAX_COMPILATION_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    model_paths = resolve_model_paths(args.model_snapshot_dir)
    cfg = _load_json(model_paths.config_path)
    weight_map = _weight_map(model_paths.index_path)

    tensors = _get_required_tensors(model_paths.snapshot_dir, weight_map, args.layer_idx)

    hidden = int(cfg["hidden_size"])
    kv_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["head_dim"])
    kv_dim = kv_heads * head_dim
    if tuple(tensors["k_w"].shape) != (kv_dim, hidden):
        raise ValueError(f"Unexpected k_proj.weight shape {tuple(tensors['k_w'].shape)}, expected {(kv_dim, hidden)}")
    if args.latent_rank > kv_dim:
        raise ValueError(f"latent_rank must be <= kv_dim (got {args.latent_rank} > {kv_dim})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "model_snapshot_dir": str(model_paths.snapshot_dir),
                "layer_idx": args.layer_idx,
                "latent_rank": args.latent_rank,
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "steps": args.steps,
                "lr": args.lr,
                "seed": args.seed,
                "calib_repo_id": args.calib_repo_id,
                "calib_data_files": args.calib_data_files,
                "max_rows_per_pack": args.max_rows_per_pack,
                "max_blocks": args.max_blocks,
            },
            indent=2,
        )
    )

    data_files = [s.strip() for s in args.calib_data_files.split(",") if s.strip()]
    texts = _load_union_texts(
        repo_id=args.calib_repo_id,
        data_files=data_files,
        max_rows_per_pack=args.max_rows_per_pack,
        seed=args.seed,
    )
    input_ids_np = _tokenize_fixed_blocks(
        tokenizer_id_or_path=str(model_paths.snapshot_dir),
        texts=texts,
        seq_len=args.seq_len,
        max_blocks=args.max_blocks,
        seed=args.seed,
    )

    import jax
    import jax.numpy as jnp
    import optax

    # Highest numerical precision for matmul/dot on JAX backends (incl TPU).
    # This must be set before the first compiled computation.
    try:
        from jax import config as jax_config
        jax_config.update("jax_default_matmul_precision", "highest")
    except Exception:
        pass

    device = jax.devices("tpu")[0] if jax.default_backend() == "tpu" else jax.devices()[0]

    embed_w = jax.device_put(tensors["embed"], device).astype(jnp.bfloat16)
    in_ln_w = jax.device_put(tensors["in_ln"], device).astype(jnp.bfloat16)
    k_w = jax.device_put(tensors["k_w"], device).astype(jnp.bfloat16)
    k_b = jax.device_put(tensors["k_b"], device).astype(jnp.bfloat16)

    eps = float(cfg.get("rms_norm_eps", 1e-5))
    rope = _build_rope(cfg)
    pos = jax.device_put(jnp.arange(args.seq_len, dtype=jnp.int32)[None, :], device)

    key = jax.random.PRNGKey(args.seed)
    key, k1, k2 = jax.random.split(key, 3)
    w_down = jax.random.normal(k1, (hidden, args.latent_rank), dtype=jnp.float32) * (1.0 / np.sqrt(hidden))
    w_up = jax.random.normal(k2, (kv_heads, args.latent_rank, head_dim), dtype=jnp.float32) * (
        1.0 / np.sqrt(args.latent_rank)
    )
    params = {"w_down": w_down.astype(jnp.bfloat16), "w_up": w_up.astype(jnp.bfloat16)}

    opt = optax.adamw(learning_rate=args.lr, b1=0.9, b2=0.999, weight_decay=0.0)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, batch_ids):
        def loss_fn(p):
            x = embed_w[batch_ids]
            x = _rms_norm(x, in_ln_w, eps).astype(jnp.bfloat16)

            k_full = jnp.einsum("bsh,kh->bsk", x, k_w) + k_b
            k_full = k_full.reshape((k_full.shape[0], k_full.shape[1], kv_heads, head_dim))

            z = jnp.einsum("bsh,hr->bsr", x, p["w_down"])
            k_hat = jnp.einsum("bsr,hrd->bshd", z, p["w_up"])

            q0 = jnp.zeros_like(k_full)
            _, k_rope = rope(pos, q0, k_full)
            _, k_hat_rope = rope(pos, q0, k_hat)

            diff = (k_hat_rope - k_rope).astype(jnp.float32)
            return jnp.mean(jnp.square(diff))

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state2 = opt.update(grads, opt_state, params)
        params2 = optax.apply_updates(params, updates)
        return params2, opt_state2, loss

    input_ids = jax.device_put(jnp.asarray(input_ids_np, dtype=jnp.int32), device)
    n = int(input_ids.shape[0])

    losses = []
    for t in range(args.steps):
        idx = np.random.randint(0, n, size=(args.batch_size,))
        batch = input_ids[idx]
        params, opt_state, loss = step(params, opt_state, batch)
        loss_f = float(loss)
        losses.append(loss_f)
        if (t + 1) % 10 == 0:
            print(f"[step {t+1:05d}] loss={loss_f:.6f}")

    np.savez(
        out_dir / "ropek_head_synth_params.npz",
        w_down=np.array(jax.device_get(params["w_down"])),
        w_up=np.array(jax.device_get(params["w_up"])),
        losses=np.array(losses, dtype=np.float32),
    )
    print(f"[done] saved {out_dir}/ropek_head_synth_params.npz")


if __name__ == "__main__":
    main()
