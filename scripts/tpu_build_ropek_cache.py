#!/usr/bin/env python3
"""Build a clean-context cache for RoPE-K head synthesis training (TPU/JAX).

This is the "clean context" analogue: we cache baseline tensors computed directly
from baseline weights and *baseline* token blocks.

For now we implement **layer 0** fast-path without running the full model:
  x_attn_in = RMSNorm(embed(token_ids))
  k_rope_teacher = RoPE( x_attn_in @ Wk + bk )

Outputs a small .npz containing:
  - input_ids: int32 [N,S]
  - x_attn_in: float32 [N,S,H]
  - k_rope: float32 [N,S,kv_heads,head_dim]

All outputs are float32 so they serialize reliably with numpy.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
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


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _weight_map(index_path: Path) -> dict[str, str]:
    idx = _load_json(index_path)
    return idx["weight_map"]


def _load_safetensor_tensor(snapshot_dir: Path, filename: str, tensor_name: str):
    from safetensors import safe_open

    path = snapshot_dir / filename
    with safe_open(str(path), framework="flax") as f:
        return f.get_tensor(tensor_name)


def _get_layer0_tensors(snapshot_dir: Path, weight_map: dict[str, str]):
    required = {
        "embed": "model.embed_tokens.weight",
        "in_ln": "model.layers.0.input_layernorm.weight",
        "k_w": "model.layers.0.self_attn.k_proj.weight",
        "k_b": "model.layers.0.self_attn.k_proj.bias",
    }
    out = {}
    for key, tensor_name in required.items():
        filename = weight_map.get(tensor_name)
        if filename is None:
            raise KeyError(f"Missing tensor in index: {tensor_name}")
        out[key] = _load_safetensor_tensor(snapshot_dir, filename, tensor_name)
    return out


def _rms_norm(x, weight, eps):
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


def _tokenize_fixed_blocks(*, tokenizer_id_or_path: str, texts: list[str], seq_len: int, num_blocks: int, seed: int) -> np.ndarray:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id_or_path, local_files_only=True, use_fast=True)
    rng = random.Random(seed)
    rng.shuffle(texts)
    texts = texts[:num_blocks]

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
    parser.add_argument("--model-snapshot-dir", type=str, required=True)
    parser.add_argument("--layer-idx", type=int, default=0, help="Only layer 0 supported in fast path.")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=f"/dev/shm/out/ropek_cache_{time.strftime('%Y%m%d_%H%M%S')}.npz")

    parser.add_argument("--calib-repo-id", type=str, default="radna0/harmony-qwen3-calib-packs-v2-20260113")
    parser.add_argument(
        "--calib-data-files",
        type=str,
        default="packs/reasoning_style_10k_v2/reasoning_style_10k_v2.parquet,"
        "tool_agentic_10k_v6.parquet,"
        "packs/calib_prompt_10000_v2/calib_prompt_10000_v2.parquet",
    )
    parser.add_argument("--max-rows-per-pack", type=int, default=2000)

    args = parser.parse_args()

    if args.layer_idx != 0:
        raise ValueError("Only --layer-idx 0 is supported by this cache builder fast path.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    _set_shm_caches()
    Path(os.environ["JAX_COMPILATION_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    import jax
    import jax.numpy as jnp

    from jax import config as jax_config

    jax_config.update("jax_default_matmul_precision", "highest")

    snapshot = Path(args.model_snapshot_dir).resolve()
    cfg = _load_json(snapshot / "config.json")
    weight_map = _weight_map(snapshot / "model.safetensors.index.json")

    data_files = [s.strip() for s in args.calib_data_files.split(",") if s.strip()]
    texts = _load_union_texts(
        repo_id=args.calib_repo_id,
        data_files=data_files,
        max_rows_per_pack=args.max_rows_per_pack,
        seed=args.seed,
    )
    input_ids_np = _tokenize_fixed_blocks(
        tokenizer_id_or_path=str(snapshot),
        texts=texts,
        seq_len=args.seq_len,
        num_blocks=args.num_blocks,
        seed=args.seed,
    )

    tensors = _get_layer0_tensors(snapshot, weight_map)

    hidden = int(cfg["hidden_size"])
    kv_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["head_dim"])
    kv_dim = kv_heads * head_dim

    # HF shapes: embed [vocab,hidden], ln [hidden], k_w [kv_dim,hidden], k_b [kv_dim]
    embed_w = tensors["embed"]
    ln_w = tensors["in_ln"]
    k_w = tensors["k_w"]
    k_b = tensors["k_b"]

    if tuple(k_w.shape) != (kv_dim, hidden):
        raise ValueError(f"Unexpected k_proj.weight shape {tuple(k_w.shape)}, expected {(kv_dim, hidden)}")

    device = jax.devices("tpu")[0] if jax.default_backend() == "tpu" else jax.devices()[0]
    embed_w = jax.device_put(embed_w, device).astype(jnp.bfloat16)
    ln_w = jax.device_put(ln_w, device).astype(jnp.bfloat16)
    k_w = jax.device_put(k_w, device).astype(jnp.bfloat16)
    k_b = jax.device_put(k_b, device).astype(jnp.bfloat16)

    eps = float(cfg.get("rms_norm_eps", 1e-5))
    rope = _build_rope(cfg)
    pos = jax.device_put(jnp.arange(args.seq_len, dtype=jnp.int32)[None, :], device)

    # Host buffers
    x_cache = np.empty((args.num_blocks, args.seq_len, hidden), dtype=np.float32)
    k_cache = np.empty((args.num_blocks, args.seq_len, kv_heads, head_dim), dtype=np.float32)

    input_ids = jax.device_put(jnp.asarray(input_ids_np, dtype=jnp.int32), device)

    t0 = time.time()
    for start in range(0, args.num_blocks, args.batch_size):
        end = min(args.num_blocks, start + args.batch_size)
        batch_ids = input_ids[start:end]

        x = embed_w[batch_ids]  # [B,S,H]
        x_in = _rms_norm(x, ln_w, eps).astype(jnp.bfloat16)

        # k_full: [B,S,kv_dim]
        k_full = jnp.einsum("bsh,kh->bsk", x_in, k_w) + k_b
        k_full = k_full.reshape((k_full.shape[0], k_full.shape[1], kv_heads, head_dim))

        q0 = jnp.zeros_like(k_full)
        _, k_rope = rope(pos, q0, k_full)

        x_cache[start:end] = np.array(jax.device_get(x_in), dtype=np.float32)
        k_cache[start:end] = np.array(jax.device_get(k_rope), dtype=np.float32)
        print(f"[cache] {end}/{args.num_blocks}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        input_ids=input_ids_np,
        x_attn_in=x_cache,
        k_rope=k_cache,
        layer_idx=np.array([args.layer_idx], dtype=np.int32),
        seq_len=np.array([args.seq_len], dtype=np.int32),
        meta=json.dumps(
            {
                "model_snapshot_dir": str(snapshot),
                "calib_repo_id": args.calib_repo_id,
                "calib_data_files": data_files,
                "max_rows_per_pack": args.max_rows_per_pack,
                "num_blocks": args.num_blocks,
                "batch_size": args.batch_size,
                "seed": args.seed,
            }
        ),
    )
    print(f"[done] wrote {out_path} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
