#!/usr/bin/env python3
"""Build a clean-context layer0 cache for DeepSeek-native MLA experiments (TPU/JAX).

This produces baseline (teacher) tensors for GPT-OSS layer 0 *without* running the full model:
  x_attn_in = RMSNorm(embed(token_ids))
  q = x_attn_in @ Wq + bq  (reshape to [B,S,q_heads,head_dim])
  k = x_attn_in @ Wk + bk  (reshape to [B,S,kv_heads,head_dim])
  v = x_attn_in @ Wv + bv  (reshape to [B,S,kv_heads,head_dim])
  (q_rope, k_rope) = RoPE(q, k)

Outputs .npz with float32 tensors for reliable serialization.
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
        "q_w": "model.layers.0.self_attn.q_proj.weight",
        "q_b": "model.layers.0.self_attn.q_proj.bias",
        "k_w": "model.layers.0.self_attn.k_proj.weight",
        "k_b": "model.layers.0.self_attn.k_proj.bias",
        "v_w": "model.layers.0.self_attn.v_proj.weight",
        "v_b": "model.layers.0.self_attn.v_proj.bias",
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
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--storage-dtype",
        type=str,
        default="float16",
        choices=("float16", "float32"),
        help="Host-side dtype for cached activations (smaller is faster to write).",
    )
    parser.add_argument("--out", type=str, default=f"/dev/shm/out/kqkv_cache_layer0_{time.strftime('%Y%m%d_%H%M%S')}.npz")

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
    q_heads = int(cfg["num_attention_heads"])
    kv_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["head_dim"])
    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim

    # HF shapes: embed [vocab,hidden], ln [hidden], proj_w [out,hidden], proj_b [out]
    if tuple(tensors["q_w"].shape) != (q_dim, hidden):
        raise ValueError(f"Unexpected q_proj.weight {tuple(tensors['q_w'].shape)} expected {(q_dim, hidden)}")
    if tuple(tensors["k_w"].shape) != (kv_dim, hidden):
        raise ValueError(f"Unexpected k_proj.weight {tuple(tensors['k_w'].shape)} expected {(kv_dim, hidden)}")
    if tuple(tensors["v_w"].shape) != (kv_dim, hidden):
        raise ValueError(f"Unexpected v_proj.weight {tuple(tensors['v_w'].shape)} expected {(kv_dim, hidden)}")

    device = jax.devices("tpu")[0] if jax.default_backend() == "tpu" else jax.devices()[0]

    embed_w = jax.device_put(tensors["embed"], device).astype(jnp.bfloat16)
    ln_w = jax.device_put(tensors["in_ln"], device).astype(jnp.bfloat16)

    q_w = jax.device_put(tensors["q_w"], device).astype(jnp.bfloat16)
    q_b = jax.device_put(tensors["q_b"], device).astype(jnp.bfloat16)
    k_w = jax.device_put(tensors["k_w"], device).astype(jnp.bfloat16)
    k_b = jax.device_put(tensors["k_b"], device).astype(jnp.bfloat16)
    v_w = jax.device_put(tensors["v_w"], device).astype(jnp.bfloat16)
    v_b = jax.device_put(tensors["v_b"], device).astype(jnp.bfloat16)

    eps = float(cfg.get("rms_norm_eps", 1e-5))
    rope = _build_rope(cfg)
    pos = jax.device_put(jnp.arange(args.seq_len, dtype=jnp.int32)[None, :], device)

    # Host buffers
    storage_dtype = np.float16 if args.storage_dtype == "float16" else np.float32
    x_cache = np.empty((args.num_blocks, args.seq_len, hidden), dtype=storage_dtype)
    q_nope_cache = np.empty((args.num_blocks, args.seq_len, q_heads, head_dim), dtype=storage_dtype)
    k_nope_cache = np.empty((args.num_blocks, args.seq_len, kv_heads, head_dim), dtype=storage_dtype)
    q_rope_cache = np.empty((args.num_blocks, args.seq_len, q_heads, head_dim), dtype=storage_dtype)
    k_rope_cache = np.empty((args.num_blocks, args.seq_len, kv_heads, head_dim), dtype=storage_dtype)
    v_cache = np.empty((args.num_blocks, args.seq_len, kv_heads, head_dim), dtype=storage_dtype)

    input_ids = jax.device_put(jnp.asarray(input_ids_np, dtype=jnp.int32), device)

    t0 = time.time()
    for start in range(0, args.num_blocks, args.batch_size):
        end = min(args.num_blocks, start + args.batch_size)
        batch_ids = input_ids[start:end]

        x = embed_w[batch_ids]
        x_in = _rms_norm(x, ln_w, eps).astype(jnp.bfloat16)

        q = jnp.einsum("bsh,qh->bsq", x_in, q_w) + q_b
        k = jnp.einsum("bsh,kh->bsk", x_in, k_w) + k_b
        v = jnp.einsum("bsh,vh->bsv", x_in, v_w) + v_b

        q = q.reshape((q.shape[0], q.shape[1], q_heads, head_dim))
        k = k.reshape((k.shape[0], k.shape[1], kv_heads, head_dim))
        v = v.reshape((v.shape[0], v.shape[1], kv_heads, head_dim))

        q_rope, k_rope = rope(pos, q, k)

        x_cache[start:end] = np.array(jax.device_get(x_in), dtype=storage_dtype)
        q_nope_cache[start:end] = np.array(jax.device_get(q), dtype=storage_dtype)
        k_nope_cache[start:end] = np.array(jax.device_get(k), dtype=storage_dtype)
        q_rope_cache[start:end] = np.array(jax.device_get(q_rope), dtype=storage_dtype)
        k_rope_cache[start:end] = np.array(jax.device_get(k_rope), dtype=storage_dtype)
        v_cache[start:end] = np.array(jax.device_get(v), dtype=storage_dtype)

        print(f"[cache] {end}/{args.num_blocks}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        input_ids=input_ids_np,
        x_attn_in=x_cache,
        q_nope=q_nope_cache,
        k_nope=k_nope_cache,
        q_rope=q_rope_cache,
        k_rope=k_rope_cache,
        v=v_cache,
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
                "rope_theta": cfg.get("rope_theta"),
                "rope_scaling": cfg.get("rope_scaling"),
                "max_position_embeddings": cfg.get("max_position_embeddings"),
                "hidden_size": hidden,
                "head_dim": head_dim,
                "num_attention_heads": q_heads,
                "num_key_value_heads": kv_heads,
                "rms_norm_eps": cfg.get("rms_norm_eps"),
                "storage_dtype": args.storage_dtype,
            }
        ),
    )
    print(f"[done] wrote {out_path} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
