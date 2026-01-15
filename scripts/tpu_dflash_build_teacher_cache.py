#!/usr/bin/env python3
"""Build a teacher cache for DFlash draft training on TPU (EasyDeL + JAX).

This script does *no draft training*. It only runs the (expensive) teacher forward
once per block and stores the conditioning features needed for draft training:

- context_features: concat of selected layer hidden states for each prefix token
  shape [N, ctx_len, K * hidden]
- anchor_embedding: embedding of the last prefix token (verified_id)
  shape [N, hidden]
- target_ids: the (block_size-1) ground-truth next tokens
  shape [N, block_size-1]

All outputs are written under `/dev/shm` by default to avoid disk IO bottlenecks.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _tokenize_blocks(*, tokenizer_path: str, texts: list[str], total_len: int, num_blocks: int, seed: int) -> np.ndarray:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, use_fast=True)
    rng = random.Random(int(seed))
    rng.shuffle(texts)
    texts = texts[: int(num_blocks)]
    toks = tok(
        texts,
        truncation=True,
        max_length=int(total_len),
        padding="max_length",
        return_tensors="np",
    )
    return toks["input_ids"].astype(np.int32)


def _load_union_texts(*, repo_id: str, data_files: list[str], max_rows_per_pack: int | None, seed: int) -> list[str]:
    from datasets import load_dataset

    rng = random.Random(int(seed))
    packs: list[list[str]] = []
    for f in data_files:
        ds = load_dataset(str(repo_id), data_files=str(f), split="train", streaming=False)
        if max_rows_per_pack is not None:
            ds = ds.select(range(min(int(max_rows_per_pack), len(ds))))
        packs.append([row["text"] for row in ds])

    rr: list[str] = []
    max_len = max(len(p) for p in packs)
    for i in range(max_len):
        for p in packs:
            if i < len(p):
                rr.append(p[i])

    rng.shuffle(rr)
    return rr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-snapshot-dir",
        required=True,
        help="Local HF snapshot dir (contains config.json + model.safetensors.index.json).",
    )
    ap.add_argument(
        "--teacher-easydel-dir",
        default="",
        help="Optional EasyDeL-native checkpoint directory. If set and exists, loads teacher from here (fast).",
    )
    ap.add_argument(
        "--save-teacher-easydel-dir",
        default="",
        help="If set, saves a converted EasyDeL-native checkpoint here after first load (speeds up future runs).",
    )
    ap.add_argument("--ctx-len", type=int, default=4096)
    ap.add_argument("--block-size", type=int, default=8, help="DFLASH verify window length (includes anchor token).")
    ap.add_argument(
        "--num-context-features",
        type=int,
        default=4,
        help="K = number of target layers to concatenate per token.",
    )
    ap.add_argument(
        "--target-layer-ids",
        default="",
        help="Override target layer ids (comma-separated). Default: evenly-spaced from model depth.",
    )
    ap.add_argument("--num-blocks", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        default=f"/dev/shm/out/dflash_teacher_cache_{_now_tag()}.npz",
        help="Legacy single-file .npz output (fine for small runs, not mmap-friendly).",
    )
    ap.add_argument(
        "--out-dir",
        default="",
        help="If set, writes an mmap-friendly cache directory (meta.json + .npy files).",
    )
    ap.add_argument(
        "--sharding-axis-dims",
        default="1,8,1,1,1",
        help="5D sharding axis dims (dp,fsdp,ep,tp,sp). Default fits v6e-8 single host.",
    )

    ap.add_argument("--calib-repo-id", default="radna0/harmony-qwen3-calib-packs-v2-20260113")
    ap.add_argument(
        "--calib-data-files",
        default="packs/reasoning_style_10k_v2/reasoning_style_10k_v2.parquet,"
        "tool_agentic_10k_v6.parquet,"
        "packs/calib_prompt_10000_v2/calib_prompt_10000_v2.parquet",
    )
    ap.add_argument("--max-rows-per-pack", type=int, default=2000)

    ap.add_argument(
        "--platform",
        default="tpu",
        choices=["tpu", "cpu"],
        help="For debugging only. Use 'cpu' if TPU is busy.",
    )
    args = ap.parse_args()

    from tpu_dflash_lib import (
        build_target_layer_ids,
        require_hf_token,
        set_shm_caches,
    )

    set_shm_caches()
    require_hf_token()

    if args.platform == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import jax
    import jax.numpy as jnp

    import ml_dtypes
    from easydel import AutoEasyDeLModelForCausalLM

    snapshot = Path(args.model_snapshot_dir).resolve()
    cfg = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))

    total_len = int(args.ctx_len) + max(1, int(args.block_size) - 1)
    data_files = [s.strip() for s in str(args.calib_data_files).split(",") if s.strip()]
    texts = _load_union_texts(
        repo_id=str(args.calib_repo_id),
        data_files=data_files,
        max_rows_per_pack=int(args.max_rows_per_pack) if args.max_rows_per_pack is not None else None,
        seed=int(args.seed),
    )
    input_ids_np = _tokenize_blocks(
        tokenizer_path=str(snapshot),
        texts=texts,
        total_len=total_len,
        num_blocks=int(args.num_blocks),
        seed=int(args.seed),
    )

    if args.target_layer_ids.strip():
        target_layer_ids = [int(x) for x in args.target_layer_ids.split(",") if x.strip()]
    else:
        target_layer_ids = build_target_layer_ids(int(cfg["num_hidden_layers"]), int(args.num_context_features))

    print(
        "[cache] start "
        f"platform={args.platform} ctx_len={int(args.ctx_len)} block_size={int(args.block_size)} "
        f"num_blocks={int(args.num_blocks)} batch_size={int(args.batch_size)} "
        f"K={len(target_layer_ids)} target_layer_ids={target_layer_ids}",
        flush=True,
    )

    sharding_axis_dims = tuple(int(x) for x in str(args.sharding_axis_dims).split(",") if x.strip())
    if len(sharding_axis_dims) != 5:
        raise ValueError("--sharding-axis-dims must have 5 comma-separated ints (dp,fsdp,ep,tp,sp)")

    teacher_easydel_dir = Path(str(args.teacher_easydel_dir)).resolve() if str(args.teacher_easydel_dir).strip() else None
    save_teacher_easydel_dir = (
        Path(str(args.save_teacher_easydel_dir)).resolve() if str(args.save_teacher_easydel_dir).strip() else None
    )

    t_load0 = time.time()
    if teacher_easydel_dir is not None and teacher_easydel_dir.exists():
        print(f"[teacher] loading EasyDeL checkpoint: {teacher_easydel_dir}", flush=True)
        model = AutoEasyDeLModelForCausalLM.from_pretrained(
            str(teacher_easydel_dir),
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            auto_shard_model=True,
            sharding_axis_dims=sharding_axis_dims,
            verbose=False,
            from_torch=False,
        )
    else:
        print(f"[teacher] converting from torch snapshot: {snapshot}", flush=True)
        model = AutoEasyDeLModelForCausalLM.from_pretrained(
            str(snapshot),
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            auto_shard_model=True,
            sharding_axis_dims=sharding_axis_dims,
            verbose=False,
            from_torch=True,
        )
        if save_teacher_easydel_dir is not None:
            save_teacher_easydel_dir.mkdir(parents=True, exist_ok=True)
            print(f"[teacher] saving EasyDeL checkpoint: {save_teacher_easydel_dir}", flush=True)
            model.save_pretrained(str(save_teacher_easydel_dir))
    print(f"[teacher] ready in {time.time() - t_load0:.1f}s", flush=True)

    hidden = int(cfg["hidden_size"])
    k = int(len(target_layer_ids))

    n = int(args.num_blocks)
    ctx_len = int(args.ctx_len)
    block_size = int(args.block_size)

    # Host arrays (bf16) in /dev/shm.
    ctx_feats = np.empty((n, ctx_len, k * hidden), dtype=ml_dtypes.bfloat16)
    anchor_emb = np.empty((n, hidden), dtype=ml_dtypes.bfloat16)
    target_ids = np.empty((n, block_size - 1), dtype=np.int32)
    anchor_ids = np.empty((n,), dtype=np.int32)

    input_ids = jnp.asarray(input_ids_np, dtype=jnp.int32)
    pos = jnp.arange(ctx_len, dtype=jnp.int32)[None, :]

    # Compile a single forward once, then reuse it for each batch. IMPORTANT: we
    # must not close over the whole `model` in a `jax.jit`, or the parameters may
    # get baked in as giant constants. Use NNX graphdef/graphstate/graphother.
    from flax import nnx as _nnx

    graphdef, graphstate, graphother = _nnx.split(model, _nnx.Param, ...)

    def _forward(gs, ctx):
        module = _nnx.merge(graphdef, gs, graphother)
        out = module(
            input_ids=ctx,
            output_hidden_states=True,
            apply_lm_head=False,
        )
        hs = out.hidden_states
        if hs is None:
            raise RuntimeError("Teacher did not return hidden_states (expected output_hidden_states=True).")
        parts = [hs[int(lid)] for lid in target_layer_ids]
        feat = jnp.concatenate(parts, axis=-1)
        a_id = ctx[:, -1]
        emb = module.get_embedding()(a_id.astype("i4"))
        return feat, emb, a_id

    forward = jax.jit(_forward)

    t0 = time.time()
    for start in range(0, n, int(args.batch_size)):
        end = min(n, start + int(args.batch_size))
        batch = input_ids[start:end]  # [B, total_len]
        ctx = batch[:, :ctx_len]
        tgt = batch[:, ctx_len : ctx_len + (block_size - 1)]
        with model.mesh:
            feat, emb, a_id = forward(graphstate, ctx)
            # Ensure we only time finished device work, not async dispatch.
            feat, emb, a_id = jax.tree_util.tree_map(jax.block_until_ready, (feat, emb, a_id))

        ctx_feats[start:end] = np.asarray(jax.device_get(feat), dtype=ml_dtypes.bfloat16)
        anchor_emb[start:end] = np.asarray(jax.device_get(emb), dtype=ml_dtypes.bfloat16)
        target_ids[start:end] = np.asarray(jax.device_get(tgt), dtype=np.int32)
        anchor_ids[start:end] = np.asarray(jax.device_get(a_id), dtype=np.int32)
        print(f"[cache] {end}/{n}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "model_snapshot_dir": str(snapshot),
        "platform": str(args.platform),
        "ctx_len": int(ctx_len),
        "block_size": int(block_size),
        "num_blocks": int(n),
        "batch_size": int(args.batch_size),
        "target_layer_ids": target_layer_ids,
        "hidden_size": int(hidden),
        "num_context_features": int(k),
        "dtype": "bf16_u16",
        "seed": int(args.seed),
        "calib_repo_id": str(args.calib_repo_id),
        "calib_data_files": data_files,
        "max_rows_per_pack": int(args.max_rows_per_pack),
        "wall_s": float(time.time() - t0),
    }
    # NOTE: NumPy does not have a native bfloat16 dtype. If we save ml_dtypes.bfloat16
    # directly, it gets serialized as a void dtype (e.g. |V2), which is annoying to
    # load robustly. Save bf16 tensors as raw uint16 bitpatterns instead.
    ctx_feats_u16 = ctx_feats.view(np.uint16)
    anchor_emb_u16 = anchor_emb.view(np.uint16)

    # Write mmap-friendly directory if requested.
    if str(args.out_dir).strip():
        out_dir = Path(str(args.out_dir)).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        np.save(out_dir / "context_features_u16.npy", ctx_feats_u16)
        np.save(out_dir / "anchor_embedding_u16.npy", anchor_emb_u16)
        np.save(out_dir / "anchor_ids.npy", anchor_ids)
        np.save(out_dir / "target_ids.npy", target_ids)
        print(f"[done] wrote cache_dir={out_dir}", flush=True)

    # Always write legacy .npz output (useful for quick smoke tests).
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        context_features_u16=ctx_feats_u16,
        anchor_embedding_u16=anchor_emb_u16,
        anchor_ids=anchor_ids,
        target_ids=target_ids,
        meta=json.dumps(meta),
    )
    print(f"[done] wrote npz={out_path}", flush=True)


if __name__ == "__main__":
    main()
