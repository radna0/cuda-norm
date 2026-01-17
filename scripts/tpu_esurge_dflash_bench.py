#!/usr/bin/env python3
"""TPU DFlash benchmark using eSurge KV cache + verify executor (no HF use_cache).

This is the TPU-first benchmarking harness for DFlash speedups.
It compares:
  - baseline greedy (1 token/step)
  - DFlash block-verify (block_size tokens/step)
using the same target model and the draft checkpoint trained earlier.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-snapshot-dir", required=True)
    ap.add_argument(
        "--teacher-easydel-dir",
        default="",
        help="Optional EasyDeL-native checkpoint dir (speeds up load on TPU).",
    )
    ap.add_argument("--draft-run-dir", required=True, help="EasyDeL run-* directory with prefix=model/")
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--page-size", type=int, default=128)
    ap.add_argument("--hbm-utilization", type=float, default=0.5)
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--prompt-repeat", type=int, default=1, help="Repeat the prompt text to reach long contexts (tokenizer-dependent).")
    ap.add_argument(
        "--prompt-from-cache-dir",
        default="",
        help="If set, ignore the text prompt and instead use prompt ids from a DFlash teacher cache dir "
        "(ctx_token_ids.npy + anchor_ids.npy). This is the fastest way to benchmark on training distribution.",
    )
    ap.add_argument(
        "--cache-sample-idx",
        type=int,
        default=0,
        help="Index into ctx_token_ids/anchor_ids when --prompt-from-cache-dir is set.",
    )
    ap.add_argument("--also-run-baseline", action="store_true", help="Also run baseline greedy (slow) for speedup_x.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    # Prefer the local EasyDeL checkout (we are modifying EasyDeL source directly).
    local_easydel = repo_root / "external" / "EasyDeL"
    if local_easydel.exists():
        sys.path.insert(0, str(local_easydel))
    _load_dotenv(repo_root / ".env")
    os.environ.setdefault("EASYDEL_SKIP_VERSION_CHECK", "1")
    os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

    import jax.numpy as jnp
    from transformers import AutoTokenizer
    from easydel import AutoEasyDeLModelForCausalLM
    from easydel.layers.rotary_embedding import get_rope
    from easydel.inference.speculative import bench_esurge_dflash_decode_single
    from easydel.inference.speculative import DFlashDraftModelConfig

    def _load_lm_head_weight(snapshot_dir: Path):
        from safetensors import safe_open

        name_candidates = ("lm_head.weight", "model.lm_head.weight")
        index_path = snapshot_dir / "model.safetensors.index.json"
        if index_path.exists():
            idx = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = idx.get("weight_map", {})
            for name in name_candidates:
                shard = weight_map.get(name)
                if shard is None:
                    continue
                with safe_open(str(snapshot_dir / shard), framework="flax") as f:
                    return f.get_tensor(name)
            raise KeyError(f"Missing {name_candidates} in {index_path.name}")

        single_path = snapshot_dir / "model.safetensors"
        if not single_path.exists():
            raise FileNotFoundError(f"Missing {index_path.name} and {single_path.name} in {snapshot_dir}")
        with safe_open(str(single_path), framework="flax") as f:
            for name in name_candidates:
                if name in f.keys():
                    return f.get_tensor(name)
        raise KeyError(f"Missing {name_candidates} in {single_path.name}")

    teacher_snapshot = Path(args.teacher_snapshot_dir).resolve()
    prompt_from_cache_dir = str(args.prompt_from_cache_dir).strip()
    position_offset = 0
    if prompt_from_cache_dir:
        cache_dir = Path(prompt_from_cache_dir).expanduser().resolve()
        ctx_tokens = np.load(cache_dir / "ctx_token_ids.npy", mmap_mode="r")
        anchor_ids = np.load(cache_dir / "anchor_ids.npy", mmap_mode="r")
        i = int(args.cache_sample_idx)
        if i < 0 or i >= int(ctx_tokens.shape[0]):
            raise ValueError(f"--cache-sample-idx={i} out of range (0..{int(ctx_tokens.shape[0]) - 1})")
        ctx = np.asarray(ctx_tokens[i], dtype=np.int32)
        anchor = int(np.asarray(anchor_ids[i]))
        prompt_ids = np.concatenate([ctx, np.asarray([anchor], dtype=np.int32)], axis=0)
        pos_path = cache_dir / "ctx_pos_start_i32.npy"
        if pos_path.exists():
            try:
                pos_arr = np.load(pos_path, mmap_mode="r")
                position_offset = int(np.asarray(pos_arr[i]))
            except Exception:
                position_offset = 0
    else:
        tok = AutoTokenizer.from_pretrained(str(teacher_snapshot), local_files_only=True, use_fast=True)
        base_prompt = "You are a helpful assistant.\n\nUser: Explain speculative decoding in one paragraph.\nAssistant:"
        prompt = (base_prompt + "\n") * max(1, int(args.prompt_repeat))
        enc = tok(prompt, truncation=True, max_length=int(args.prompt_len), return_tensors="np")
        prompt_ids = enc["input_ids"][0].astype(np.int32)

    teacher_easydel_dir = Path(str(args.teacher_easydel_dir)).resolve() if str(args.teacher_easydel_dir).strip() else None
    if teacher_easydel_dir is not None and teacher_easydel_dir.exists():
        teacher = AutoEasyDeLModelForCausalLM.from_pretrained(
            str(teacher_easydel_dir),
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            auto_shard_model=True,
            sharding_axis_dims=(1, 8, 1, 1, 1),
            verbose=False,
            from_torch=False,
        )
    else:
        teacher = AutoEasyDeLModelForCausalLM.from_pretrained(
            str(teacher_snapshot),
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            auto_shard_model=True,
            sharding_axis_dims=(1, 8, 1, 1, 1),
            verbose=False,
            from_torch=True,
        )
    # TPU correctness: our runtime ships a compatibility implementation of
    # ragged_page_attention_v2 (gathers KV from pages). The v3 path requires a
    # full EJKernel v3 implementation and otherwise breaks multi-token verify.
    # Force v2 so DFlash verify blocks are causal + KV-correct.
    if os.environ.get("DFLASH_FORCE_RAGGED_V2", "1").lower() in ("1", "true", "yes", "y", "on"):
        teacher = teacher.merge_module(
            teacher.new_graphdef(attn_mechanism="ragged_page_attention_v2"),
            teacher.graphstate,
            teacher.graphother,
        )

    cfg = json.loads((teacher_snapshot / "config.json").read_text(encoding="utf-8"))
    rope = get_rope(
        head_size=int(cfg["head_dim"]),
        rotary_dim=int(cfg["head_dim"]),
        max_position=int(cfg["max_position_embeddings"]),
        base=int(cfg["rope_theta"]),
        is_neox_style=True,
        rope_scaling=cfg.get("rope_scaling"),
        dtype=jnp.bfloat16,
    )
    # Use the same frozen LM head weights as DFlash training, so draft-token
    # argmax is comparable to training semantics.
    lm_w = _load_lm_head_weight(teacher_snapshot)

    # Draft cfg must match the training run. Load it from run_dir/config.json so
    # we don't silently benchmark with a mismatched architecture (K/layers/etc.).
    run_dir = Path(args.draft_run_dir).resolve()
    run_cfg_path = run_dir / "config.json"
    if not run_cfg_path.exists():
        raise FileNotFoundError(f"Missing {run_cfg_path} (expected EasyDeL DFlash run config).")
    draft_cfg_dict = json.loads(run_cfg_path.read_text(encoding="utf-8"))

    # Back-compat: early TPU draft runs did not persist `target_layer_ids` into
    # run-*/config.json. For parity (and correct verify feature capture), infer
    # from the cache meta referenced by run_config.json next to the runs.
    if not draft_cfg_dict.get("target_layer_ids"):
        run_root_cfg = run_dir.parent / "run_config.json"
        cache_meta = None
        if run_root_cfg.exists():
            try:
                run_root = json.loads(run_root_cfg.read_text(encoding="utf-8"))
                cache_dir = run_root.get("cache_dir")
                if cache_dir:
                    meta_path = Path(str(cache_dir)).expanduser().resolve() / "meta.json"
                    if meta_path.exists():
                        cache_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                cache_meta = None
        if cache_meta and cache_meta.get("target_layer_ids"):
            draft_cfg_dict["target_layer_ids"] = [int(x) for x in cache_meta["target_layer_ids"]]
        # Default in all training scripts is True; persist it here for parity.
        draft_cfg_dict.setdefault("add_one_for_pre_layer_capture", True)
    # Allow overriding block_size from CLI (weights are independent of block size),
    # but keep all other fields fixed to what training used.
    draft_cfg_dict["block_size"] = int(args.block_size)
    draft_cfg = DFlashDraftModelConfig(**draft_cfg_dict)

    dflash_res, base_res = bench_esurge_dflash_decode_single(
        teacher=teacher,
        prompt_ids=prompt_ids,
        draft_run_dir=str(run_dir),
        draft_cfg=draft_cfg,
        target_rope=rope,
        lm_head_weight=lm_w,
        position_offset=int(position_offset),
        block_size=int(args.block_size),
        max_new_tokens=int(args.max_new_tokens),
        max_model_len=int(args.max_model_len),
        page_size=int(args.page_size),
        hbm_utilization=float(args.hbm_utilization),
        also_run_baseline=bool(args.also_run_baseline),
    )

    print(json.dumps({"dflash": dflash_res.__dict__, "baseline": base_res.__dict__ if base_res else None}, indent=2))


if __name__ == "__main__":
    main()
