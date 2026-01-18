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
import sys
import time
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


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _parse_csv_ints(s: str) -> list[int]:
    out: list[int] = []
    for part in (s or "").split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out


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
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    # Prefer the local EasyDeL checkout (we are modifying EasyDeL source directly).
    local_easydel = repo_root / "external" / "EasyDeL"
    if local_easydel.exists():
        sys.path.insert(0, str(local_easydel))

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
    ap.add_argument(
        "--rollout-steps",
        type=int,
        default=1,
        help="How many consecutive DFlash steps to simulate per base block (produces num_blocks*rollout_steps samples).",
    )
    ap.add_argument(
        "--rollout-accept-len-mode",
        type=str,
        default="full",
        choices=["full", "uniform", "geometric"],
        help=(
            "How to advance the synthetic teacher-only rollout after each verify block.\n"
            "\n"
            "In real DFlash decoding, accept_len is often < (block_size-1) early; if we build caches assuming perfect\n"
            "acceptance, the next-step contexts seen during inference diverge from training distribution and\n"
            "acceptance can collapse after the first few blocks.\n"
            "\n"
            "full: accept_len = block_size-1 (old behavior).\n"
            "uniform: accept_len ~ Uniform{0..block_size-1}.\n"
            "geometric: accept_len sampled from geometric(p) biased toward small values.\n"
            "\n"
            "This changes only the rollout state transition (next ctx_ids/anchor_id). Labels remain teacher greedy tokens."
        ),
    )
    ap.add_argument(
        "--rollout-accept-len-p",
        type=float,
        default=0.35,
        help="Geometric accept_len parameter p (only used when --rollout-accept-len-mode=geometric).",
    )
    ap.add_argument(
        "--rollout-state-evolution",
        type=str,
        default="dflash_commit",
        choices=["dflash_commit", "teacher_commit"],
        help=(
            "How to advance the synthetic rollout token history after each verify.\n"
            "\n"
            "dflash_commit (recommended): commit the *candidate* tokens that a draft would have produced, i.e.\n"
            "  committed = [anchor] + cand[1:1+accept_len]\n"
            "and set the next anchor to the target bonus token.\n"
            "\n"
            "teacher_commit: commit the target-predicted tokens (tgt_verify[:accept_len]) instead.\n"
            "This can drift from the actual KV/features used during verification if cand != tgt_verify.\n"
        ),
    )
    ap.add_argument("--batch-size", type=int, default=1, help="Blocks per host loop iteration (1 = simplest/correct).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--prefill-chunk",
        type=int,
        default=256,
        help="Prefill chunk size for eSurge verify-mode prefill (keeps TPU compile stable).",
    )
    ap.add_argument("--page-size", type=int, default=128, help="Paged-KV page size (must match downstream bench).")
    ap.add_argument("--hbm-utilization", type=float, default=0.20, help="HBM fraction reserved for KV cache (TPU).")
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
        "--stream-out-dir",
        type=str,
        default="true",
        help="If out_dir is set, write arrays directly into .npy memmaps (reduces peak RAM).",
    )
    ap.add_argument(
        "--write-npz",
        type=str,
        default="true",
        help="Whether to also write the legacy .npz output. For large runs, set to 'false' to avoid doubling /dev/shm usage.",
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
    ap.add_argument(
        "--position-offsets",
        default="0",
        help="Comma-separated absolute position offsets added to all position_ids during teacher prefill+verify. "
        "This trains positional parity for long decode without actually decoding to 65k/131k. "
        "Example: '0,4096,65536,129536'.",
    )
    ap.add_argument(
        "--position-offset-mode",
        default="per_prompt_rollout",
        choices=["cycle_samples", "per_prompt_rollout"],
        help=(
            "How to assign position offsets when writing cache samples.\n"
            "- cycle_samples: cycles the provided offsets every sample (old behavior; breaks rollout positional parity).\n"
            "- per_prompt_rollout: choose a base offset per prompt and keep it for the whole rollout, while advancing "
            "ctx_pos_start by block_size each rollout step to match real sliding-window decode."
        ),
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _load_dotenv(repo_root / ".env")

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
    from easydel.inference.esurge.runners.sequence_buffer import SequenceBuffer
    from easydel.inference.esurge.runners.execution_manager import ExecutionManager
    from easydel.inference.esurge.runners.states import CachedRequestState
    from easydel.inference.sampling_params import SamplingParams

    snapshot = Path(args.model_snapshot_dir).resolve()
    cfg = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))

    # Total length includes:
    # - context tokens (ctx_len-1)
    # - anchor token (1)
    # - draft targets (block_size-1)
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
    # TPU correctness: force ragged_page_attention_v2 so multi-token verify uses
    # the KV page table (our runtime has a correctness-first v2 fallback).
    if os.environ.get("DFLASH_FORCE_RAGGED_V2", "1").lower() in ("1", "true", "yes", "y", "on"):
        model = model.merge_module(
            model.new_graphdef(attn_mechanism="ragged_page_attention_v2"),
            model.graphstate,
            model.graphother,
        )
    print(f"[teacher] ready in {time.time() - t_load0:.1f}s", flush=True)

    hidden = int(cfg["hidden_size"])
    k = int(len(target_layer_ids))

    n = int(args.num_blocks)
    rollout_steps = int(max(1, int(args.rollout_steps)))
    # DFlash definition: context excludes the anchor token.
    ctx_len_full = int(args.ctx_len)
    ctx_len = int(max(1, ctx_len_full - 1))
    block_size = int(args.block_size)

    # Host arrays (bf16) in /dev/shm.
    n_out = int(n * rollout_steps)
    stream_out_dir = str(args.stream_out_dir).strip().lower() in ("1", "true", "yes", "y", "on")
    out_dir = Path(str(args.out_dir)).resolve() if str(args.out_dir).strip() else None
    using_memmap = bool(out_dir is not None and stream_out_dir)

    if using_memmap:
        from numpy.lib.format import open_memmap

        out_dir.mkdir(parents=True, exist_ok=True)
        ctx_feats_u16 = open_memmap(
            out_dir / "context_features_u16.npy",
            mode="w+",
            dtype=np.uint16,
            shape=(n_out, ctx_len, k * hidden),
        )
        anchor_emb_u16 = open_memmap(
            out_dir / "anchor_embedding_u16.npy",
            mode="w+",
            dtype=np.uint16,
            shape=(n_out, hidden),
        )
        target_ids = open_memmap(out_dir / "target_ids.npy", mode="w+", dtype=np.int32, shape=(n_out, block_size - 1))
        anchor_ids = open_memmap(out_dir / "anchor_ids.npy", mode="w+", dtype=np.int32, shape=(n_out,))
        ctx_token_ids = open_memmap(out_dir / "ctx_token_ids.npy", mode="w+", dtype=np.int32, shape=(n_out, ctx_len))
        ctx_pos_start_i32 = open_memmap(out_dir / "ctx_pos_start_i32.npy", mode="w+", dtype=np.int32, shape=(n_out,))
        anchor_pos_i32 = open_memmap(out_dir / "anchor_pos_i32.npy", mode="w+", dtype=np.int32, shape=(n_out,))
    else:
        ctx_feats_u16 = np.empty((n_out, ctx_len, k * hidden), dtype=np.uint16)
        anchor_emb_u16 = np.empty((n_out, hidden), dtype=np.uint16)
        target_ids = np.empty((n_out, block_size - 1), dtype=np.int32)
        anchor_ids = np.empty((n_out,), dtype=np.int32)
        ctx_token_ids = np.empty((n_out, ctx_len), dtype=np.int32)
    # Absolute position metadata for positional-parity training.
    # ctx positions are contiguous: [ctx_pos_start .. ctx_pos_start + ctx_len - 1]
    # anchor token position is anchor_pos.
        ctx_pos_start_i32 = np.empty((n_out,), dtype=np.int32)
        anchor_pos_i32 = np.empty((n_out,), dtype=np.int32)

    pos_offsets = _parse_csv_ints(str(args.position_offsets))
    if not pos_offsets:
        pos_offsets = [0]
    pos_offsets = [max(0, int(x)) for x in pos_offsets]

    # Clamp offsets so we never exceed max_position_embeddings during rollout.
    #
    # Positions used by verify can reach:
    #   ctx_pos_start + ctx_len + (block_size - 1)
    # During our (perfect-accept) rollouts we advance ctx_pos_start by block_size
    # each step, so the worst-case position occurs at step_idx=rollout_steps-1.
    max_pos_emb = int(cfg.get("max_position_embeddings", 0) or 0)
    if max_pos_emb > 0:
        max_pos_index = int(max_pos_emb - 1)
        max_ctx_start_one = int(max_pos_index - (int(ctx_len) + int(block_size) - 1))
        max_ctx_start_rollout = int(max_ctx_start_one - (int(rollout_steps) - 1) * int(block_size))
        if max_ctx_start_rollout < 0:
            raise ValueError(
                "Invalid max_position_embeddings for requested ctx_len/rollout: "
                f"max_position_embeddings={max_pos_emb} ctx_len={int(ctx_len)} block_size={int(block_size)} "
                f"rollout_steps={int(rollout_steps)}"
            )
        clamped: list[int] = []
        for x in pos_offsets:
            x_i = int(x)
            if x_i > max_ctx_start_rollout:
                print(
                    "[cache][warn] clamping position_offset "
                    f"{x_i} -> {int(max_ctx_start_rollout)} "
                    f"(max_position_embeddings={max_pos_emb}, ctx_len={int(ctx_len)}, block_size={int(block_size)}, rollout_steps={int(rollout_steps)})",
                    flush=True,
                )
                x_i = int(max_ctx_start_rollout)
            clamped.append(x_i)
        pos_offsets = clamped

    input_ids = np.asarray(input_ids_np, dtype=np.int32)
    rng = np.random.default_rng(int(args.seed))

    # ---- eSurge-parity verify-mode prefill for context features ----
    mesh = model.mesh
    text_cfg = model.config.get_text_config()
    vocab_size = int(text_cfg.vocab_size)
    # We run verify-mode prefill to build ctx features, then generate
    # (block_size-1) greedy labels. Ensure the SequenceBuffer can hold the
    # anchor token + the generated label tokens.
    max_model_len = int(ctx_len_full + max(1, int(block_size) - 1))
    empty_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())
    seqbuf = SequenceBuffer(
        max_num_reqs=1,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        vocab_size=vocab_size,
        page_sizes=[int(args.page_size)],
        sharding=empty_sharding,
    )

    max_pages_per_req = int(
        getattr(
            model.create_ragged_page_cache_config(
                hbm_utilization=float(args.hbm_utilization),
                page_size=int(args.page_size),
                max_length=max_model_len,
            ),
            "max_num_pages_per_req",
        )
    )
    page_ids = (list(range(max_pages_per_req)),)

    rid = "cache-req"
    sp = SamplingParams(max_tokens=1, temperature=0.0, top_k=1, top_p=1.0)
    req_state = CachedRequestState(
        req_id=rid,
        prompt_token_ids=[0, 0],
        sampling_params=sp,
        generator=jax.random.PRNGKey(0),
        page_ids=page_ids,
        num_computed_tokens=0,
        output_token_ids=[],
    )
    seqbuf.add_request(req_state, req_index=0)

    metadata = model.create_ragged_page_cache_config(
        hbm_utilization=float(args.hbm_utilization),
        page_size=int(args.page_size),
        max_length=max_model_len,
    )
    executor = ExecutionManager(
        model=model.esurge_compatible_model,
        use_aot_forward=True,
        min_input_pad=1,
        max_model_len=max_model_len,
        max_num_reqs=1,
        max_num_tokens=max_model_len,
        metadata=metadata,
        verbose=False,
        verify_target_layer_ids=target_layer_ids,
        verify_add_one_for_pre_layer_capture=True,
    )
    compile_prefill_bucket = int(max(1, min(ctx_len, int(args.prefill_chunk))))
    executor.compile(
        num_tokens_paddings=sorted({1, int(block_size), int(compile_prefill_bucket)}),
        num_reqs_max_model_len=1,
        max_pages_per_req=int(metadata.max_num_pages_per_req),
        max_num_reqs=1,
        metadata=metadata,
        num_reqs_paddings=[1],
    )

    input_ids_buf = jax.device_put(jnp.zeros((int(max_model_len),), dtype=jnp.int32), empty_sharding)
    position_ids_buf = jax.device_put(jnp.zeros((int(max_model_len),), dtype=jnp.int32), empty_sharding)
    scheduled_full_cpu = np.zeros((1,), dtype=np.int32)
    active_mask_full_cpu = np.asarray([True], dtype=bool)
    page_table_cpu = seqbuf.page_table[0].get_cpu_tensor()
    page_table_version = getattr(seqbuf.page_table[0], "cpu_version", None)

    def _reset_kv_pages():
        nonlocal executor

        def _zero(x):
            if isinstance(x, jax.Array):
                return jnp.zeros_like(x)
            return x

        with mesh:
            executor.kv_pages = jax.tree_util.tree_map(_zero, executor.kv_pages)
            executor.kv_pages = jax.block_until_ready(executor.kv_pages)

    if rollout_steps > 1 and int(args.batch_size) != 1:
        raise ValueError("--batch-size must be 1 when --rollout-steps > 1 (simplify correctness-first rollout).")

    prefill_chunk = int(max(1, int(args.prefill_chunk)))

    t0 = time.time()
    out_pos = 0
    for i in range(int(n)):
        # Initial state (from packed dataset tokens).
        ctx_ids = input_ids[i, :ctx_len].astype(np.int32)  # [ctx_len] (prefix excluding anchor)
        anchor_id = int(input_ids[i, ctx_len])  # "current token" (anchor)
        if str(args.position_offset_mode) == "cycle_samples":
            ctx_pos_start = int(pos_offsets[int(out_pos) % len(pos_offsets)])
        else:
            ctx_pos_start = int(pos_offsets[int(i) % len(pos_offsets)])

        block_t0 = time.time()
        # Maintain a rolling context-feature window that evolves exactly like
        # runtime DFlash: append ctx features for committed tokens coming from
        # the block-verify forward, and drop from the left to keep ctx_len.
        #
        # IMPORTANT: these features correspond to the *candidate* tokens (cand)
        # used during verification, not to the target predictions. If rollout
        # advances token IDs using a different sequence, ctx_token_ids and
        # context_features will silently mismatch.
        ctx_feat_win_bf16: np.ndarray | None = None  # [ctx_len, K*hidden]

        for step_idx in range(int(rollout_steps)):
            # Generate one training sample for this (ctx_ids, anchor_id) state.
            tgt = np.empty((int(block_size - 1),), dtype=np.int32)
            pos_off = int(ctx_pos_start)
            pos_off_cpu = np.asarray([np.int32(pos_off)], dtype=np.int32)
            if max_pos_emb > 0:
                max_pos_index = int(max_pos_emb - 1)
                worst = int(pos_off + int(ctx_len) + int(block_size) - 1)
                if worst > max_pos_index:
                    raise ValueError(
                        f"position overflow during rollout: ctx_pos_start={pos_off} "
                        f"ctx_len={int(ctx_len)} block_size={int(block_size)} "
                        f"worst_pos={worst} max_pos={max_pos_index}"
                    )

            _reset_kv_pages()

            prompt_len = int(ctx_len + 1)
            seqbuf.token_ids[0, :prompt_len] = np.concatenate([ctx_ids, np.asarray([anchor_id], dtype=np.int32)], axis=0)
            seqbuf.num_tokens[0] = int(prompt_len)
            seqbuf.num_tokens_no_spec[0] = int(prompt_len)
            seqbuf.num_computed_tokens[0] = 0
            seqbuf.temperature[0] = 0.0
            seqbuf.top_k[0] = 1
            seqbuf.top_p[0] = 1.0
            seqbuf.min_p[0] = 0.0

            total = int(prompt_len - 1)
            done = 0
            parts = []
            prefill_bucket = int(min(prefill_chunk, total))
            while done < total:
                step = int(min(prefill_chunk, total - done))
                seqbuf.num_computed_tokens[0] = int(done)
                scheduled_full_cpu[0] = int(step)
                ctx_part, _greedy_unused, input_ids_buf, position_ids_buf, _m = executor.execute_verify(
                    num_tokens=int(prefill_bucket),
                    scheduled_full_cpu=scheduled_full_cpu,
                    active_mask_full_cpu=active_mask_full_cpu,
                    input_ids_buf=input_ids_buf,
                    position_ids_buf=position_ids_buf,
                    padded_num_reqs=1,
                    token_ids_cpu=seqbuf.token_ids,
                    num_computed_tokens_cpu=seqbuf.num_computed_tokens,
                    position_offset_cpu=pos_off_cpu,
                    temperature_cpu=seqbuf.temperature,
                    top_p_cpu=seqbuf.top_p,
                    top_k_cpu=seqbuf.top_k,
                    min_p_cpu=seqbuf.min_p,
                    page_table_cpu=page_table_cpu,
                    page_table_version=page_table_version,
                )
                parts.append(jnp.asarray(ctx_part)[:step, :])
                done += step

            ctx_full = jnp.concatenate(parts, axis=0) if parts else jnp.zeros((0, k * hidden), dtype=jnp.bfloat16)
            if int(ctx_full.shape[0]) != int(ctx_len):
                raise RuntimeError(f"ctx_full length mismatch: got {int(ctx_full.shape[0])}, expected {int(ctx_len)}")
            # Initialize rolling window from the very first prefill; subsequent
            # steps update this window using verify-mode ctx features for
            # committed tokens (runtime-parity).
            if ctx_feat_win_bf16 is None:
                ctx_feat_win_bf16 = np.asarray(
                    jax.device_get(ctx_full.reshape((int(ctx_len), int(k * hidden)))),
                    dtype=ml_dtypes.bfloat16,
                ).copy()

            # Greedy decode candidates (block_size-1) using 1-token verify steps.
            seqbuf.num_computed_tokens[0] = int(ctx_len)
            seqbuf.num_tokens[0] = int(prompt_len)
            seqbuf.num_tokens_no_spec[0] = int(prompt_len)
            for j in range(int(block_size - 1)):
                base_len = int(seqbuf.num_computed_tokens[0])
                scheduled_full_cpu[0] = 1
                _ctx_unused, greedy_ids, input_ids_buf, position_ids_buf, _m = executor.execute_verify(
                    # Use a fixed token bucket (block_size) for 1-token steps to
                    # keep greedy candidate generation numerically consistent
                    # with the block-verify kernel on TPU.
                    num_tokens=int(block_size),
                    scheduled_full_cpu=scheduled_full_cpu,
                    active_mask_full_cpu=active_mask_full_cpu,
                    input_ids_buf=input_ids_buf,
                    position_ids_buf=position_ids_buf,
                    padded_num_reqs=1,
                    token_ids_cpu=seqbuf.token_ids,
                    num_computed_tokens_cpu=seqbuf.num_computed_tokens,
                    position_offset_cpu=pos_off_cpu,
                    temperature_cpu=seqbuf.temperature,
                    top_p_cpu=seqbuf.top_p,
                    top_k_cpu=seqbuf.top_k,
                    min_p_cpu=seqbuf.min_p,
                    page_table_cpu=page_table_cpu,
                    page_table_version=page_table_version,
                )
                next_id = int(np.asarray(greedy_ids)[0])
                tgt[j] = np.int32(next_id)
                seqbuf.token_ids[0, base_len + 1] = np.int32(next_id)
                seqbuf.num_computed_tokens[0] = int(base_len + 1)
                seqbuf.num_tokens[0] = int(base_len + 2)
                seqbuf.num_tokens_no_spec[0] = int(base_len + 2)

            # Compute block-verify predictions (targets) on [anchor + greedy tokens].
            cand = np.empty((int(block_size),), dtype=np.int32)
            cand[0] = np.int32(anchor_id)
            cand[1:] = tgt

            _reset_kv_pages()
            seqbuf.token_ids[0, :prompt_len] = np.concatenate([ctx_ids, np.asarray([anchor_id], dtype=np.int32)], axis=0)
            seqbuf.num_tokens[0] = int(prompt_len)
            seqbuf.num_tokens_no_spec[0] = int(prompt_len)
            seqbuf.num_computed_tokens[0] = 0
            seqbuf.temperature[0] = 0.0
            seqbuf.top_k[0] = 1
            seqbuf.top_p[0] = 1.0
            seqbuf.min_p[0] = 0.0

            done = 0
            while done < total:
                step = int(min(prefill_chunk, total - done))
                seqbuf.num_computed_tokens[0] = int(done)
                scheduled_full_cpu[0] = int(step)
                _ctx_unused, _greedy_unused, input_ids_buf, position_ids_buf, _m = executor.execute_verify(
                    num_tokens=int(prefill_bucket),
                    scheduled_full_cpu=scheduled_full_cpu,
                    active_mask_full_cpu=active_mask_full_cpu,
                    input_ids_buf=input_ids_buf,
                    position_ids_buf=position_ids_buf,
                    padded_num_reqs=1,
                    token_ids_cpu=seqbuf.token_ids,
                    num_computed_tokens_cpu=seqbuf.num_computed_tokens,
                    position_offset_cpu=pos_off_cpu,
                    temperature_cpu=seqbuf.temperature,
                    top_p_cpu=seqbuf.top_p,
                    top_k_cpu=seqbuf.top_k,
                    min_p_cpu=seqbuf.min_p,
                    page_table_cpu=page_table_cpu,
                    page_table_version=page_table_version,
                )
                done += step

            seqbuf.num_computed_tokens[0] = int(ctx_len)
            seqbuf.token_ids[0, int(ctx_len) : int(ctx_len + int(block_size))] = cand
            seqbuf.num_tokens[0] = int(ctx_len + int(block_size))
            seqbuf.num_tokens_no_spec[0] = int(ctx_len + int(block_size))

            scheduled_full_cpu[0] = int(block_size)
            ctx_v, greedy_block, input_ids_buf, position_ids_buf, _m = executor.execute_verify(
                num_tokens=int(block_size),
                scheduled_full_cpu=scheduled_full_cpu,
                active_mask_full_cpu=active_mask_full_cpu,
                input_ids_buf=input_ids_buf,
                position_ids_buf=position_ids_buf,
                padded_num_reqs=1,
                token_ids_cpu=seqbuf.token_ids,
                num_computed_tokens_cpu=seqbuf.num_computed_tokens,
                position_offset_cpu=pos_off_cpu,
                temperature_cpu=seqbuf.temperature,
                top_p_cpu=seqbuf.top_p,
                top_k_cpu=seqbuf.top_k,
                min_p_cpu=seqbuf.min_p,
                page_table_cpu=page_table_cpu,
                page_table_version=page_table_version,
            )
            greedy_block = np.asarray(greedy_block, dtype=np.int32)
            if greedy_block.shape[0] < int(block_size):
                raise RuntimeError(
                    f"verify greedy_block length mismatch: got {greedy_block.shape}, expected >=({int(block_size)},)"
                )
            tgt_verify = greedy_block[: int(block_size) - 1].astype(np.int32)
            # NOTE: actual bonus used by DFlash is greedy_block[accept_len]; we
            # compute it after sampling accept_len below.

            with mesh:
                emb = model.get_embedding()(jnp.asarray([[anchor_id]], dtype=jnp.int32))[:, 0, :]
                emb = jax.block_until_ready(emb)

            assert ctx_feat_win_bf16 is not None
            emb_bf16 = np.asarray(jax.device_get(emb[0]), dtype=ml_dtypes.bfloat16)
            ctx_feats_u16[out_pos] = np.asarray(ctx_feat_win_bf16, dtype=ml_dtypes.bfloat16).view(np.uint16)
            anchor_emb_u16[out_pos] = emb_bf16.view(np.uint16)
            target_ids[out_pos] = tgt_verify
            anchor_ids[out_pos] = np.int32(anchor_id)
            ctx_token_ids[out_pos] = np.asarray(ctx_ids, dtype=np.int32)
            ctx_pos_start_i32[out_pos] = np.int32(ctx_pos_start)
            anchor_pos_i32[out_pos] = np.int32(int(ctx_pos_start) + int(ctx_len))
            out_pos += 1

            # Roll forward to synthesize the next prompt prefix.
            #
            # Real DFlash decoding commits:
            #   keep = 1 + accept_len
            # tokens from the verify block, and advances the "current token" to the
            # target greedy token at index accept_len (bonus token).
            #
            # We approximate this by sampling an accept_len and advancing the
            # teacher-only rollout accordingly. This keeps the rollout on teacher
            # distribution while covering partial-commit regimes.
            mode = str(args.rollout_accept_len_mode).strip().lower()
            if mode == "full":
                accept_len = int(block_size) - 1
            elif mode == "uniform":
                accept_len = int(rng.integers(0, int(block_size), dtype=np.int64))
            elif mode == "geometric":
                p = float(args.rollout_accept_len_p)
                if not (0.0 < p <= 1.0):
                    raise ValueError(f"rollout_accept_len_p must be in (0,1], got {p}")
                accept_len = int(min(int(rng.geometric(p)) - 1, int(block_size) - 1))
            else:
                raise ValueError(f"Unknown rollout_accept_len_mode={mode!r}")

            keep = int(1 + accept_len)
            bonus = int(greedy_block[accept_len])

            # Update rolling ctx features (runtime-parity): append ctx features
            # for committed tokens [anchor] + accepted draft tokens.
            #
            # `ctx_v` is the verify-mode captured ctx features for the *candidate*
            # tokens in `cand`. This is exactly what runtime uses to condition
            # the next draft step.
            ctx_block_bf16 = np.asarray(
                jax.device_get(jnp.asarray(ctx_v)[: int(block_size), :]),
                dtype=ml_dtypes.bfloat16,
            )
            if keep > 0:
                if keep >= int(ctx_len):
                    ctx_feat_win_bf16 = ctx_block_bf16[:keep, :][-int(ctx_len) :, :].copy()
                else:
                    assert ctx_feat_win_bf16 is not None
                    ctx_feat_win_bf16[:-keep, :] = ctx_feat_win_bf16[keep:, :]
                    ctx_feat_win_bf16[-keep:, :] = ctx_block_bf16[:keep, :]

            # Advance token history. For DFlash parity, commit candidate tokens.
            if str(args.rollout_state_evolution).strip().lower() == "teacher_commit":
                committed = tgt_verify[:accept_len].astype(np.int32, copy=False)
            else:
                committed = cand[1 : 1 + int(accept_len)].astype(np.int32, copy=False)

            if keep > 0:
                keep_tokens = np.concatenate(
                    [np.asarray([anchor_id], dtype=np.int32), np.asarray(committed, dtype=np.int32)], axis=0
                )
                if keep >= int(ctx_len):
                    ctx_ids = keep_tokens[-int(ctx_len) :].astype(np.int32, copy=False)
                else:
                    ctx_ids[:-keep] = ctx_ids[keep:]
                    ctx_ids[-keep:] = keep_tokens

            anchor_id = int(bonus)
            ctx_pos_start = int(ctx_pos_start + keep)

            # Heartbeat for long rollouts: print a stable progress line without spamming.
            if (step_idx + 1) % 8 == 0 or (step_idx + 1) == int(rollout_steps):
                elapsed = max(1e-6, float(time.time() - block_t0))
                done_steps = int(step_idx + 1)
                total_steps = int(rollout_steps)
                rate = float(done_steps) / float(elapsed)
                remaining_s = float(total_steps - done_steps) / max(1e-9, rate)
                print(
                    f"[cache] block={i + 1}/{n} rollout={done_steps}/{total_steps} "
                    f"(out={out_pos}/{n_out}) rate={rate:.2f} steps/s eta={remaining_s:.0f}s",
                    flush=True,
                )

        if (i + 1) % 1 == 0:
            print(f"[cache] {i + 1}/{n} (out={out_pos}/{n_out})", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "model_snapshot_dir": str(snapshot),
        "platform": str(args.platform),
        "ctx_len": int(ctx_len),
        "ctx_len_full": int(ctx_len_full),
        "block_size": int(block_size),
        "num_blocks": int(n),
        "rollout_steps": int(rollout_steps),
        "num_samples": int(n_out),
        "batch_size": int(args.batch_size),
        "prefill_chunk": int(args.prefill_chunk),
        "page_size": int(args.page_size),
        "hbm_utilization": float(args.hbm_utilization),
        "sharding_axis_dims": list(sharding_axis_dims),
        "target_layer_ids": target_layer_ids,
        "add_one_for_pre_layer_capture": True,
        "target_ids_mode": "block_verify_target_predict_shifted",
        "hidden_size": int(hidden),
        "num_context_features": int(k),
        "dtype": "bf16_u16",
        "seed": int(args.seed),
        "position_offsets": pos_offsets,
        "position_offset_mode": str(args.position_offset_mode),
        "positions_contiguous": True,
        "ctx_pos_start_file": "ctx_pos_start_i32.npy",
        "anchor_pos_file": "anchor_pos_i32.npy",
        "calib_repo_id": str(args.calib_repo_id),
        "calib_data_files": data_files,
        "max_rows_per_pack": int(args.max_rows_per_pack),
        "wall_s": float(time.time() - t0),
    }
    # NOTE: NumPy does not have a native bfloat16 dtype. If we save ml_dtypes.bfloat16
    # directly, it gets serialized as a void dtype (e.g. |V2), which is annoying to
    # load robustly. Save bf16 tensors as raw uint16 bitpatterns instead.
    # ctx_feats_u16 / anchor_emb_u16 are already stored as uint16 bitpatterns.

    # Write mmap-friendly directory if requested.
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        if not using_memmap:
            np.save(out_dir / "context_features_u16.npy", ctx_feats_u16)
            np.save(out_dir / "anchor_embedding_u16.npy", anchor_emb_u16)
            np.save(out_dir / "anchor_ids.npy", anchor_ids)
            np.save(out_dir / "ctx_pos_start_i32.npy", ctx_pos_start_i32)
            np.save(out_dir / "anchor_pos_i32.npy", anchor_pos_i32)
            np.save(out_dir / "target_ids.npy", target_ids)
            np.save(out_dir / "ctx_token_ids.npy", ctx_token_ids)
        else:
            # Ensure memmaps are flushed before we return control to the caller.
            try:
                ctx_feats_u16.flush()
                anchor_emb_u16.flush()
                target_ids.flush()
                anchor_ids.flush()
                ctx_token_ids.flush()
                ctx_pos_start_i32.flush()
                anchor_pos_i32.flush()
            except Exception:
                pass
        print(f"[done] wrote cache_dir={out_dir}", flush=True)

    write_npz = str(args.write_npz).strip().lower() in ("1", "true", "yes", "y", "on")
    if not write_npz:
        return

    # Write legacy .npz output (useful for quick smoke tests).
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        context_features_u16=ctx_feats_u16,
        anchor_embedding_u16=anchor_emb_u16,
        anchor_ids=anchor_ids,
        ctx_pos_start_i32=ctx_pos_start_i32,
        anchor_pos_i32=anchor_pos_i32,
        target_ids=target_ids,
        ctx_token_ids=ctx_token_ids,
        meta=json.dumps(meta),
    )
    print(f"[done] wrote npz={out_path}", flush=True)


if __name__ == "__main__":
    main()
