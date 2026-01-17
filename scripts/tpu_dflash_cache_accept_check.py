#!/usr/bin/env python3
"""Sanity-check DFlash draft acceptance on *training-distribution* cache samples (TPU).

Why this exists:
- End-to-end decode benchmarks can show accept_rate=0 simply because the prompt
  distribution differs from the draft training cache (especially early).
- This script verifies the *mechanics*:
  cache sample -> prefill teacher KV -> draft propose -> verify -> accept_len.

If this script shows accept_len > 0 on cache samples, training+plumbing is
correct; then the remaining work is (1) data mixture alignment and (2) fast-path
KV materialization/rollback for speed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def _bf16_from_u16(x_u16: np.ndarray):
    import ml_dtypes

    return x_u16.view(ml_dtypes.bfloat16)


def _resolve_draft_run_dir(draft_run_dir: Path) -> Path:
    # Accept either:
    # - a specific checkpoint directory (contains config.json), or
    # - a checkpoint root directory containing run-*/ subdirectories.
    if (draft_run_dir / "config.json").is_file():
        return draft_run_dir

    run_dirs = [p for p in draft_run_dir.glob("run-*") if p.is_dir() and (p / "config.json").is_file()]
    if not run_dirs:
        raise FileNotFoundError(
            f"Could not find config.json in {draft_run_dir} or any run-*/ subdirectory. "
            "Pass --draft-run-dir pointing at a specific run-### directory."
        )

    def _run_step(p: Path) -> int:
        try:
            return int(p.name.split("-", 1)[1])
        except Exception:
            return -1

    run_dirs.sort(key=_run_step, reverse=True)
    return run_dirs[0]


def _lm_head_logits(*, hidden, lm_head):
    import jax
    import jax.numpy as jnp

    kernel = jax.lax.stop_gradient(lm_head.kernel.value)
    bias = jax.lax.stop_gradient(lm_head.bias.value) if getattr(lm_head, "bias", None) is not None else None
    if int(kernel.shape[0]) == int(hidden.shape[-1]):
        logits = jnp.einsum("bsh,hv->bsv", hidden, kernel, precision=jax.lax.Precision.HIGHEST)
    else:
        logits = jnp.einsum("bsh,vh->bsv", hidden, kernel, precision=jax.lax.Precision.HIGHEST)
    if bias is not None:
        logits = logits + bias[None, None, :]
    return logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--teacher-snapshot-dir", required=True)
    ap.add_argument("--teacher-easydel-dir", default="", help="Optional EasyDeL-native teacher dir (faster).")
    ap.add_argument("--draft-run-dir", required=True)
    ap.add_argument("--sample-idx", type=int, default=0)
    ap.add_argument(
        "--use-cache-labels-as-draft",
        action="store_true",
        help="Debug: set draft_tokens := cache_labels to test label/verify parity without draft inference.",
    )
    ap.add_argument(
        "--use-ctx-kv-forward",
        action="store_true",
        help="Use the ctx-KV materialization path (materialize_draft_ctx_kv + draft_forward_with_ctx_kv) instead of draft(...)",
    )
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--page-size", type=int, default=32)
    ap.add_argument("--hbm-utilization", type=float, default=0.20)
    ap.add_argument("--prefill-chunk", type=int, default=256)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    local_easydel = repo_root / "external" / "EasyDeL"
    if local_easydel.exists():
        sys.path.insert(0, str(local_easydel))
    os.environ.setdefault("EASYDEL_SKIP_VERSION_CHECK", "1")
    os.environ.setdefault("TMPDIR", "/dev/shm/tmp")
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    import jax
    import jax.numpy as jnp
    from easydel import AutoEasyDeLModelForCausalLM
    from easydel.inference.esurge.runners.sequence_buffer import SequenceBuffer
    from easydel.inference.esurge.runners.execution_manager import ExecutionManager
    from easydel.inference.esurge.runners.states import CachedRequestState
    from easydel.inference.sampling_params import SamplingParams
    from easydel.layers.rotary_embedding import get_rope
    from easydel.inference.speculative import DFlashDraftModelConfig, load_dflash_draft_from_run_dir
    from easydel.inference.speculative.dflash import dflash_accept_len_and_bonus
    from easydel.inference.speculative.dflash_kv_cache import draft_forward_with_ctx_kv, materialize_draft_ctx_kv

    cache_dir = Path(args.cache_dir).resolve()
    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    ctx_len = int(meta["ctx_len"])
    block_size = int(meta["block_size"])

    ctx_feats_u16 = np.load(cache_dir / "context_features_u16.npy", mmap_mode="r")
    ctx_tokens = np.load(cache_dir / "ctx_token_ids.npy", mmap_mode="r")
    anchor_ids = np.load(cache_dir / "anchor_ids.npy", mmap_mode="r")
    target_ids = np.load(cache_dir / "target_ids.npy", mmap_mode="r")
    pos_path = cache_dir / "ctx_pos_start_i32.npy"
    ctx_pos_start_i32 = np.load(pos_path, mmap_mode="r") if pos_path.exists() else None

    i = int(args.sample_idx)
    ctx_feat = _bf16_from_u16(np.asarray(ctx_feats_u16[i])).astype(np.float16)  # bf16->f16 host; moved to device later
    ctx_tok = np.asarray(ctx_tokens[i]).astype(np.int32)
    anchor_id = int(np.asarray(anchor_ids[i]))
    labels = np.asarray(target_ids[i]).astype(np.int32)
    pos_off = int(np.asarray(ctx_pos_start_i32[i])) if ctx_pos_start_i32 is not None else 0
    pos_off_cpu = np.asarray([np.int32(pos_off)], dtype=np.int32)

    teacher_snapshot = Path(args.teacher_snapshot_dir).resolve()
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
    # TPU correctness: keep verify-mode on ragged_page_attention_v2 unless
    # explicitly overridden, to match how we build the teacher cache.
    if os.environ.get("DFLASH_FORCE_RAGGED_V2", "1").lower() in ("1", "true", "yes", "y", "on"):
        teacher = teacher.merge_module(
            teacher.new_graphdef(attn_mechanism="ragged_page_attention_v2"),
            teacher.graphstate,
            teacher.graphother,
        )

    # Draft cfg must match the run.
    run_dir = _resolve_draft_run_dir(Path(args.draft_run_dir).resolve())
    draft_cfg = DFlashDraftModelConfig(**json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    draft = load_dflash_draft_from_run_dir(run_dir=run_dir, cfg=draft_cfg, mesh=teacher.mesh)

    # eSurge setup.
    mesh = teacher.mesh
    empty_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())
    max_model_len = int(args.max_model_len)
    vocab_size = int(teacher.config.get_text_config().vocab_size)
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
            teacher.create_ragged_page_cache_config(
                hbm_utilization=float(args.hbm_utilization),
                page_size=int(args.page_size),
                max_length=max_model_len,
            ),
            "max_num_pages_per_req",
        )
    )
    page_ids = (list(range(max_pages_per_req)),)
    sp = SamplingParams(max_tokens=1, temperature=0.0, top_k=1, top_p=1.0)
    req = CachedRequestState(
        req_id="cache-req",
        prompt_token_ids=(ctx_tok.tolist() + [int(anchor_id)]),
        sampling_params=sp,
        generator=jax.random.PRNGKey(0),
        page_ids=page_ids,
        num_computed_tokens=0,
        output_token_ids=[],
    )
    seqbuf.add_request(req, req_index=0)
    metadata = teacher.create_ragged_page_cache_config(
        hbm_utilization=float(args.hbm_utilization),
        page_size=int(args.page_size),
        max_length=max_model_len,
    )
    executor = ExecutionManager(
        model=teacher.esurge_compatible_model,
        use_aot_forward=True,
        min_input_pad=1,
        max_model_len=max_model_len,
        max_num_reqs=1,
        max_num_tokens=max_model_len,
        metadata=metadata,
        verbose=False,
        verify_target_layer_ids=[int(x) for x in meta["target_layer_ids"]],
        verify_add_one_for_pre_layer_capture=bool(meta.get("add_one_for_pre_layer_capture", True)),
    )
    prefill_bucket = int(min(ctx_len, int(args.prefill_chunk)))
    executor.compile(
        num_tokens_paddings=sorted({1, int(block_size), int(max(1, prefill_bucket))}),
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

    # Prefill ctx tokens (exclude anchor).
    seqbuf.temperature[0] = 0.0
    seqbuf.top_k[0] = 1
    seqbuf.top_p[0] = 1.0
    seqbuf.min_p[0] = 0.0
    seqbuf.num_tokens[0] = int(ctx_len + 1)
    seqbuf.num_tokens_no_spec[0] = int(ctx_len + 1)
    done = 0
    while done < int(ctx_len):
        step = int(min(prefill_bucket, int(ctx_len) - done))
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

    # Pending anchor at position ctx_len.
    seqbuf.num_computed_tokens[0] = int(ctx_len)

    if bool(args.use_cache_labels_as_draft):
        if int(labels.shape[0]) != int(block_size - 1):
            raise ValueError(f"cache_labels length mismatch: {labels.shape} vs block_size={int(block_size)}")
        draft_tokens = labels
    else:
        with mesh:
            ctx_feat_dev = jnp.asarray(ctx_feat, dtype=jnp.bfloat16).reshape((1, int(ctx_len), -1))
            anchor_emb = teacher.get_embedding()(jnp.asarray([[anchor_id]], dtype=jnp.int32))[:, 0, :]
            if bool(args.use_ctx_kv_forward):
                ctx_hidden = draft.project_context_features(ctx_feat_dev)
                ctx_kv = materialize_draft_ctx_kv(draft=draft, rope=rope, ctx_hidden=ctx_hidden, max_len=int(ctx_len + block_size + 8))
                d_hidden = draft_forward_with_ctx_kv(
                    draft=draft,
                    rope=rope,
                    cache=ctx_kv,
                    anchor_embedding=anchor_emb.astype(jnp.bfloat16),
                    mask_embedding=draft.mask_embedding.value.astype(jnp.bfloat16),
                    block_size=int(block_size),
                )
            else:
                d_hidden = draft(context_features=ctx_feat_dev, anchor_embedding=anchor_emb.astype(jnp.bfloat16), rope=rope)
            hs_d = d_hidden[:, 1:, :]
            d_logits = _lm_head_logits(hidden=hs_d.astype(jnp.bfloat16), lm_head=teacher.get_lm_head())
            draft_tokens = jnp.argmax(d_logits, axis=-1).astype(jnp.int32)[0]  # [B-1]

    cand = np.concatenate([np.asarray([anchor_id], dtype=np.int32), np.asarray(draft_tokens, dtype=np.int32)], axis=0)
    seqbuf.token_ids[0, int(ctx_len) : int(ctx_len) + int(block_size)] = cand
    # IMPORTANT: verify-mode uses `seq_lens` to build cache metadata and to
    # decide how many tokens exist in the request. We must extend num_tokens to
    # include the full verify block, matching the cache builder + runtime decode.
    seqbuf.num_tokens[0] = int(ctx_len + int(block_size))
    seqbuf.num_tokens_no_spec[0] = int(ctx_len + int(block_size))
    scheduled_full_cpu[0] = int(block_size)
    _ctx_part, greedy_ids, input_ids_buf, position_ids_buf, _m = executor.execute_verify(
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
    greedy_ids = np.asarray(greedy_ids, dtype=np.int32)
    accept_len, bonus = dflash_accept_len_and_bonus(
        candidates=jnp.asarray(cand[None, :], dtype=jnp.int32),
        target_predict=jnp.asarray(greedy_ids[None, :], dtype=jnp.int32),
    )
    n_acc = int(np.asarray(accept_len)[0])
    bonus_id = int(np.asarray(bonus)[0])

    print(
        json.dumps(
            {
                "sample_idx": i,
                "ctx_len": int(ctx_len),
                "block_size": int(block_size),
                "accept_len": int(n_acc),
                "bonus_id": int(bonus_id),
                "anchor_id": int(anchor_id),
                "draft_tokens": [int(x) for x in np.asarray(draft_tokens).tolist()],
                "teacher_greedy_ids": [int(x) for x in greedy_ids.tolist()],
                "cache_labels": [int(x) for x in labels.tolist()],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
