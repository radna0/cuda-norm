#!/usr/bin/env python3
"""Decode-first DFlash benchmark on TPU using cached context features (mechanics/perf probe).

This is NOT a deployment benchmark (real serving must extract context features
from the target model). This benchmark isolates the DFlash inner loop:
draft propose + target verify + commit.

It answers:
- Given high accept_len on cache distribution, is our DFlash loop faster than
  token-by-token baseline verify on TPU?
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def _resolve_draft_run_dir(draft_run_dir: Path) -> Path:
    """Accept either a specific run dir (contains config.json) or a checkpoint root (run-*)."""
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


def _bf16_from_u16(x_u16: np.ndarray):
    import ml_dtypes

    return x_u16.view(ml_dtypes.bfloat16)


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
    ap.add_argument("--teacher-easydel-dir", default="")
    ap.add_argument("--draft-run-dir", required=True)
    ap.add_argument("--sample-idx", type=int, default=0)
    ap.add_argument("--blocks", type=int, default=32, help="How many verify blocks to run in the bench loop.")
    ap.add_argument("--warmup-blocks", type=int, default=2, help="Warmup blocks (excluded from timing) to amortize JIT.")
    ap.add_argument("--debug", action="store_true", help="Print per-block accept_len and token IDs (slow).")
    ap.add_argument("--prefill-chunk", type=int, default=256)
    ap.add_argument("--page-size", type=int, default=128)
    ap.add_argument("--hbm-utilization", type=float, default=0.20)
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    local_easydel = repo_root / "external" / "EasyDeL"
    if local_easydel.exists():
        sys.path.insert(0, str(local_easydel))

    os.environ.setdefault("EASYDEL_SKIP_VERSION_CHECK", "1")
    os.environ.setdefault("DFLASH_FORCE_RAGGED_V2", "1")
    os.environ.setdefault("EASYDEL_VERIFY_DISALLOW_ON_DEMAND_COMPILE", "1")

    import jax
    import jax.numpy as jnp
    from easydel import AutoEasyDeLModelForCausalLM
    from easydel.inference.esurge.runners.execution_manager import ExecutionManager
    from easydel.inference.esurge.runners.sequence_buffer import SequenceBuffer
    from easydel.inference.esurge.runners.states import CachedRequestState
    from easydel.inference.sampling_params import SamplingParams
    from easydel.layers.rotary_embedding import get_rope
    from easydel.inference.speculative import DFlashDraftModelConfig, load_dflash_draft_from_run_dir
    from easydel.inference.speculative.dflash import dflash_accept_len_and_bonus
    from easydel.inference.speculative.dflash_kv_cache import (
        append_draft_ctx_kv,
        draft_forward_with_ctx_kv,
        materialize_draft_ctx_kv,
    )

    cache_dir = Path(args.cache_dir).resolve()
    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    ctx_len = int(meta["ctx_len"])
    block_size = int(meta["block_size"])
    k = int(meta["num_context_features"])
    hidden = int(meta["hidden_size"])
    k_hidden = k * hidden

    i = int(args.sample_idx)
    ctx_feats_u16 = np.load(cache_dir / "context_features_u16.npy", mmap_mode="r")
    ctx_token_ids = np.load(cache_dir / "ctx_token_ids.npy", mmap_mode="r")
    anchor_ids = np.load(cache_dir / "anchor_ids.npy", mmap_mode="r")

    ctx_feat = _bf16_from_u16(np.asarray(ctx_feats_u16[i])).reshape((1, ctx_len, k_hidden))
    ctx_ids = np.asarray(ctx_token_ids[i]).astype(np.int32)
    anchor_id = int(np.asarray(anchor_ids[i]))

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

    run_dir = _resolve_draft_run_dir(Path(args.draft_run_dir).resolve())
    draft_cfg = DFlashDraftModelConfig(**json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    draft = load_dflash_draft_from_run_dir(run_dir=run_dir, cfg=draft_cfg, mesh=teacher.mesh)

    # eSurge setup for verify.
    mesh = teacher.mesh
    empty_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())
    max_model_len = int(args.max_model_len)

    text_cfg = teacher.config.get_text_config()
    vocab_size = int(text_cfg.vocab_size)
    seqbuf = SequenceBuffer(
        max_num_reqs=1,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        vocab_size=vocab_size,
        page_sizes=[int(args.page_size)],
        sharding=empty_sharding,
    )
    metadata = teacher.create_ragged_page_cache_config(
        hbm_utilization=float(args.hbm_utilization),
        page_size=int(args.page_size),
        max_length=max_model_len,
    )
    max_pages_per_req = int(getattr(metadata, "max_num_pages_per_req"))
    page_ids = (list(range(max_pages_per_req)),)
    sp = SamplingParams(max_tokens=1, temperature=0.0, top_k=1, top_p=1.0)
    req_state = CachedRequestState(
        req_id="bench",
        prompt_token_ids=[0, 0],
        sampling_params=sp,
        generator=jax.random.PRNGKey(0),
        page_ids=page_ids,
        num_computed_tokens=0,
        output_token_ids=[],
    )
    seqbuf.add_request(req_state, req_index=0)

    target_layer_ids = [int(x) for x in draft_cfg.target_layer_ids]  # type: ignore[attr-defined]
    executor = ExecutionManager(
        model=teacher.esurge_compatible_model,
        use_aot_forward=True,
        min_input_pad=1,
        max_model_len=max_model_len,
        max_num_reqs=1,
        max_num_tokens=max_model_len,
        metadata=metadata,
        verbose=False,
        verify_target_layer_ids=target_layer_ids,
        verify_add_one_for_pre_layer_capture=bool(getattr(draft_cfg, "add_one_for_pre_layer_capture", True)),
    )
    prefill_bucket = int(min(int(args.prefill_chunk), int(ctx_len)))
    executor.compile(
        num_tokens_paddings=sorted({1, int(block_size), int(prefill_bucket)}),
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

    # Populate prompt tokens: ctx + anchor (pending).
    prompt_len = int(ctx_len + 1)
    seqbuf.token_ids[0, :ctx_len] = ctx_ids
    seqbuf.token_ids[0, ctx_len] = np.int32(anchor_id)
    seqbuf.num_tokens[0] = int(prompt_len)
    seqbuf.num_tokens_no_spec[0] = int(prompt_len)
    seqbuf.temperature[0] = 0.0
    seqbuf.top_k[0] = 1
    seqbuf.top_p[0] = 1.0
    seqbuf.min_p[0] = 0.0

    def _prefill_from_scratch() -> None:
        nonlocal input_ids_buf, position_ids_buf
        done = 0
        scheduled_full_cpu[0] = 0
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
                temperature_cpu=seqbuf.temperature,
                top_p_cpu=seqbuf.top_p,
                top_k_cpu=seqbuf.top_k,
                min_p_cpu=seqbuf.min_p,
                page_table_cpu=page_table_cpu,
                page_table_version=page_table_version,
            )
            done += step
        seqbuf.num_computed_tokens[0] = int(ctx_len)

    # Prefill ctx tokens into target KV (no feature capture used here).
    _prefill_from_scratch()

    # Draft ctx KV from cached ctx features.
    with mesh:
        ctx_feat_dev_full = jnp.asarray(ctx_feat, dtype=jnp.bfloat16)
        ctx_hidden = draft.project_context_features(ctx_feat_dev_full)
        # IMPORTANT: draft attention currently operates over the full allocated
        # ctx-KV buffer length. Keep this buffer as small as possible for the
        # benchmark (enough for ctx + a few blocks), otherwise we pay O(max_len)
        # attention cost even when ctx_len is small.
        draft_need = int(ctx_len + int(block_size) * (int(args.blocks) + int(args.warmup_blocks) + 8))
        draft_max_len = int(min(int(max_model_len + int(block_size)), max(int(ctx_len), draft_need)))
        ctx_kv = materialize_draft_ctx_kv(draft=draft, rope=rope, ctx_hidden=ctx_hidden, max_len=int(draft_max_len))
        jax.block_until_ready(ctx_kv.k_full[0])

    # ---- DFlash loop ----
    active_mask_full_cpu[:] = False
    active_mask_full_cpu[0] = True
    scheduled_full_cpu[:] = 0

    def _dflash_one_block() -> tuple[int, int]:
        nonlocal ctx_kv, input_ids_buf, position_ids_buf
        base_len = int(seqbuf.num_computed_tokens[0])
        cur_id = int(seqbuf.token_ids[0, base_len])
        with mesh:
            anchor_emb = teacher.get_embedding()(jnp.asarray([[cur_id]], dtype=jnp.int32))[:, 0, :]
            d_hidden = draft_forward_with_ctx_kv(
                draft=draft,
                rope=rope,
                cache=ctx_kv,
                anchor_embedding=anchor_emb.astype(jnp.bfloat16),
                mask_embedding=draft.mask_embedding.value.astype(jnp.bfloat16),
                block_size=int(block_size),
            )
            hs_d = d_hidden[:, 1:, :]
            d_logits = _lm_head_logits(hidden=hs_d.astype(jnp.bfloat16), lm_head=teacher.get_lm_head())
            draft_tokens = jnp.argmax(d_logits, axis=-1).astype(jnp.int32)[0]

        cand = np.empty((int(block_size),), dtype=np.int32)
        cand[0] = np.int32(cur_id)
        cand[1:] = np.asarray(draft_tokens, dtype=np.int32)
        seqbuf.token_ids[0, base_len : base_len + int(block_size)] = cand
        seqbuf.num_tokens[0] = int(base_len + int(block_size))
        seqbuf.num_tokens_no_spec[0] = int(base_len + int(block_size))

        scheduled_full_cpu[0] = int(block_size)
        ctx_part, greedy_ids, input_ids_buf, position_ids_buf, _m = executor.execute_verify(
            num_tokens=int(block_size),
            scheduled_full_cpu=scheduled_full_cpu,
            active_mask_full_cpu=active_mask_full_cpu,
            input_ids_buf=input_ids_buf,
            position_ids_buf=position_ids_buf,
            padded_num_reqs=1,
            token_ids_cpu=seqbuf.token_ids,
            num_computed_tokens_cpu=seqbuf.num_computed_tokens,
            temperature_cpu=seqbuf.temperature,
            top_p_cpu=seqbuf.top_p,
            top_k_cpu=seqbuf.top_k,
            min_p_cpu=seqbuf.min_p,
            page_table_cpu=page_table_cpu,
            page_table_version=page_table_version,
        )

        greedy_ids = jnp.asarray(greedy_ids)[None, :]
        cand_j = jnp.asarray(cand[None, :], dtype=jnp.int32)
        accept_len, bonus = dflash_accept_len_and_bonus(candidates=cand_j, target_predict=greedy_ids)
        n_acc = int(jnp.asarray(accept_len)[0])
        keep = 1 + n_acc
        if bool(args.debug):
            tgt0 = np.asarray(jax.device_get(greedy_ids[0]))[: int(block_size)]
            print(
                f"[debug] base_len={base_len} cur_id={cur_id} accept_len={n_acc} "
                f"cand1..3={cand[1:4].tolist()} tgt0..2={tgt0[:3].tolist()} bonus={int(jax.device_get(bonus[0]))}",
                flush=True,
            )

        ctx_commit_feat = jnp.asarray(ctx_part)[:keep, :].reshape((1, keep, -1)).astype(jnp.bfloat16)
        if bool(args.debug) and int(base_len) == int(ctx_len):
            # Diagnose whether the problem is in ctx_part->features or in KV append:
            # compute the next-block draft tokens using the *direct* draft(...) path
            # with context_features := [ctx_feat_dev_full + commit_features].
            with mesh:
                ctx2 = jnp.concatenate([ctx_feat_dev_full, ctx_commit_feat], axis=1)
                bonus_id2 = jnp.asarray(bonus[0], dtype=jnp.int32)
                anchor2 = teacher.get_embedding()(bonus_id2.reshape((1, 1)))[:, 0, :]
                d2 = draft(context_features=ctx2, anchor_embedding=anchor2.astype(jnp.bfloat16), rope=rope)
                d2_logits = _lm_head_logits(hidden=d2[:, 1:, :].astype(jnp.bfloat16), lm_head=teacher.get_lm_head())
                d2_tok = jnp.argmax(d2_logits, axis=-1).astype(jnp.int32)[0]
                print(f"[debug] direct-next cand1..3={np.asarray(jax.device_get(d2_tok))[:3].tolist()}", flush=True)
        with mesh:
            new_ctx_hidden = draft.project_context_features(ctx_commit_feat)
            ctx_kv = append_draft_ctx_kv(draft=draft, rope=rope, cache=ctx_kv, new_ctx_hidden=new_ctx_hidden)
            jax.block_until_ready(ctx_kv.k_full[0])

        seqbuf.num_computed_tokens[0] = int(base_len + keep)
        seqbuf.token_ids[0, int(base_len + keep)] = np.int32(jnp.asarray(bonus)[0])
        seqbuf.num_tokens[0] = int(base_len + keep + 1)
        seqbuf.num_tokens_no_spec[0] = int(base_len + keep + 1)
        # Emitted tokens per DFlash step: accepted draft tokens + 1 bonus token.
        # (The anchor is the current token being verified; it is not counted as "new".)
        return int(n_acc + 1), int(n_acc)

    # Warmup excluded from timing (compiles / stabilizes executors).
    for _ in range(int(args.warmup_blocks)):
        _dflash_one_block()

    # Reset back to the original (ctx + pending anchor) state so warmup doesn't
    # contaminate the timed acceptance (append_draft_ctx_kv mutates ctx_kv).
    seqbuf.token_ids[0, :ctx_len] = ctx_ids
    seqbuf.token_ids[0, ctx_len] = np.int32(anchor_id)
    seqbuf.num_tokens[0] = int(prompt_len)
    seqbuf.num_tokens_no_spec[0] = int(prompt_len)
    seqbuf.num_computed_tokens[0] = 0
    seqbuf.temperature[0] = 0.0
    seqbuf.top_k[0] = 1
    seqbuf.top_p[0] = 1.0
    seqbuf.min_p[0] = 0.0
    _prefill_from_scratch()
    with mesh:
        ctx_hidden = draft.project_context_features(jnp.asarray(ctx_feat, dtype=jnp.bfloat16))
        draft_need = int(ctx_len + int(block_size) * (int(args.blocks) + 8))
        draft_max_len = int(min(int(max_model_len + int(block_size)), max(int(ctx_len), draft_need)))
        ctx_kv = materialize_draft_ctx_kv(draft=draft, rope=rope, ctx_hidden=ctx_hidden, max_len=int(draft_max_len))
        jax.block_until_ready(ctx_kv.k_full[0])

    dflash_tokens = 0
    accept_lens: list[int] = []
    t0 = time.time()
    for _ in range(int(args.blocks)):
        emitted, n_acc = _dflash_one_block()
        dflash_tokens += int(emitted)
        accept_lens.append(int(n_acc))

    dflash_s = max(1e-9, time.time() - t0)

    # ---- Baseline: token-by-token verify for same number of emitted tokens ----
    baseline_tokens = int(dflash_tokens)
    # Reset the request back to the original (ctx + pending anchor) state and
    # re-prefill, so baseline starts from the same prompt distribution.
    seqbuf.token_ids[0, :ctx_len] = ctx_ids
    seqbuf.token_ids[0, ctx_len] = np.int32(anchor_id)
    seqbuf.num_tokens[0] = int(prompt_len)
    seqbuf.num_tokens_no_spec[0] = int(prompt_len)
    seqbuf.num_computed_tokens[0] = 0
    seqbuf.temperature[0] = 0.0
    seqbuf.top_k[0] = 1
    seqbuf.top_p[0] = 1.0
    seqbuf.min_p[0] = 0.0
    _prefill_from_scratch()

    active_mask_full_cpu[:] = True
    scheduled_full_cpu[:] = 0

    # Warmup excluded from timing.
    for _ in range(int(args.warmup_blocks) * int(block_size)):
        scheduled_full_cpu[0] = 1
        _ctx_unused, greedy_ids, input_ids_buf, position_ids_buf, _m = executor.execute_verify(
            num_tokens=1,
            scheduled_full_cpu=scheduled_full_cpu,
            active_mask_full_cpu=active_mask_full_cpu,
            input_ids_buf=input_ids_buf,
            position_ids_buf=position_ids_buf,
            padded_num_reqs=1,
            token_ids_cpu=seqbuf.token_ids,
            num_computed_tokens_cpu=seqbuf.num_computed_tokens,
            temperature_cpu=seqbuf.temperature,
            top_p_cpu=seqbuf.top_p,
            top_k_cpu=seqbuf.top_k,
            min_p_cpu=seqbuf.min_p,
            page_table_cpu=page_table_cpu,
            page_table_version=page_table_version,
        )
        next_id = int(np.asarray(greedy_ids)[0])
        base_len = int(seqbuf.num_computed_tokens[0])
        seqbuf.token_ids[0, base_len + 1] = np.int32(next_id)
        seqbuf.num_computed_tokens[0] = int(base_len + 1)
        seqbuf.num_tokens[0] = int(base_len + 2)
        seqbuf.num_tokens_no_spec[0] = int(base_len + 2)

    t1 = time.time()
    for _ in range(baseline_tokens):
        scheduled_full_cpu[0] = 1
        _ctx_unused, greedy_ids, input_ids_buf, position_ids_buf, _m = executor.execute_verify(
            num_tokens=1,
            scheduled_full_cpu=scheduled_full_cpu,
            active_mask_full_cpu=active_mask_full_cpu,
            input_ids_buf=input_ids_buf,
            position_ids_buf=position_ids_buf,
            padded_num_reqs=1,
            token_ids_cpu=seqbuf.token_ids,
            num_computed_tokens_cpu=seqbuf.num_computed_tokens,
            temperature_cpu=seqbuf.temperature,
            top_p_cpu=seqbuf.top_p,
            top_k_cpu=seqbuf.top_k,
            min_p_cpu=seqbuf.min_p,
            page_table_cpu=page_table_cpu,
            page_table_version=page_table_version,
        )
        next_id = int(np.asarray(greedy_ids)[0])
        base_len = int(seqbuf.num_computed_tokens[0])
        seqbuf.token_ids[0, base_len + 1] = np.int32(next_id)
        seqbuf.num_computed_tokens[0] = int(base_len + 1)
        seqbuf.num_tokens[0] = int(base_len + 2)
        seqbuf.num_tokens_no_spec[0] = int(base_len + 2)
    baseline_s = max(1e-9, time.time() - t1)
    accept_mean = float(sum(accept_lens)) / float(max(1, len(accept_lens)))

    print(
        json.dumps(
            {
                "sample_idx": int(i),
                "ctx_len": int(ctx_len),
                "block_size": int(block_size),
                "blocks": int(args.blocks),
                "dflash_tokens": int(dflash_tokens),
                "dflash_s": float(dflash_s),
                "dflash_tok_s": float(dflash_tokens) / float(dflash_s),
                "accept_len_mean": float(accept_mean),
                "baseline_tokens": int(baseline_tokens),
                "baseline_s": float(baseline_s),
                "baseline_tok_s": float(baseline_tokens) / float(baseline_s),
                "speedup_x": float(baseline_s) / float(dflash_s),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
