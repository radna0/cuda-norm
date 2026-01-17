#!/usr/bin/env python3
"""Sweep multi-token verify parity on many cache samples (TPU, EasyDeL eSurge).

This is the *batch* version of `tpu_verify_multitoken_parity.py`:
- loads the teacher once
- compiles verify buckets once
- runs multiple cache indices and writes JSONL + a summary

We track three things:
1) `match_cached_targets_vs_verify`: cached `target_ids` match verify-block greedy tokens.
2) `match_1tok_vs_block0`: 1-token greedy equals verify-block position 0 greedy.
   This can fail due to kernel numerical differences; it is informative but not
   strictly required for DFlash acceptance on the block-verify path.
3) `verify_block0_changes_with_future`: whether verify-block token 0 changes if
   we overwrite future draft tokens. This MUST be false (causality).
"""

from __future__ import annotations

import argparse
import json
import os
import random
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


def _parse_csv_ints(s: str) -> list[int]:
    out: list[int] = []
    for part in (s or "").split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--teacher-snapshot-dir", required=True)
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--page-size", type=int, default=128)
    ap.add_argument("--hbm-utilization", type=float, default=0.20)
    ap.add_argument("--prefill-chunk", type=int, default=256)
    ap.add_argument("--sample-indices", default="", help="Comma-separated indices to test.")
    ap.add_argument("--random-samples", type=int, default=0, help="If >0, sample N random indices.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-jsonl", default="", help="If set, write per-sample results to JSONL.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    local_easydel = repo_root / "external" / "EasyDeL"
    if local_easydel.exists():
        sys.path.insert(0, str(local_easydel))
    _load_dotenv(repo_root / ".env")

    os.environ.setdefault("EASYDEL_SKIP_VERSION_CHECK", "1")
    os.environ.setdefault("DFLASH_FORCE_RAGGED_V2", "1")
    os.environ.setdefault("EASYDEL_VERIFY_DISALLOW_ON_DEMAND_COMPILE", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    import jax
    import jax.numpy as jnp

    try:
        from transformers import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:
        pass

    from easydel import AutoEasyDeLModelForCausalLM
    from easydel.inference.esurge.runners.execution_manager import ExecutionManager
    from easydel.inference.esurge.runners.sequence_buffer import SequenceBuffer
    from easydel.inference.esurge.runners.states import CachedRequestState
    from easydel.inference.sampling_params import SamplingParams

    cache_dir = Path(args.cache_dir).resolve()
    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    ctx_len = int(meta["ctx_len"])

    ctx_token_ids = np.load(cache_dir / "ctx_token_ids.npy", mmap_mode="r")
    anchor_ids = np.load(cache_dir / "anchor_ids.npy", mmap_mode="r")
    target_ids = np.load(cache_dir / "target_ids.npy", mmap_mode="r")
    n_samples = int(ctx_token_ids.shape[0])

    pos_path = cache_dir / meta.get("ctx_pos_start_file", "ctx_pos_start_i32.npy")
    pos_arr = np.load(pos_path, mmap_mode="r") if pos_path.exists() else None

    indices: list[int] = []
    if str(args.sample_indices).strip():
        indices = _parse_csv_ints(str(args.sample_indices))
    elif int(args.random_samples) > 0:
        rng = random.Random(int(args.seed))
        pop = list(range(n_samples))
        rng.shuffle(pop)
        indices = pop[: min(int(args.random_samples), n_samples)]
    else:
        indices = [0]

    seen: set[int] = set()
    indices = [i for i in indices if (i not in seen and not seen.add(i))]
    for i in indices:
        if i < 0 or i >= n_samples:
            raise ValueError(f"sample_idx={i} out of range (0..{n_samples-1})")

    teacher_snapshot = Path(args.teacher_snapshot_dir).resolve()
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

    mesh = teacher.mesh
    empty_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())
    max_model_len = int((ctx_len + 1) + (int(args.block_size) - 1))
    vocab_size = int(teacher.config.get_text_config().vocab_size)

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
    max_pages_per_req = int(metadata.max_num_pages_per_req)
    page_ids = (list(range(max_pages_per_req)),)

    sp = SamplingParams(max_tokens=1, temperature=0.0, top_k=1, top_p=1.0)
    req = CachedRequestState(
        req_id="parity-sweep",
        prompt_token_ids=[0],
        sampling_params=sp,
        generator=jax.random.PRNGKey(0),
        page_ids=page_ids,
        num_computed_tokens=0,
        output_token_ids=[],
    )
    seqbuf.add_request(req, req_index=0)
    seqbuf.temperature[0] = 0.0
    seqbuf.top_k[0] = 1
    seqbuf.top_p[0] = 1.0
    seqbuf.min_p[0] = 0.0

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
    prefill_bucket = int(min(int(args.prefill_chunk), int(ctx_len)))
    executor.compile(
        num_tokens_paddings=sorted({1, int(args.block_size), int(max(1, prefill_bucket))}),
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

    def _reset_kv_pages() -> None:
        def _zero(x):
            if isinstance(x, jax.Array):
                return jnp.zeros_like(x)
            return x

        with mesh:
            executor.kv_pages = jax.tree_util.tree_map(_zero, executor.kv_pages)
            executor.kv_pages = jax.block_until_ready(executor.kv_pages)

    def _prefill_ctx_only(*, prompt_ids: np.ndarray, prompt_len: int, pos_off_cpu: np.ndarray) -> None:
        nonlocal input_ids_buf, position_ids_buf

        seqbuf.token_ids[0, :prompt_len] = prompt_ids
        seqbuf.num_tokens[0] = int(prompt_len)
        seqbuf.num_tokens_no_spec[0] = int(prompt_len)
        seqbuf.num_computed_tokens[0] = 0
        seqbuf.temperature[0] = 0.0
        seqbuf.top_k[0] = 1
        seqbuf.top_p[0] = 1.0
        seqbuf.min_p[0] = 0.0

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
        seqbuf.num_computed_tokens[0] = int(ctx_len)

    out_f = None
    if str(args.output_jsonl).strip():
        out_path = Path(str(args.output_jsonl)).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_path.open("w", encoding="utf-8")

    results = []
    for i in indices:
        ctx_ids = np.asarray(ctx_token_ids[i]).astype(np.int32)
        anchor_id = int(np.asarray(anchor_ids[i]))
        cached_targets = np.asarray(target_ids[i]).astype(np.int32)
        prompt_ids = np.concatenate([ctx_ids, np.asarray([anchor_id], dtype=np.int32)], axis=0)
        prompt_len = int(prompt_ids.shape[0])

        try:
            pos_off = int(np.asarray(pos_arr[int(i)])) if pos_arr is not None else 0
        except Exception:
            pos_off = 0
        pos_off_cpu = np.asarray([np.int32(pos_off)], dtype=np.int32)

        # 1-token greedy next (fresh KV + ctx-only prefill).
        _reset_kv_pages()
        _prefill_ctx_only(prompt_ids=prompt_ids, prompt_len=prompt_len, pos_off_cpu=pos_off_cpu)
        scheduled_full_cpu[0] = 1
        _ctx_unused, greedy1, input_ids_buf, position_ids_buf, _m = executor.execute_verify(
            num_tokens=int(args.block_size),
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
        next1 = int(np.asarray(greedy1)[0])

        # Rollout candidate tokens (fresh KV + ctx-only prefill).
        _reset_kv_pages()
        _prefill_ctx_only(prompt_ids=prompt_ids, prompt_len=prompt_len, pos_off_cpu=pos_off_cpu)
        seqbuf.num_tokens[0] = int(prompt_len)
        seqbuf.num_tokens_no_spec[0] = int(prompt_len)
        tgt = np.empty((int(args.block_size) - 1,), dtype=np.int32)
        for j in range(int(args.block_size) - 1):
            base_len = int(seqbuf.num_computed_tokens[0])
            scheduled_full_cpu[0] = 1
            _ctx_unused, greedy_j, input_ids_buf, position_ids_buf, _m = executor.execute_verify(
                num_tokens=int(args.block_size),
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
            tok = int(np.asarray(greedy_j)[0])
            tgt[j] = np.int32(tok)
            seqbuf.token_ids[0, base_len + 1] = np.int32(tok)
            seqbuf.num_computed_tokens[0] = int(base_len + 1)
            seqbuf.num_tokens[0] = int(base_len + 2)
            seqbuf.num_tokens_no_spec[0] = int(base_len + 2)

        cand = np.empty((int(args.block_size),), dtype=np.int32)
        cand[0] = np.int32(anchor_id)
        cand[1:] = tgt

        # Verify block greedy on candidate tokens (fresh KV + ctx-only prefill).
        _reset_kv_pages()
        _prefill_ctx_only(prompt_ids=prompt_ids, prompt_len=prompt_len, pos_off_cpu=pos_off_cpu)
        seqbuf.token_ids[0, int(ctx_len) : int(ctx_len + int(args.block_size))] = cand
        seqbuf.num_tokens[0] = int(ctx_len + int(args.block_size))
        seqbuf.num_tokens_no_spec[0] = int(ctx_len + int(args.block_size))
        scheduled_full_cpu[0] = int(args.block_size)
        _ctx_unused, greedy_block, input_ids_buf, position_ids_buf, _m = executor.execute_verify(
            num_tokens=int(args.block_size),
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

        # Causality probe: overwrite future tokens.
        cand_zero = cand.copy()
        cand_zero[1:] = np.int32(0)
        _reset_kv_pages()
        _prefill_ctx_only(prompt_ids=prompt_ids, prompt_len=prompt_len, pos_off_cpu=pos_off_cpu)
        seqbuf.token_ids[0, int(ctx_len) : int(ctx_len + int(args.block_size))] = cand_zero
        seqbuf.num_tokens[0] = int(ctx_len + int(args.block_size))
        seqbuf.num_tokens_no_spec[0] = int(ctx_len + int(args.block_size))
        scheduled_full_cpu[0] = int(args.block_size)
        _ctx_unused, greedy_block_zero, input_ids_buf, position_ids_buf, _m = executor.execute_verify(
            num_tokens=int(args.block_size),
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
        greedy_block_zero = np.asarray(greedy_block_zero, dtype=np.int32)

        out = {
            "sample_idx": int(i),
            "position_offset": int(pos_off),
            "anchor_id": int(anchor_id),
            "cached_target_ids": [int(x) for x in cached_targets.tolist()],
            "rollout_target_ids": [int(x) for x in tgt.tolist()],
            "next_token_1tok": int(next1),
            "cand_tokens": [int(x) for x in cand.tolist()],
            "verify_block_greedy": [int(x) for x in greedy_block[: int(args.block_size)].tolist()],
            "verify_block_zero_greedy": [int(x) for x in greedy_block_zero[: int(args.block_size)].tolist()],
            "verify_block0_changes_with_future": bool(int(greedy_block_zero[0]) != int(greedy_block[0])),
            "match_1tok_vs_block0": bool(int(next1) == int(greedy_block[0])),
            "match_cached_targets_vs_verify": bool(
                cached_targets.shape[0] == (int(args.block_size) - 1)
                and np.all(cached_targets == greedy_block[: int(args.block_size) - 1])
            ),
        }
        results.append(out)
        if out_f is not None:
            out_f.write(json.dumps(out, sort_keys=True) + "\n")

    if out_f is not None:
        out_f.close()

    n = len(results)
    match_targets = sum(1 for r in results if r["match_cached_targets_vs_verify"])
    match_1tok = sum(1 for r in results if r["match_1tok_vs_block0"])
    future_dep = sum(1 for r in results if r["verify_block0_changes_with_future"])
    print(
        json.dumps(
            {
                "num_samples": int(n),
                "match_cached_targets_vs_verify": int(match_targets),
                "match_1tok_vs_block0": int(match_1tok),
                "verify_block0_changes_with_future": int(future_dep),
                "first_sample": results[0] if results else None,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

