#!/usr/bin/env python3
"""
Minimal repro for TPU crashes in eSurge verify execution (DFLASH path).

Goal: compile the verify executor, run a single verify-mode "prefill" step,
and exit. This avoids dataset loading so we can isolate backend/compiler issues.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np


def _load_dotenv(path: Path) -> None:
    try:
        import dotenv
    except Exception:
        return
    if path.exists():
        dotenv.load_dotenv(path, override=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-snapshot-dir", required=True, help="Local HF snapshot dir (from_torch=True).")
    ap.add_argument("--teacher-easydel-dir", default="", help="Optional EasyDeL-native dir (from_torch=False).")
    ap.add_argument("--ctx-len", type=int, default=1024, help="Full context length including anchor token.")
    ap.add_argument("--prefill-chunk", type=int, default=256)
    ap.add_argument("--page-size", type=int, default=128)
    ap.add_argument("--hbm-utilization", type=float, default=0.15)
    ap.add_argument("--attn-mechanism", default="ragged_page_attention_v2")
    ap.add_argument("--target-layer-ids", default="1,8,14,21")
    ap.add_argument("--verify-add-one-for-pre-layer-capture", action="store_true")
    ap.add_argument("--sharding-axis-dims", default="1,8,1,1,1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--platform", choices=["tpu", "cpu"], default="tpu")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _load_dotenv(repo_root / ".env")
    if args.platform == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import jax
    import jax.numpy as jnp

    from easydel import AutoEasyDeLModelForCausalLM
    from easydel.inference.esurge.runners.execution_manager import ExecutionManager
    from easydel.inference.esurge.runners.sequence_buffer import SequenceBuffer
    from easydel.inference.esurge.runners.states import CachedRequestState
    from easydel.inference.sampling_params import SamplingParams

    snapshot = Path(args.model_snapshot_dir).resolve()
    teacher_easydel_dir = Path(args.teacher_easydel_dir).resolve() if args.teacher_easydel_dir.strip() else None
    sharding_axis_dims = tuple(int(x) for x in str(args.sharding_axis_dims).split(",") if x.strip())
    if len(sharding_axis_dims) != 5:
        raise ValueError("--sharding-axis-dims must have 5 comma-separated ints (dp,fsdp,ep,tp,sp)")

    target_layer_ids = [int(x) for x in str(args.target_layer_ids).split(",") if x.strip()]

    t0 = time.time()
    if teacher_easydel_dir is not None and teacher_easydel_dir.exists():
        print(f"[repro] loading teacher easydel: {teacher_easydel_dir}", flush=True)
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
        print(f"[repro] converting teacher from torch snapshot: {snapshot}", flush=True)
        model = AutoEasyDeLModelForCausalLM.from_pretrained(
            str(snapshot),
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            auto_shard_model=True,
            sharding_axis_dims=sharding_axis_dims,
            verbose=False,
            from_torch=True,
        )

    model = model.merge_module(
        model.new_graphdef(attn_mechanism=str(args.attn_mechanism)),
        model.graphstate,
        model.graphother,
    )
    print(f"[repro] teacher ready in {time.time() - t0:.1f}s (attn={args.attn_mechanism})", flush=True)

    mesh = model.mesh
    text_cfg = model.config.get_text_config()
    vocab_size = int(text_cfg.vocab_size)

    ctx_len_full = int(args.ctx_len)
    ctx_len = int(max(1, ctx_len_full - 1))
    max_model_len = int(ctx_len_full)

    empty_sharding = jax.NamedSharding(mesh, jax.sharding.PartitionSpec())
    seqbuf = SequenceBuffer(
        max_num_reqs=1,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        vocab_size=vocab_size,
        page_sizes=[int(args.page_size)],
        sharding=empty_sharding,
    )

    metadata = model.create_ragged_page_cache_config(
        hbm_utilization=float(args.hbm_utilization),
        page_size=int(args.page_size),
        max_length=max_model_len,
    )
    max_pages_per_req = int(getattr(metadata, "max_num_pages_per_req"))
    page_ids = (list(range(max_pages_per_req)),)

    rid = "repro"
    sp = SamplingParams(max_tokens=1, temperature=0.0, top_k=1, top_p=1.0)
    req_state = CachedRequestState(
        req_id=rid,
        prompt_token_ids=[0, 0],
        sampling_params=sp,
        generator=jax.random.PRNGKey(int(args.seed)),
        page_ids=page_ids,
        num_computed_tokens=0,
        output_token_ids=[],
    )
    seqbuf.add_request(req_state, req_index=0)

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
        verify_add_one_for_pre_layer_capture=bool(args.verify_add_one_for_pre_layer_capture),
    )
    t1 = time.time()
    executor.compile(
        num_tokens_paddings=sorted({1, int(max(1, min(ctx_len, int(args.prefill_chunk))))}),
        num_reqs_max_model_len=1,
        max_pages_per_req=int(metadata.max_num_pages_per_req),
        max_num_reqs=1,
        metadata=metadata,
        num_reqs_paddings=[1],
    )
    print(f"[repro] executor compiled in {time.time() - t1:.1f}s", flush=True)

    input_ids_buf = jax.device_put(jnp.zeros((int(max_model_len),), dtype=jnp.int32), empty_sharding)
    position_ids_buf = jax.device_put(jnp.zeros((int(max_model_len),), dtype=jnp.int32), empty_sharding)
    scheduled_full_cpu = np.zeros((1,), dtype=np.int32)
    active_mask_full_cpu = np.asarray([True], dtype=bool)
    page_table_cpu = seqbuf.page_table[0].get_cpu_tensor()
    page_table_version = getattr(seqbuf.page_table[0], "cpu_version", None)

    rng = np.random.default_rng(int(args.seed))
    ctx_ids = rng.integers(low=0, high=vocab_size, size=(ctx_len,), dtype=np.int32)
    anchor_id = int(rng.integers(low=0, high=vocab_size, dtype=np.int32))

    prompt_len = int(ctx_len + 1)
    seqbuf.token_ids[0, :prompt_len] = np.concatenate([ctx_ids, np.asarray([anchor_id], dtype=np.int32)], axis=0)
    seqbuf.num_tokens[0] = int(prompt_len)
    seqbuf.num_tokens_no_spec[0] = int(prompt_len)
    seqbuf.num_computed_tokens[0] = 0
    seqbuf.temperature[0] = 0.0
    seqbuf.top_k[0] = 1
    seqbuf.top_p[0] = 1.0
    seqbuf.min_p[0] = 0.0

    total = int(prompt_len - 1)  # exclude anchor token
    done = 0
    prefill_chunk = int(max(1, int(args.prefill_chunk)))

    print(f"[repro] starting execute_verify prefill: total={total} chunk={prefill_chunk}", flush=True)
    with mesh:
        while done < total:
            step = int(min(prefill_chunk, total - done))
            seqbuf.num_computed_tokens[0] = int(done)
            scheduled_full_cpu[0] = int(step)
            ctx_part, _greedy_unused, input_ids_buf, position_ids_buf, _m = executor.execute_verify(
                num_tokens=int(step),
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
            ctx_part = jax.block_until_ready(ctx_part)
            done += step
            print(f"[repro] ok chunk done={done}/{total} ctx_part={tuple(ctx_part.shape)}", flush=True)

    print("[repro] SUCCESS: execute_verify ran without crashing.", flush=True)


if __name__ == "__main__":
    main()
