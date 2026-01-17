#!/usr/bin/env python3
"""TPU baseline decode benchmark using EasyDeL eSurge (paged KV cache).

This avoids the HF-style `use_cache/past_key_values` API (not supported by
AutoEasyDeLModelForCausalLM.__call__). eSurge is the canonical cached-decode
path on TPU.
"""

from __future__ import annotations

import argparse
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-snapshot-dir", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--page-size", type=int, default=128)
    ap.add_argument("--hbm-utilization", type=float, default=0.5)
    ap.add_argument("--prompt", default="You are a helpful assistant.\n\nUser: Explain DFlash in one paragraph.\nAssistant:")
    args = ap.parse_args()

    os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

    import jax.numpy as jnp
    from easydel import AutoEasyDeLModelForCausalLM
    from easydel.inference import SamplingParams, eSurge
    from transformers import AutoTokenizer

    teacher = AutoEasyDeLModelForCausalLM.from_pretrained(
        str(args.teacher_snapshot_dir),
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        auto_shard_model=True,
        sharding_axis_dims=(1, 8, 1, 1, 1),
        verbose=False,
        from_torch=True,
    )
    # TPU correctness: our runtime ships a compatibility implementation of
    # ragged_page_attention_v2 (gathers KV from pages). The v3 path requires a
    # full EJKernel v3 implementation and otherwise breaks multi-token verify
    # and can also skew baseline comparisons. Force v2 when requested.
    if os.environ.get("DFLASH_FORCE_RAGGED_V2", "1").lower() in ("1", "true", "yes", "y", "on"):
        teacher = teacher.merge_module(
            teacher.new_graphdef(attn_mechanism="ragged_page_attention_v2"),
            teacher.graphstate,
            teacher.graphother,
        )

    tok = AutoTokenizer.from_pretrained(str(args.teacher_snapshot_dir), local_files_only=True, use_fast=True)
    engine = eSurge(
        model=teacher,
        max_model_len=int(args.max_model_len),
        max_num_seqs=1,
        hbm_utilization=float(args.hbm_utilization),
        page_size=int(args.page_size),
        processor=tok,
    )
    engine.initiate()

    sp = SamplingParams(max_tokens=int(args.max_new_tokens), temperature=0.0, top_k=1, top_p=1.0)
    t0 = time.time()
    out = engine.generate([str(args.prompt)], sampling_params=sp, use_tqdm=False)[0]
    dt = max(1e-9, time.time() - t0)
    # Best-effort token count.
    toks = 0
    try:
        toks = len(out.outputs[0].token_ids)
    except Exception:
        pass
    print(
        {
            "mode": "esurge_baseline",
            "max_new_tokens": int(args.max_new_tokens),
            "wall_s": float(dt),
            "output_tokens": int(toks),
            "output_toks_per_s": float(toks) / float(dt) if toks else None,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
