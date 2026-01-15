#!/usr/bin/env python3
"""Naive DFlash speculative decoding for correctness (not for speed).

This script intentionally does NOT implement an optimized KV-cache verify kernel.
Instead it recomputes the target forward on the full prefix each verify step.

Why keep this:
- Lets us validate the DFlash training objective and block semantics on TPU
  *before* investing in high-performance inference integration.
- Produces acceptance-rate diagnostics (how many draft tokens match target greedy).

Use small contexts for this correctness harness.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def _encode_prompt(tokenizer_path: str, text: str, *, max_len: int) -> np.ndarray:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, use_fast=True)
    out = tok(text, truncation=True, max_length=int(max_len), return_tensors="np")
    return out["input_ids"].astype(np.int32)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-snapshot-dir", required=True)
    ap.add_argument("--draft-params", required=True, help="draft_params.msgpack from tpu_dflash_train_draft_from_cache.py")
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--num-context-features", type=int, default=4)
    ap.add_argument("--draft-layers", type=int, default=4)
    ap.add_argument("--mlp-ratio", type=float, default=4.0)
    ap.add_argument("--hidden-act", type=str, default="silu")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-prompt-len", type=int, default=512)
    ap.add_argument("--sharding-axis-dims", default="1,8,1,1,1")
    ap.add_argument("--platform", default="tpu", choices=["tpu", "cpu"])
    ap.add_argument("--prompt", default="You are a helpful assistant.\n\nUser: Explain DFlash in one paragraph.\nAssistant:")
    args = ap.parse_args()

    from tpu_dflash_lib import (
        DFlashDraftConfig,
        build_rope,
        build_target_layer_ids,
        load_json,
        make_dflash_draft_module,
        require_hf_token,
        set_shm_caches,
    )

    set_shm_caches()
    require_hf_token()
    if args.platform == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import jax
    import jax.numpy as jnp

    from easydel import AutoEasyDeLModelForCausalLM

    teacher_snapshot = Path(args.teacher_snapshot_dir).resolve()
    cfg = load_json(teacher_snapshot / "config.json")
    sharding_axis_dims = tuple(int(x) for x in str(args.sharding_axis_dims).split(",") if x.strip())
    if len(sharding_axis_dims) != 5:
        raise ValueError("--sharding-axis-dims must have 5 comma-separated ints (dp,fsdp,ep,tp,sp)")

    teacher = AutoEasyDeLModelForCausalLM.from_pretrained(
        str(teacher_snapshot),
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        auto_shard_model=True,
        sharding_axis_dims=sharding_axis_dims,
        verbose=False,
        from_torch=True,
    )

    target_layer_ids = build_target_layer_ids(int(cfg["num_hidden_layers"]), int(args.num_context_features))
    rope = build_rope(cfg=cfg, dtype=jnp.bfloat16)

    hidden = int(cfg["hidden_size"])
    k = int(len(target_layer_ids))
    dcfg = DFlashDraftConfig(
        hidden_size=hidden,
        num_layers=int(args.draft_layers),
        mlp_ratio=float(args.mlp_ratio),
        hidden_act=str(args.hidden_act),
        num_attention_heads=int(cfg["num_attention_heads"]),
        num_key_value_heads=int(cfg["num_key_value_heads"]),
        head_dim=int(cfg["head_dim"]),
        max_position_embeddings=int(cfg["max_position_embeddings"]),
        rope_theta=float(cfg["rope_theta"]),
        rope_scaling=cfg.get("rope_scaling"),
        rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-5)),
        block_size=int(args.block_size),
        num_context_features=k,
    )

    Draft = make_dflash_draft_module()
    draft = Draft(cfg=dcfg)

    # Tokenize prompt.
    prompt_ids = _encode_prompt(str(teacher_snapshot), str(args.prompt), max_len=int(args.max_prompt_len))
    prompt_ids = prompt_ids.astype(np.int32)

    seq = jnp.asarray(prompt_ids[None, :], dtype=jnp.int32)  # [1, S]

    # Load draft params (need a template with the correct shapes).
    import flax

    rng = jax.random.PRNGKey(0)
    dummy_ctx = jnp.zeros((1, int(seq.shape[1]), k * hidden), dtype=jnp.bfloat16)
    dummy_anchor = jnp.zeros((1, hidden), dtype=jnp.bfloat16)
    with teacher.mesh:
        params_template = draft.init(rng, context_features=dummy_ctx, anchor_embedding=dummy_anchor, rope=rope)
    params = flax.serialization.from_bytes(params_template, Path(args.draft_params).read_bytes())

    lm_head = teacher.get_lm_head()
    kernel = jax.lax.stop_gradient(lm_head.kernel.value)
    bias = jax.lax.stop_gradient(lm_head.bias.value) if getattr(lm_head, "bias", None) is not None else None

    accepted = 0
    proposed = 0
    t0 = time.time()

    for _ in range(int(args.max_new_tokens)):
        with teacher.mesh:
            out = teacher(input_ids=seq, output_hidden_states=True, apply_lm_head=True)
            logits_last = out.logits[:, -1, :]  # [1, vocab]
            next_id = jnp.argmax(logits_last, axis=-1).astype(jnp.int32)  # [1]

            # Build context features for the *current* prefix tokens.
            hs = out.hidden_states
            parts = [hs[int(lid)] for lid in target_layer_ids]
            ctx_feat = jnp.concatenate(parts, axis=-1)  # [1, S, K*hidden]

            anchor_id = seq[:, -1]
            anchor_emb = teacher.get_embedding()(anchor_id.astype("i4"))

            draft_h = draft.apply(params, context_features=ctx_feat, anchor_embedding=anchor_emb, rope=rope)
            hs_d = draft_h[:, 1:, :]  # [1, block-1, hidden]
            d_logits = jnp.einsum("bsh,hv->bsv", hs_d.astype(jnp.bfloat16), kernel.astype(jnp.bfloat16))
            if bias is not None:
                d_logits = d_logits + bias[None, None, :]
            draft_tokens = jnp.argmax(d_logits, axis=-1).astype(jnp.int32)  # [1, block-1]

            # Verify greedily by recomputing target tokens one by one (slow but correct).
            # Accept while draft matches target greedy.
            accept_len = 0
            cur = seq
            for j in range(int(args.block_size) - 1):
                out2 = teacher(input_ids=cur, apply_lm_head=True)
                t_next = jnp.argmax(out2.logits[:, -1, :], axis=-1).astype(jnp.int32)
                if int(draft_tokens[0, j]) != int(t_next[0]):
                    break
                accept_len += 1
                cur = jnp.concatenate([cur, t_next[:, None]], axis=1)

            if accept_len > 0:
                seq = jnp.concatenate([seq, draft_tokens[:, :accept_len]], axis=1)
                accepted += accept_len
                proposed += int(args.block_size) - 1
            else:
                seq = jnp.concatenate([seq, next_id[:, None]], axis=1)
                proposed += int(args.block_size) - 1

    dt = max(1e-9, time.time() - t0)
    print(
        json.dumps(
            {
                "prompt_len": int(prompt_ids.shape[0]),
                "final_len": int(seq.shape[1]),
                "max_new_tokens": int(args.max_new_tokens),
                "block_size": int(args.block_size),
                "accepted_tokens": int(accepted),
                "proposed_tokens": int(proposed),
                "accept_rate": float(accepted) / float(max(1, proposed)),
                "wall_s": float(dt),
                "tok_s_total": float(int(args.max_new_tokens)) / float(dt),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
