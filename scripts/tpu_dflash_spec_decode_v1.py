#!/usr/bin/env python3
"""DFlash speculative decoding spec-v1 (correctness-first, block verify).

Compared to `tpu_dflash_spec_decode_naive.py`, this matches the SGLang PR #16818
verify rule by running a *single* target verify forward on the whole block and
computing:

  accept_len = prefix-match length of (candidates[:,1:] == target_predict[:,:-1])
  bonus      = target_predict[:, accept_len]

This script is still not performance-optimized on TPU (it recomputes target
forward on the full prefix each iteration), but it is the correctness harness we
use before implementing KV-cache rollback/commit.
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
    ap.add_argument("--draft-params", required=True, help="draft_params.msgpack from TPU trainer.")
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--num-context-features", type=int, default=4)
    ap.add_argument("--draft-layers", type=int, default=4)
    ap.add_argument("--mlp-ratio", type=float, default=4.0)
    ap.add_argument("--hidden-act", type=str, default="silu")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-prompt-len", type=int, default=512)
    ap.add_argument(
        "--also-run-baseline",
        action="store_true",
        help="Also run baseline greedy (target-only) with the same prompt and report speedup_x.",
    )
    ap.add_argument("--sharding-axis-dims", default="1,8,1,1,1")
    ap.add_argument("--platform", default="tpu", choices=["tpu", "cpu"])
    ap.add_argument("--prompt", default="You are a helpful assistant.\n\nUser: Explain DFlash in one paragraph.\nAssistant:")
    ap.add_argument(
        "--layer-ids-prelayer",
        default="",
        help="Comma-separated pre-layer capture ids (EasyDeL hidden_states indices). "
        "If empty, uses build_target_layer_ids(num_layers, K) (which yields pre-layer ids in our pipeline).",
    )
    args = ap.parse_args()

    from dflash_gptoss.easydel_dflash_spec_v1 import dflash_accept_len_and_bonus
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

    num_target_layers = int(cfg["num_hidden_layers"])
    if args.layer_ids_prelayer.strip():
        target_layer_ids = [int(x) for x in args.layer_ids_prelayer.split(",") if x.strip()]
    else:
        # NOTE: in our EasyDeL pipeline, we treat these as *pre-layer* capture ids.
        target_layer_ids = build_target_layer_ids(num_target_layers, int(args.num_context_features))
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

    prompt_ids = _encode_prompt(str(teacher_snapshot), str(args.prompt), max_len=int(args.max_prompt_len))
    seq = jnp.asarray(prompt_ids[None, :], dtype=jnp.int32)  # [1, S]

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

    block_size = int(args.block_size)
    for _ in range(int(args.max_new_tokens)):
        with teacher.mesh:
            out = teacher(input_ids=seq, output_hidden_states=True, apply_lm_head=True)
            hs = out.hidden_states
            if hs is None:
                raise RuntimeError("Expected teacher.hidden_states (output_hidden_states=True).")

            parts = [hs[int(lid)] for lid in target_layer_ids]
            ctx_feat = jnp.concatenate(parts, axis=-1)  # [1, S, K*hidden]

            anchor_id = seq[:, -1]  # current verified token
            anchor_emb = teacher.get_embedding()(anchor_id.astype("i4"))

            draft_h = draft.apply(params, context_features=ctx_feat, anchor_embedding=anchor_emb, rope=rope)
            hs_d = draft_h[:, 1:, :]  # [1, B-1, hidden]
            d_logits = jnp.einsum("bsh,hv->bsv", hs_d.astype(jnp.bfloat16), kernel.astype(jnp.bfloat16))
            if bias is not None:
                d_logits = d_logits + bias[None, None, :]
            draft_tokens = jnp.argmax(d_logits, axis=-1).astype(jnp.int32)  # [1, B-1]

            # candidates: [1, B]
            candidates = jnp.concatenate([anchor_id[:, None], draft_tokens], axis=1)

            # Verify: run target on prefix + proposed tokens, then take logits for the
            # last B positions (current token + draft tokens).
            seq_ext = jnp.concatenate([seq, draft_tokens], axis=1)
            out_v = teacher(input_ids=seq_ext, apply_lm_head=True)
            logits_window = out_v.logits[:, -block_size:, :]  # [1, B, vocab]
            target_predict = jnp.argmax(logits_window, axis=-1).astype(jnp.int32)  # [1, B]

            accept_len, bonus = dflash_accept_len_and_bonus(
                candidates=candidates,
                target_predict=target_predict,
            )

            # Commit: append accepted draft tokens and then the bonus token.
            n_acc = int(accept_len[0])
            if n_acc > 0:
                seq = jnp.concatenate([seq, draft_tokens[:, :n_acc]], axis=1)
                accepted += n_acc
            seq = jnp.concatenate([seq, bonus[:, None]], axis=1)

            proposed += block_size - 1

    dt = max(1e-9, time.time() - t0)
    out_obj = {
        "prompt_len": int(prompt_ids.shape[0]),
        "final_len": int(seq.shape[1]),
        "max_new_tokens": int(args.max_new_tokens),
        "block_size": int(args.block_size),
        "accepted_tokens": int(accepted),
        "proposed_tokens": int(proposed),
        "accept_rate": float(accepted) / float(max(1, proposed)),
        "wall_s": float(dt),
        "tok_s_total": float(int(args.max_new_tokens)) / float(dt),
        "target_layer_ids_prelayer": target_layer_ids,
    }

    if bool(args.also_run_baseline):
        # Baseline: greedy decode target-only (recomputes full prefix each step; correctness harness).
        seq_b = jnp.asarray(prompt_ids[None, :], dtype=jnp.int32)
        t1 = time.time()
        for _ in range(int(args.max_new_tokens)):
            with teacher.mesh:
                out_b = teacher(input_ids=seq_b, apply_lm_head=True)
                next_id = jnp.argmax(out_b.logits[:, -1, :], axis=-1).astype(jnp.int32)
                seq_b = jnp.concatenate([seq_b, next_id[:, None]], axis=1)
        dt_b = max(1e-9, time.time() - t1)
        tok_s_b = float(int(args.max_new_tokens)) / float(dt_b)
        out_obj.update(
            {
                "baseline_wall_s": float(dt_b),
                "baseline_tok_s_total": float(tok_s_b),
                "speedup_x": float(out_obj["tok_s_total"]) / float(max(tok_s_b, 1e-9)),
            }
        )

    print(json.dumps(out_obj, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
