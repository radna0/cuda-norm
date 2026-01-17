#!/usr/bin/env python3
"""TPU DFlash spec-v1 decode with KV-cache (target) + draft ctx-KV caching (draft).

This is the first *speed-oriented* TPU decode harness:
- Target (teacher) runs in cached decode mode (no full-prefix recompute).
- Verification is implemented as sequential cached decode steps (correctness-first),
  using the SGLang accept rule:
    candidates[:, 1:] vs target_predict[:, :-1]
  but computed without a block-parallel TARGET_VERIFY kernel.
- Draft uses pre-materialized ctx KV per draft layer to avoid O(ctx_len) k/v projection.

Limitations (expected initially):
- batch_size=1 only
- verify is sequential (still O(block_size) teacher steps per iteration)
- no overlap scheduling
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
    ap.add_argument("--draft-run-dir", required=True, help="EasyDeL run dir containing config.json + model/ tensorstore.")
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-prompt-len", type=int, default=512)
    ap.add_argument("--sharding-axis-dims", default="1,8,1,1,1")
    ap.add_argument("--platform", default="tpu", choices=["tpu", "cpu"])
    ap.add_argument("--prompt", default="You are a helpful assistant.\n\nUser: Explain DFlash in one paragraph.\nAssistant:")
    ap.add_argument("--target-layer-ids", default="", help="Comma-separated pre-layer hidden_states indices to use for context features.")
    ap.add_argument("--use-afterlayer-ids", action="store_true", help="Interpret target-layer-ids as after-layer ids (apply +1 prelayer shift).")
    ap.add_argument("--also-run-baseline", action="store_true", help="Also run baseline greedy target-only cached decode for speedup_x.")
    args = ap.parse_args()

    if args.platform == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    from flax import nnx
    import jax
    import jax.numpy as jnp

    from easydel import AutoEasyDeLModelForCausalLM
    from easydel.inference.speculative import (
        DFlashDraftModel,
        DFlashDraftModelConfig,
        append_draft_ctx_kv,
        draft_forward_with_ctx_kv,
        materialize_draft_ctx_kv,
    )
    from easydel.inference.speculative.dflash import (
        dflash_accept_len_and_bonus,
        extract_dflash_context_features_from_hidden_states,
    )

    teacher_snapshot = Path(args.teacher_snapshot_dir).resolve()
    draft_run = Path(args.draft_run_dir).resolve()
    if not (draft_run / "config.json").exists():
        raise FileNotFoundError(draft_run / "config.json")

    # Transformers 4.57+ treats presence of `quantization_config` as "pre-quantized",
    # and expects a dict with `quant_method`. Our teacher snapshots are BF16; remove
    # null quantization fields entirely to avoid false quantization paths.
    cfg_path = teacher_snapshot / "config.json"
    try:
        if cfg_path.exists():
            d = json.loads(cfg_path.read_text(encoding="utf-8"))
            changed = False
            if d.get("quantization_config", "MISSING") is None:
                d.pop("quantization_config", None)
                changed = True
            if d.get("kv_cache_quantization_config", "MISSING") is None:
                d.pop("kv_cache_quantization_config", None)
                changed = True
            if changed:
                cfg_path.write_text(json.dumps(d, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        pass

    # Load teacher.
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

    # Load draft checkpoint (tensorstore) via nnx state restore.
    draft_cfg_json = json.loads((draft_run / "config.json").read_text(encoding="utf-8"))
    dcfg = DFlashDraftModelConfig(
        hidden_size=int(draft_cfg_json["hidden_size"]),
        num_layers=int(draft_cfg_json["num_layers"]),
        mlp_ratio=float(draft_cfg_json.get("mlp_ratio", 4.0)),
        hidden_act=str(draft_cfg_json.get("hidden_act", "silu")),
        num_attention_heads=int(draft_cfg_json["num_attention_heads"]),
        num_key_value_heads=int(draft_cfg_json["num_key_value_heads"]),
        head_dim=int(draft_cfg_json["head_dim"]),
        rms_norm_eps=float(draft_cfg_json.get("rms_norm_eps", 1e-5)),
        block_size=int(draft_cfg_json.get("block_size", int(args.block_size))),
        num_context_features=int(draft_cfg_json["num_context_features"]),
        qk_norm=bool(draft_cfg_json.get("qk_norm", True)),
        remat=bool(draft_cfg_json.get("remat", True)),
    )
    if int(args.block_size) != int(dcfg.block_size):
        raise ValueError(f"block_size mismatch: args={args.block_size} ckpt={dcfg.block_size}")

    rngs = nnx.Rngs(0)
    draft = DFlashDraftModel(dcfg, rngs=rngs)

    # Restore nnx state from tensorstore directory layout:
    # draft_run/model/... is zarr; rely on EasyDeL trainer's tensorstore layout.
    # We use nnx.state()/update_state() pattern via tensorstore in TPU env.
    draft_state = None
    # EasyDeL helper (preferred).
    try:
        from easydel.trainers.base_trainer import load_state_from_checkpoint  # type: ignore

        draft_state = load_state_from_checkpoint(str(draft_run), state_like=nnx.state(draft))
    except Exception:
        draft_state = None
    if draft_state is None:
        # Fallback: some EasyDeL builds expose the checkpoint loader under `easydel.trainers.utils`.
        try:
            from easydel.trainers.utils import load_state_from_checkpoint as load2  # type: ignore

            draft_state = load2(str(draft_run), state_like=nnx.state(draft))
        except Exception as e:
            raise RuntimeError(
                "Could not load EasyDeL tensorstore checkpoint for the draft model. "
                "Ensure the TPU environment has EasyDeL's checkpoint loader available."
            ) from e
    nnx.update(draft, draft_state)

    # Tokenize prompt and split into (prefix_without_last, current_token).
    prompt_ids = _encode_prompt(str(teacher_snapshot), str(args.prompt), max_len=int(args.max_prompt_len))
    if prompt_ids.shape[0] < 2:
        raise ValueError("prompt too short for cached decode harness")
    prefix_ids = prompt_ids[:-1]
    current_id = prompt_ids[-1:]

    prefix = jnp.asarray(prefix_ids[None, :], dtype=jnp.int32)
    cur = jnp.asarray(current_id[None, :], dtype=jnp.int32)  # [1,1]

    # Resolve layer ids for context features.
    if args.target_layer_ids.strip():
        tli = [int(x) for x in args.target_layer_ids.split(",") if x.strip()]
    else:
        # default: use the draft's K, evenly spaced.
        from tpu_dflash_lib import build_target_layer_ids, load_json

        cfg = load_json(teacher_snapshot / "config.json")
        tli = build_target_layer_ids(int(cfg["num_hidden_layers"]), int(dcfg.num_context_features))
    add_one = bool(args.use_afterlayer_ids)

    # Prefill teacher cache on prefix (without current token).
    with teacher.mesh:
        out_prefill = teacher(
            input_ids=prefix,
            output_hidden_states=True,
            apply_lm_head=False,
            use_cache=True,
        )
    cache = out_prefill.past_key_values
    if cache is None:
        raise RuntimeError("Teacher did not return past_key_values; cannot do cached decode.")

    # Build initial context features for prefix tokens (from prefill hidden_states).
    hs = out_prefill.hidden_states
    if hs is None:
        raise RuntimeError("Teacher did not return hidden_states (output_hidden_states=True).")
    ctx_feat = extract_dflash_context_features_from_hidden_states(
        hidden_states=hs,
        target_layer_ids=tli,
        add_one_for_pre_layer_capture=bool(add_one),
    )  # [1, S, K*hidden]
    ctx_hidden = draft.project_context_features(ctx_feat)  # [1, S, hidden]

    # Materialize draft ctx KV cache for all layers.
    rope = None
    try:
        from tpu_dflash_lib import build_rope, load_json

        cfg = load_json(teacher_snapshot / "config.json")
        rope = build_rope(cfg=cfg, dtype=jnp.bfloat16)
    except Exception as e:
        raise RuntimeError("Failed to build RoPE for draft.") from e

    ctx_kv = materialize_draft_ctx_kv(draft=draft, rope=rope, ctx_hidden=ctx_hidden)

    # Teacher embedding + lm_head for draft tokenization.
    lm_head = teacher.get_lm_head()
    emb_fn = teacher.get_embedding()
    kernel = jax.lax.stop_gradient(lm_head.kernel.value)
    bias = jax.lax.stop_gradient(lm_head.bias.value) if getattr(lm_head, "bias", None) is not None else None

    accepted = 0
    proposed = 0
    t0 = time.time()

    seq_len = int(prefix.shape[1]) + 1
    for _ in range(int(args.max_new_tokens)):
        # Draft propose B-1 tokens using ctx KV cache.
        with teacher.mesh:
            anchor_emb = emb_fn(cur.astype("i4"))[:, 0, :]  # [1,hidden]
        draft_h = draft_forward_with_ctx_kv(
            draft=draft,
            rope=rope,
            cache=ctx_kv,
            anchor_embedding=anchor_emb.astype(jnp.bfloat16),
            mask_embedding=draft.mask_embedding.value.astype(jnp.bfloat16),
            block_size=int(args.block_size),
        )
        hs_d = draft_h[:, 1:, :]  # [1, B-1, hidden]
        d_logits = jnp.einsum("bsh,hv->bsv", hs_d.astype(jnp.bfloat16), kernel.astype(jnp.bfloat16))
        if bias is not None:
            d_logits = d_logits + bias[None, None, :]
        draft_tokens = jnp.argmax(d_logits, axis=-1).astype(jnp.int32)  # [1, B-1]

        # Verify sequentially with cached teacher (early-stop; no rollback required):
        # Start from cache(prefix_without_current), process current token once to get pred0,
        # then compare draft token j with pred(j-1). Stop at first mismatch.
        cache_tmp = cache
        new_ctx_feats = []

        # Step 0: process current token -> pred0
        with teacher.mesh:
            out0 = teacher(
                input_ids=cur,
                past_key_values=cache_tmp,
                output_hidden_states=True,
                apply_lm_head=True,
                use_cache=True,
            )
        cache_tmp = out0.past_key_values
        pred = jnp.argmax(out0.logits[:, -1, :], axis=-1).astype(jnp.int32)  # [1]
        hs0 = out0.hidden_states
        if hs0 is None:
            raise RuntimeError("Teacher step missing hidden_states; need output_hidden_states=True")
        new_ctx_feats.append(
            extract_dflash_context_features_from_hidden_states(
                hidden_states=hs0,
                target_layer_ids=tli,
                add_one_for_pre_layer_capture=bool(add_one),
            )
        )

        n_acc = 0
        bonus = pred
        for j in range(int(args.block_size) - 1):
            cand_j = draft_tokens[:, j]  # [1]
            match = bool((cand_j == bonus).astype(jnp.bool_)[0])
            if not match:
                # accept_len = n_acc, bonus already set to pred after last committed token
                break

            # Accept this token and advance teacher cache by consuming it.
            n_acc += 1
            tok = cand_j[:, None].astype(jnp.int32)  # [1,1]
            with teacher.mesh:
                outj = teacher(
                    input_ids=tok,
                    past_key_values=cache_tmp,
                    output_hidden_states=True,
                    apply_lm_head=True,
                    use_cache=True,
                )
            cache_tmp = outj.past_key_values
            bonus = jnp.argmax(outj.logits[:, -1, :], axis=-1).astype(jnp.int32)
            hsj = outj.hidden_states
            if hsj is None:
                raise RuntimeError("Teacher step missing hidden_states; need output_hidden_states=True")
            new_ctx_feats.append(
                extract_dflash_context_features_from_hidden_states(
                    hidden_states=hsj,
                    target_layer_ids=tli,
                    add_one_for_pre_layer_capture=bool(add_one),
                )
            )

        # Commit: cache_tmp includes current token + accepted draft tokens; bonus is the next token.
        feat_commit = jnp.concatenate(new_ctx_feats, axis=1)  # [1, 1+n_acc, K*hidden]
        new_ctx_hidden = draft.project_context_features(feat_commit)
        ctx_kv = append_draft_ctx_kv(draft=draft, rope=rope, cache=ctx_kv, new_ctx_hidden=new_ctx_hidden)

        cache = cache_tmp
        cur = bonus[:, None].astype(jnp.int32)
        seq_len += 1 + n_acc
        accepted += int(n_acc)
        proposed += int(args.block_size) - 1

    dt = max(1e-9, time.time() - t0)
    out = {
        "mode": "cached_sequential_verify",
        "prompt_len": int(prompt_ids.shape[0]),
        "max_new_tokens": int(args.max_new_tokens),
        "block_size": int(args.block_size),
        "accepted_tokens": int(accepted),
        "proposed_tokens": int(proposed),
        "accept_rate": float(accepted) / float(max(1, proposed)),
        "wall_s": float(dt),
        "tok_s_total": float(int(args.max_new_tokens)) / float(dt),
        "target_layer_ids": tli,
        "target_layer_ids_afterlayer_mode": bool(args.use_afterlayer_ids),
    }
    if bool(args.also_run_baseline):
        # Baseline: cached greedy decode (no DFlash). This is still a correctness harness,
        # not a production sampler.
        cache_b = cache
        cur_b = cur
        t1 = time.time()
        for _ in range(int(args.max_new_tokens)):
            with teacher.mesh:
                out_b = teacher(
                    input_ids=cur_b,
                    past_key_values=cache_b,
                    output_hidden_states=False,
                    apply_lm_head=True,
                    use_cache=True,
                )
            cache_b = out_b.past_key_values
            logits = out_b.logits[:, -1, :]
            cur_b = jnp.argmax(logits, axis=-1).astype(jnp.int32)[:, None]
        dt_b = max(1e-9, time.time() - t1)
        tok_s_b = float(int(args.max_new_tokens)) / float(dt_b)
        out.update(
            {
                "baseline_wall_s": float(dt_b),
                "baseline_tok_s_total": float(tok_s_b),
                "speedup_x": float(out["tok_s_total"]) / float(max(tok_s_b, 1e-9)),
            }
        )
    print(json.dumps(out, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
