#!/usr/bin/env python3
"""TPU DFlash spec-v1 decode with block-parallel verify (TARGET_VERIFY-style).

This is the first TPU decode harness that matches DFlash’s *core* scaling trick:
verify a whole draft block with a single target forward, then crop/commit KV.

Key properties:
- Target (teacher) runs with KV-cache (past_key_values).
- Verify runs as ONE teacher forward on (anchor + draft_tokens[0..B-2]) producing:
    target_predict[i] = argmax(logits[i]) for i in [0..B-1]
  and uses the SGLang accept rule:
    accept while candidates[:, 1:] == target_predict[:, :-1]
    bonus = target_predict[:, accept_len]
- If the teacher cache supports `.crop(seq_len)`, we crop back to the committed
  length; otherwise we fall back to a conservative sequential verify path.
- Draft uses per-layer ctx KV caching to avoid O(ctx_len) k/v projection.

Limitations (initial, but deliberate):
- batch_size=1 only
- greedy verification only (temp=0)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import sys

import numpy as np


def _encode_prompt(tokenizer_path: str, text: str, *, max_len: int) -> np.ndarray:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, use_fast=True)
    out = tok(text, truncation=True, max_length=int(max_len), return_tensors="np")
    return out["input_ids"].astype(np.int32)[0]


def _cache_seq_len(cache) -> int:
    # Prefer cache-native methods if available.
    if hasattr(cache, "get_seq_length"):
        try:
            return int(cache.get_seq_length())
        except Exception:
            pass
    if hasattr(cache, "get_seq_len"):
        try:
            return int(cache.get_seq_len())
        except Exception:
            pass

    # Fall back to reading the first leaf array we can find.
    import jax

    leaves = jax.tree_util.tree_leaves(cache)
    for x in leaves:
        if hasattr(x, "shape") and getattr(x, "ndim", 0) >= 3:
            # Common layouts: [B, H, S, D] (seq axis=-2) or [B, S, H, D] (seq axis=1)
            s1 = int(x.shape[1]) if int(x.ndim) >= 2 else 0
            sm2 = int(x.shape[-2]) if int(x.ndim) >= 2 else 0
            return max(s1, sm2)
    raise RuntimeError("Unable to infer cache seq_len; cache has no array leaves?")


def _crop_cache(cache, *, new_seq_len: int):
    # The best path is a native crop method.
    if hasattr(cache, "crop"):
        cache.crop(int(new_seq_len))
        return cache

    # Heuristic crop for tuple/list/dict pytree caches.
    import jax
    import jax.numpy as jnp

    def _crop_leaf(x):
        if not hasattr(x, "shape") or getattr(x, "ndim", 0) < 3:
            return x
        x = jnp.asarray(x)
        # Prefer cropping along axis=-2 if it looks like sequence.
        if int(x.shape[-2]) >= int(new_seq_len) and int(x.shape[-2]) >= int(x.shape[1]):
            return x[..., : int(new_seq_len), :]
        if int(x.shape[1]) >= int(new_seq_len):
            return x[:, : int(new_seq_len), ...]
        return x

    return jax.tree_util.tree_map(_crop_leaf, cache)


def _slice_hidden_states(hidden_states, *, keep_first: int, keep_count: int):
    # hidden_states is a tuple/list of arrays; slice the seq dimension.
    import jax.numpy as jnp

    hs = list(hidden_states)
    out = []
    for x in hs:
        x = jnp.asarray(x)
        if x.ndim < 3:
            out.append(x)
            continue
        out.append(x[:, int(keep_first) : int(keep_first + keep_count), :])
    return tuple(out)


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

    # Prefer the local EasyDeL checkout (we are modifying EasyDeL source directly).
    repo_root = Path(__file__).resolve().parents[1]
    local_easydel = repo_root / "external" / "EasyDeL"
    if local_easydel.exists():
        sys.path.insert(0, str(local_easydel))

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

    # Load teacher embedding + lm_head (teacher snapshot must include them).
    # We keep this logic minimal; the trainer already produces a compatible snapshot layout.
    teacher_cfg = json.loads((teacher_snapshot / "config.json").read_text(encoding="utf-8"))
    tokenizer_path = str(teacher_snapshot)

    # Build RoPE using the same TPU helpers used by the other harnesses.
    try:
        from tpu_dflash_lib import build_rope, load_json

        rope = build_rope(cfg=load_json(teacher_snapshot / "config.json"), dtype=jnp.bfloat16)
    except Exception as e:
        raise RuntimeError("Failed to build RoPE for draft.") from e

    # ---- Target layer IDs for context features
    if str(args.target_layer_ids).strip():
        tli = [int(x) for x in str(args.target_layer_ids).split(",") if x.strip()]
    else:
        # Default: evenly spaced after-layer IDs (like reference); keep small K for speed.
        n_layers = int(teacher_cfg.get("num_hidden_layers", 1))
        k = int(dcfg.num_context_features)
        tli = [int(round(i * (n_layers - 1) / max(k - 1, 1))) for i in range(k)]
    add_one = bool(args.use_afterlayer_ids)

    prompt_ids = _encode_prompt(tokenizer_path, args.prompt, max_len=int(args.max_prompt_len))
    if int(prompt_ids.shape[0]) < 2:
        raise ValueError("Prompt too short; need at least 2 tokens")

    prefix_ids = prompt_ids[:-1]
    current_id = prompt_ids[-1:]
    prefix = jnp.asarray(prefix_ids[None, :], dtype=jnp.int32)
    cur = jnp.asarray(current_id[None, :], dtype=jnp.int32)  # [1,1]

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

    hs_prefill = out_prefill.hidden_states
    if hs_prefill is None:
        raise RuntimeError("Teacher did not return hidden_states (output_hidden_states=True).")

    # Build initial context features for prefix tokens.
    ctx_feats = extract_dflash_context_features_from_hidden_states(
        hidden_states=hs_prefill,
        target_layer_ids=tli,
        add_one_for_pre_layer_capture=bool(add_one),
    )  # [1, S, K*hidden]
    ctx_hidden = draft.project_context_features(ctx_feats)  # [1, S, hidden]
    ctx_kv = materialize_draft_ctx_kv(
        draft=draft,
        rope=rope,
        ctx_hidden=ctx_hidden,
        max_len=int(args.max_model_len + int(args.block_size)),
    )

    # Teacher embedding + lm_head for draft tokenization.
    lm_head = teacher.get_lm_head()
    emb_fn = teacher.get_embedding()
    kernel = jax.lax.stop_gradient(lm_head.kernel.value)
    bias = jax.lax.stop_gradient(lm_head.bias.value) if getattr(lm_head, "bias", None) is not None else None

    accepted = 0
    proposed = 0

    t0 = time.time()
    for _ in range(int(args.max_new_tokens)):
        # Draft proposal for block_size-1 tokens.
        with teacher.mesh:
            anchor_emb = emb_fn(cur.astype("i4"))[:, 0, :]  # [1,hidden]
        d_hidden = draft_forward_with_ctx_kv(
            draft=draft,
            rope=rope,
            cache=ctx_kv,
            anchor_embedding=anchor_emb.astype(jnp.bfloat16),
            mask_embedding=draft.mask_embedding.value.astype(jnp.bfloat16),
            block_size=int(args.block_size),
        )
        # Apply teacher lm_head to draft hidden to propose tokens.
        hs_d = d_hidden[:, 1:, :]  # [1, B-1, hidden]
        d_logits = jnp.einsum("bsh,hv->bsv", hs_d.astype(jnp.bfloat16), kernel.astype(jnp.bfloat16))
        if bias is not None:
            d_logits = d_logits + bias[None, None, :]
        draft_tokens = jnp.argmax(d_logits, axis=-1).astype(jnp.int32)  # [1, B-1]

        # Prefer the core DFlash path: block-parallel verify + cache crop.
        # If the cache cannot be cropped safely, fall back to sequential verify.
        try:
            cand = jnp.concatenate([cur, draft_tokens], axis=1)  # [1, B]
            base_len = _cache_seq_len(cache)
            with teacher.mesh:
                out_v = teacher(
                    input_ids=cand,
                    past_key_values=cache,
                    output_hidden_states=True,
                    apply_lm_head=True,
                    use_cache=True,
                )
            cache_full = out_v.past_key_values
            if cache_full is None:
                raise RuntimeError("Teacher verify forward missing past_key_values")
            logits = out_v.logits  # [1, B, V]
            target_predict = jnp.argmax(logits, axis=-1).astype(jnp.int32)  # [1, B]
            accept_len, bonus = dflash_accept_len_and_bonus(candidates=cand, target_predict=target_predict)
            n_acc = int(accept_len[0])
            keep_in_block = 1 + n_acc
            cache_try = _crop_cache(cache_full, new_seq_len=int(base_len + keep_in_block))
            if _cache_seq_len(cache_try) != int(base_len + keep_in_block):
                raise RuntimeError("Cache crop did not take effect (seq_len mismatch).")
            cache = cache_try
            cur = bonus.astype(jnp.int32)[:, None]

            hs_v = out_v.hidden_states
            if hs_v is None:
                raise RuntimeError("Teacher verify forward missing hidden_states")
            seq_dim = int(hs_v[0].shape[1])
            start = 0 if seq_dim == int(args.block_size) else seq_dim - int(args.block_size)
            hs_commit = _slice_hidden_states(hs_v, keep_first=start, keep_count=keep_in_block)
            feat_commit = extract_dflash_context_features_from_hidden_states(
                hidden_states=hs_commit,
                target_layer_ids=tli,
                add_one_for_pre_layer_capture=bool(add_one),
            )
            new_ctx_hidden = draft.project_context_features(feat_commit)
            ctx_kv = append_draft_ctx_kv(draft=draft, rope=rope, cache=ctx_kv, new_ctx_hidden=new_ctx_hidden)

            accepted += n_acc
            proposed += int(args.block_size) - 1
            continue
        except Exception:
            # Conservative sequential verify (works even when cache cannot crop).
            cache_tmp = cache
            new_ctx_feats = []

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
                cand_j = draft_tokens[:, j]
                if not bool((cand_j == bonus).astype(jnp.bool_)[0]):
                    break
                n_acc += 1
                tok = cand_j[:, None].astype(jnp.int32)
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

            feat_commit = jnp.concatenate(new_ctx_feats, axis=1)  # [1, 1+n_acc, K*hidden]
            new_ctx_hidden = draft.project_context_features(feat_commit)
            ctx_kv = append_draft_ctx_kv(draft=draft, rope=rope, cache=ctx_kv, new_ctx_hidden=new_ctx_hidden)
            cache = cache_tmp
            cur = bonus[:, None].astype(jnp.int32)

            accepted += n_acc
            proposed += int(args.block_size) - 1

    dt = max(1e-9, time.time() - t0)
    out = {
        "mode": "cached_block_verify_or_fallback",
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
