# DFlash spec‑v1 (from SGLang PR #16818) → EasyDeL port notes

This document is the **algorithmic contract** we implement in EasyDeL/JAX (TPU). It is derived from the upstream SGLang implementation added in PR `sgl-project/sglang#16818`.

## Objects / terminology

- **Target (teacher)**: the large causal LM we want to accelerate (GPT‑OSS‑20B/120B).
- **Draft**: a small “DFLASH draft model” that predicts a **block** of tokens at once.
- **Block size** `B`: number of tokens verified per speculative iteration (`speculative_num_draft_tokens` in SGLang; `dflash_config.block_size` in the draft ckpt).
- **K context features**: number of intermediate target-layer hidden states concatenated per token. In SGLang, `K = len(dflash_config.target_layer_ids)` when provided.
- **Mask token**: string `dflash_config.mask_token` resolved to a token id in the target tokenizer. Some implementations fall back to `"<|MASK|>"`.

## Core rule: accept length + bonus token (greedy verification)

Given a draft block of proposed tokens and the target model’s greedy predictions over that block:

- `candidates`: shape `[bs, B]` proposed by draft (includes the *current* token at index 0).
- `target_predict`: shape `[bs, B]` where `target_predict[:, t]` is argmax token predicted by target for position `t` in the verify window.

Accept while consecutive tokens match under the DFlash rule:

```
matches = candidates[:, 1:] == target_predict[:, :-1]
accept_len = cumprod(matches).sum(axis=1)     # int32, number of accepted draft tokens (excluding current token)
bonus = target_predict[range(bs), accept_len] # the next token to append after accept_len (always appended)
```

This is implemented in SGLang as `compute_dflash_accept_len_and_bonus(...)`.

## Draft step semantics (spec‑v1, non‑overlap)

Per decoding iteration, for each request:

1. Start with a **verified current token** `verified_id` (1 token per request).
2. Create a draft input block of length `B`:
   - position 0 uses the real embedding of `verified_id`
   - positions `1..B-1` use the **mask token embedding** (or a learned `mask_embedding` vector if present in the draft checkpoint)
3. Draft model forward is **non‑causal / encoder‑only over the B-token block**, while conditioning on target context features projected via `fc + hidden_norm`.
4. Convert draft hidden states to logits using the **target** `lm_head`, then sample **greedy** to get `candidates[:, 1:]` (SGLang uses greedy only in v1).

## Verify step semantics (TARGET_VERIFY)

Target runs a verify forward over the `B` tokens (prefix + verify window) to produce:

- `target_predict`: greedy argmax tokens for each position in the window
- **aux hidden states** for the target layers listed in `target_layer_ids` (K features per token), which become the context features for the next draft step(s)

SGLang uses `CaptureHiddenMode.FULL` and `ForwardMode.TARGET_VERIFY` to collect these hidden features during verify.

## Key invariants we must preserve in EasyDeL

1. **Accept rule** exactly matches the formula above (bit-exact for integer tokens).
2. **K features contract**:
   - draft `fc.in_features == K * hidden_size`
   - the verify forward produces K hidden vectors per token (same ordering as `target_layer_ids`)
3. **Masking/positions**: the draft window is treated as encoder-only; verify window is causal.
4. **HF compatibility**: the trained draft checkpoint must be exportable as:
   - `config.json` + `model.safetensors(.index.json)` for GPU runtimes (SGLang)
   - and still loadable by TPU/JAX training/inference (EasyDeL), even if we keep a TPU-native checkpoint format as well.

## TPU correctness harness scripts

- Block-verify spec-v1 decode (matches accept_len+bonus rule):
  - `harmony/cuda-norm/scripts/tpu_dflash_spec_decode_v1.py`
- Logged runner:
  - `harmony/cuda-norm/scripts/run_tpu_dflash_decode_logged.sh`

