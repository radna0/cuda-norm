# Soft prune → structural prune policy (GPT‑OSS MoE)

This document defines how we convert an *inference-time* “soft prune” policy (expert restrictions) into a *checkpoint rewrite* (“structural prune”) plan, now that PPL parity is fixed.

## Preconditions (must hold before using PPL deltas)

- Use the parity harness rules in `reports/pruning_eval_parity.md`.
- Verify baseline is in-family for GPT‑OSS‑20B on Harmony packed blocks.
  - Current parity run: baseline PPL ≈ **2.81** on `radna0/nemotron-math-v2-harmony-tools` split `high_part00` (`reports/20b_soft_prune_eval_parity.md`).

## Soft prune knobs

Soft prune is expressed as:

- **Allowed experts per layer**: keep a per-layer subset `keep_by_layer[layer] = [expert_ids...]`.
  - Source options:
    - Domain topical ranks from `third_party/GPT-OSS-MoE-ExpertFingerprinting/topical_analytics/*.json`
    - Our measured usage histogram (`reports/20b_expert_usage_profile.md`, `data/20b_expert_usage.parquet`)
- **Routing top‑k**: reduce experts-per-token (`top_k`) where supported.

Operational constraints:

- Always enforce `top_k <= num_local_experts` for the evaluated model/checkpoint.
- Prefer keeping `top_k=4` while changing `keep_n` first; reduce `top_k` only when we explicitly want a speed/quality trade.

## Structural prune mapping

Structural prune is the checkpoint rewrite that makes the soft policy “real” by:

1. Selecting `keep_by_layer[layer]` (same list used for soft prune).
2. Rewriting MoE expert tensors to only include the kept experts (new leading dim).
3. Updating model config:
   - `num_local_experts = keep_n`
   - `num_experts_per_tok = min(old, keep_n)` (and `experts_per_token` alias)
4. Producing an `expert_mapping.json`:
   - maps old expert ids → new contiguous ids `[0..keep_n-1]` per layer.

Notes:

- For GPT‑OSS, Transformers loads all tensors present in each referenced shard file. A correct structural prune must ensure referenced shard files do **not** contain leftover (shape-mismatched) 32‑expert tensors.
- Our 20B structural prune implementation solves this by writing:
  - `pruned_layer_{i}.safetensors` (only router+experts keys for layer i, sliced)
  - `base_model-*.safetensors` (all other keys, excluding pruned keys)

## Recommended workflow

1. **Profile experts** for the target domain(s) (usage histogram + confidence stats).
2. **Soft prune sweep** on parity PPL:
   - grid over `(keep_frac, top_k)` and/or domain-specific `keep_by_layer`.
3. Pick candidate policies by:
   - PPL delta bound (quality)
   - tokens/s improvement (speed)
4. **Structural prune build** for the chosen policies.
5. Re-run:
   - parity PPL (same harness) on structurally-pruned checkpoints
   - smoke inference + router stats (`reports/20b_structural_prune_smoke.md`)

