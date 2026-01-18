# REAP Recipes (Pruning + Testing)

This folder implements the “REAP recipe + REAP scores” idea (see GH issue #2) for our GPT‑OSS MoE pruning track.

The intent:
- **Reproducibility**: a pruned checkpoint should ship with a `reap-recipe.yaml` (how it was produced).
- **Reusability**: the ranking/scores used for pruning should ship with a `reap-scores.yaml` (so others can re‑cut keep_frac without re‑profiling).
- **Comparability**: evaluation runs should be described by `eval-*.yaml` and should inherit from a shared base config rather than copy‑pasting knobs.

We use two recipe types:
- `reap-recipe.yaml` (pruning recipe): model + dataset + scoring + constraints + output.
- `reap-scores.yaml` (scores snapshot): per-layer expert scores + ranking + metadata.

Evaluation recipes:
- `eval-base.yaml` defines packs + EAFT gate file + default “bigblocks” token budget regime.
- `eval-*.yaml` extends the base and specifies models + seq/batch/token budget.

## Conventions

- **No finetune**: these recipes are **calibrate+rewrite only**.
- **top_k stays fixed** for the “near-lossless pruning” goal (top_k=4 for GPT‑OSS).
- **Uniform keep_n**: GPT‑OSS requires the same number of experts per layer.
- **Token budget**: for long-seq ablations we keep per-pack token count ~constant:
  - `num_blocks ≈ target_tokens / seq_len`
  - e.g. `seq=4096` uses `num_blocks=256` to match `~1M tokens` used at `seq=2048, blocks=512`.

## Execution backends

Recipes include an `execution` section with suggested commands for:
- **Modal** (`modal run ...`) when Modal budgets are available.
- **Kaggle/VERSA** (`scripts/versa_run_*`) when Modal is blocked or for cheaper iteration.

The YAML is intentionally tool-agnostic; it’s a single source of truth for what to run and what outputs to expect.

## Current recipes

**Pruning recipes**
- `reap-recipe_20b_eaftreap_keepfrac075_keep24_uniform.yaml`
- `reap-recipe_20b_eaftreap_keepfrac060_keep20.yaml` (0.60 target, 0.625 actual due to keep_n%4 constraint)
- `reap-recipe_120b_eaftreap_keepfrac075_planned.yaml` (planned)

**Scores snapshots**
- `reap-scores_20b_eaftreap_keepfrac075_keep24_uniform.yaml`
- `reap-scores_20b_eaftreap_keepfrac075_budgeted.yaml` (historical)
- `reap-scores_20b_eaftreap_keepfrac060_keep19.yaml`
  - historical: the old “floor(0.60*32)=19” target; we now treat 0.60 as keep_n=20 (0.625) for kernel compatibility.

**Eval recipes**
- `eval-20b_base_vs_base_noise_bigblocks.yaml`
- `eval-20b_noop_rewrite_lossless_bigblocks.yaml`
- `eval-20b_eaftreap075_bigblocks_keep24_uniform.yaml`
- `eval-20b_eaftreap075_longseq_tokenbudget1m_keep24_uniform.yaml`
- `eval-20b_eaftreap060_bigblocks_keep20_uniform.yaml`
- `eval-20b_eaftreap060_longseq_tokenbudget1m_keep20_uniform.yaml`
- `eval-20b_eaftreap075_extreme_ctx65k_131k.yaml` (production stress)
- `eval-120b_base_bigblocks.yaml`
- `eval-120b_base_longseq_tokenbudget1m.yaml`

**Cost probes**
- `costprobe-120b_partial_prune_layer0_keepfrac075.yaml`
