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

