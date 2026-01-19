# 120B Structural Prune Cost Probe (Kaggle / Versa plan)

Goal: measure whether a 120B structural prune is primarily an I/O job (and how big), without running any training.

## What we measure

For a single MoE layer (e.g. layer 0) and keep_frac=0.50:
- bytes read (weights needed for that layer + related router tensors)
- bytes written (new pruned tensors + rewritten base shards)
- wall time
- peak RSS

Then extrapolate:
- expected full-prune wall time (linear in bytes + shard count effects)
- expected disk footprint during rewrite (need headroom for output shards)

## Why Kaggle is acceptable for the probe

The cost probe is dominated by:
- safetensors reads/writes
- CPU tensor slicing/concat

It does not require a B200. A GPU is optional unless we want to validate load/inference.

## Implementation path in this repo

Use the existing pruning utilities:
- `harmony/cuda-norm/pruning/partial_prune_dryrun_120b.py`
- `harmony/cuda-norm/pruning/gpt_oss_moe_cost.py`

If these scripts assume Modal volumes, run them under Versa with:
- `HF_HOME=/kaggle/working/hf_cache`
- output under `/kaggle/working/artifacts/120b_cost_probe/...`

## Execution checklist

1. Pre-download 120B weights on CPU (no GPU time)
   - `snapshot_download` to `/kaggle/working/hf_cache`
2. Run a one-layer prune rewrite (keep_frac=0.50) and record timings
3. Validate that the produced partial checkpoint can be loaded (optional)
4. Write:
   - `reports/120b_partial_prune_dryrun.md`
   - `reports/120b_prune_cost_probe.md` (extrapolation)

## Guardrails

- Do not attempt full-prune in Kaggle unless the probe shows sufficient disk headroom.
- Prefer probing a couple layers to capture shard locality variance (1-shard vs multi-shard layers).

