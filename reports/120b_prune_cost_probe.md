# 120B pruning cost probe (Modal profile `locthaokien1201`)

This is a **cost probe** for structural MoE expert pruning on GPT‑OSS‑120B. It is **not** a full prune run.

## What was run

- Script: `harmony/cuda-norm/modal/partial_prune_dryrun_120b_modal.py`
- Model: `openai/gpt-oss-120b`
- Probe: layer `0`, keep_frac `0.50` (kept `64/128` experts)
- Log: `harmony/cuda-norm/unsloth_logs/120b_partial_prune_dryrun_20260113_123326.log`
- Output (Modal volume): `/root/data/artifacts/120b_partial_prune_dryrun/20260113_123339/layer0_keep64.safetensors`

Modal run link:
- https://modal.com/apps/locthaokien1201/main/ap-m4BILoDam5a1Dp5fqfEcdZ

## Observed measurements (this probe)

### Summary table (keep_frac=0.5, keep_n=64)

| layer | shards | shard_files | download_gib | download_s | write_s | total_s |
|---:|---:|---|---:|---:|---:|---:|
| 0 | 1 | `model-00009-of-00014` | 3.88 | 12.45 | 2.50 | 33.34 |
| 9 | 2 | `model-00006-of-00014`, `model-00007-of-00014` | 8.09 | 38.98 | 1.49 | 54.92 |
| 18 | 2 | `model-00008-of-00014`, `model-00009-of-00014` | 8.19 | 23.51 | 7.34 | 43.97 |
| 27 | 2 | `model-00000-of-00014`, `model-00001-of-00014` | 8.14 | 22.03 | 1.65 | 30.77 |
| 35 | 1 | `model-00014-of-00014` | 3.83 | 10.13 | 1.41 | 19.18 |

Notes:
- Shard count varies by layer (1–2 shards in these probes).
- Download time varies a lot due to cache state and transient HF/network conditions; shard_files is the stable “locality” signal.

### Layer 0 (keep_frac=0.5)

From `[RESULT]` in `harmony/cuda-norm/unsloth_logs/120b_partial_prune_dryrun_20260113_123326.log`:

- Downloaded shards: `1`
  - `model-00009-of-00014.safetensors` (downloaded `3.88 GiB` total)
- Timings:
  - download: `12.45 s`  (≈ `0.31 GiB/s`)
  - load+slice: `0.19 s`
  - write: `2.50 s` (wrote `0.79 GiB`, ≈ `0.32 GiB/s` to Modal volume)
  - total: `33.34 s`
- Peak RSS: `13.81 GiB`

### Layer 18 (keep_frac=0.5)

From `[RESULT]` in `harmony/cuda-norm/unsloth_logs/120b_partial_prune_dryrun_layer18_20260113_133436.log`:

- Downloaded shards: `2`
  - `model-00008-of-00014.safetensors`, `model-00009-of-00014.safetensors` (downloaded `8.19 GiB` total)
- Timings:
  - download: `23.51 s`  (≈ `0.35 GiB/s`)
  - load+slice: `0.23 s`
  - write: `7.34 s` (wrote `0.79 GiB`, ≈ `0.11 GiB/s` to Modal volume)
  - total: `43.97 s`
- Peak RSS: `14.73 GiB`

## Checkpoint sizing context (HF listing)

For `openai/gpt-oss-120b` the “main” sharded checkpoint used by our pruning scripts is:

- `15` files matching `model-00000-of-00014.safetensors` … `model-00014-of-00014.safetensors`
- Total size: `60.77 GiB`

Note: the repo also contains `original/` shards; we are **not** using those in this pruning path.

## Extrapolation (order-of-magnitude)

This extrapolation assumes:

1. A full structural prune rewrite will need the full `model-*.safetensors` set available locally at least once (downloaded into the HF cache volume).
2. Per-layer “MoE MLP” tensor size is roughly similar across layers (using our layer‑0 output file as a proxy).

Shard locality matters:

- Across sampled layers (0/9/18/27/35), MoE tensors needed **1–2 shards per layer** in this checkpoint layout, not a single shard globally.
- The shard_files are not monotonic in layer index (e.g., layer 27 hit `model-00000/00001`, layer 35 hit `model-00014`), so a “full sweep” will effectively touch most or all shards even if any one layer is local.

Estimated first-run download time (cold cache, full `model-*.safetensors` set):

- `60.77 GiB` / `0.31 GiB/s` ≈ **~3.3 minutes**

Estimated pruned MoE tensor output size (keep_frac=0.5):

- Observed per-layer (kept 64 experts): `~0.789 GiB`
- `0.789 GiB * 36 layers` ≈ **~28.4 GiB** of MoE tensors written

Estimated write time for pruned MoE tensors (to volume), if throughput holds:

- `28.4 GiB` / `0.32 GiB/s` ≈ **~1.5 minutes**

However, the layer‑18 probe saw a slower effective write rate (`~0.11 GiB/s`). Using that as a pessimistic bound:

- `28.4 GiB` / `0.11 GiB/s` ≈ **~4.3 minutes**

Implication:

- **Full prune wall time is dominated by I/O**, not slicing compute. With cache warm, the expensive part becomes “read+rewrite” on disk/volume.
- For a “full layer sweep” of MoE tensors, prefer a **shard-grouped sweep** (download each shard once, then slice all layers that live inside it) rather than calling the single-layer dryrun 36 times.
  - Implemented: `harmony/cuda-norm/modal/partial_prune_sweep_120b_modal.py` (writes per-layer `layer{L}_keep{K}.safetensors` into the `pruning-data` Modal volume).

## Measured end-to-end runs (real numbers)

### Full MoE tensor sweep (layers 0–35, keep_frac=0.5)

- Log: `harmony/cuda-norm/unsloth_logs/120b_partial_prune_sweep_full_20260113_140713.log`
- Outputs:
  - `harmony/cuda-norm/reports/120b_partial_prune_sweep_20260113_140715.md`
  - `harmony/cuda-norm/reports/120b_partial_prune_sweep_20260113_140715.json`
- Result summary:
  - unique_shards_downloaded: 15 (download_gib≈60.77)
  - download_s≈173.82
  - total_s≈290.95
  - sum_output_gib≈28.41 (36 × ~0.789 GiB per layer)

### Full structural checkpoint build baseline (first64, keep_frac=0.5)

This is the *actual* “rewrite a loadable pruned checkpoint” workload (base shards + pruned_layer shards + new index + updated config).

- Log: `harmony/cuda-norm/unsloth_logs/120b_structural_prune_build_first64_retry2_20260113_143647.log`
- Output model_dir (Modal volume): `/root/model/artifacts/120b_pruned_models/first64_experts_keepfrac50`
- Report: `harmony/cuda-norm/reports/120b_structural_prune_build_first64.md`
- Validation artifact: `harmony/cuda-norm/artifacts/120b_pruned_models_first64/manifest.json` (mismatch_count=0)
- Observed build time: dt_s≈656.2

## Reproduce (logged)

```bash
mkdir -p harmony/cuda-norm/unsloth_logs
ts=$(date +%Y%m%d_%H%M%S)
nohup env MODAL_PROFILE=locthaokien1201 \
  modal run harmony/cuda-norm/modal/partial_prune_dryrun_120b_modal.py --layer 0 --keep-frac 0.5 \
  > "harmony/cuda-norm/unsloth_logs/120b_partial_prune_dryrun_${ts}.log" 2>&1 &
```

## Next probe (recommended)

To de-risk extrapolation further, repeat for a couple more layers (e.g. `--layer 9`, `--layer 27`, `--layer 35`) and confirm:

- how many shards are needed per layer
- whether per-layer output sizes are consistent

Status:
- completed for `--layer 9` (2 shards), `--layer 18` (2 shards), `--layer 27` (2 shards), `--layer 35` (1 shard); output size stayed ~0.79 GiB across all runs
