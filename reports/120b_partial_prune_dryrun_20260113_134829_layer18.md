# 120B partial prune dry-run

- run_id: `20260113_134829`
- Model: `openai/gpt-oss-120b`
- Layer: 18
- Keep: 64/128 (0.50)
- Downloaded shards: 2 (8.19 GiB)
- Output file: `/root/data/artifacts/120b_partial_prune_dryrun/20260113_134829/layer18_keep64.safetensors` (0.79 GiB)

## Timings

- download: 39.3s
- load+slice: 0.2s
- write: 1.6s
- total: 45.2s

## Peak memory

- peak RSS: 14.72 GiB

## Reproduce

```bash
modal run modal/partial_prune_dryrun_120b_modal.py --layer 18 --keep-frac 0.5
```
