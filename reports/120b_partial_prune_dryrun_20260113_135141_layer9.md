# 120B partial prune dry-run

- run_id: `20260113_135141`
- Model: `openai/gpt-oss-120b`
- Layer: 9
- Keep: 64/128 (0.50)
- Downloaded shards: 2 (8.09 GiB)
- Output file: `/root/data/artifacts/120b_partial_prune_dryrun/20260113_135141/layer9_keep64.safetensors` (0.79 GiB)

## Timings

- download: 39.0s
- load+slice: 0.2s
- write: 1.5s
- total: 54.9s

## Peak memory

- peak RSS: 15.15 GiB

## Reproduce

```bash
modal run modal/partial_prune_dryrun_120b_modal.py --layer 9 --keep-frac 0.5
```
