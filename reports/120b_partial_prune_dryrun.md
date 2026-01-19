# 120B partial prune dry-run

- Model: `openai/gpt-oss-120b`
- Layer: 18
- Keep: 64/128 (0.50)
- Downloaded shards: 2 (8.19 GiB)
- Output file: `/root/data/artifacts/120b_partial_prune_dryrun/20260113_133439/layer18_keep64.safetensors` (0.79 GiB)

## Timings

- download: 23.5s
- load+slice: 0.2s
- write: 7.3s
- total: 44.0s

## Peak memory

- peak RSS: 14.73 GiB

## Reproduce

```bash
modal run modal/partial_prune_dryrun_120b_modal.py --layer 0 --keep-frac 0.5
```
