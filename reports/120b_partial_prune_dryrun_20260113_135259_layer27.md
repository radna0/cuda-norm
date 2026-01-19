# 120B partial prune dry-run

- run_id: `20260113_135259`
- Model: `openai/gpt-oss-120b`
- Layer: 27
- Keep: 64/128 (0.50)
- Downloaded shards: 2 (8.14 GiB)
- Output file: `/root/data/artifacts/120b_partial_prune_dryrun/20260113_135259/layer27_keep64.safetensors` (0.79 GiB)

## Timings

- download: 22.0s
- load+slice: 0.2s
- write: 1.6s
- total: 30.8s

## Peak memory

- peak RSS: 14.77 GiB

## Reproduce

```bash
modal run modal/partial_prune_dryrun_120b_modal.py --layer 27 --keep-frac 0.5
```
