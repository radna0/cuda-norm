# 120B partial prune dry-run

- run_id: `20260113_135349`
- Model: `openai/gpt-oss-120b`
- Layer: 35
- Keep: 64/128 (0.50)
- Downloaded shards: 1 (3.83 GiB)
- Output file: `/root/data/artifacts/120b_partial_prune_dryrun/20260113_135349/layer35_keep64.safetensors` (0.79 GiB)

## Timings

- download: 10.1s
- load+slice: 0.2s
- write: 1.4s
- total: 19.2s

## Peak memory

- peak RSS: 14.03 GiB

## Reproduce

```bash
modal run modal/partial_prune_dryrun_120b_modal.py --layer 35 --keep-frac 0.5
```
