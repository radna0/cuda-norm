# 120B partial prune sweep

- run_id: `20260113_135731`
- Model: `openai/gpt-oss-120b`
- layers: 3 (0..35)
- Keep: 64/128 (0.50)
- Unique shards downloaded: 3 (12.02 GiB)
- Download time: 61.4s
- Peak RSS: 14.60 GiB
- Output dir (Modal volume): `/root/data/artifacts/120b_partial_prune_sweep/20260113_135731`

## Layer outputs

| layer | shards | out_gib | load_slice_s | write_s |
|---:|---:|---:|---:|---:|
| 0 | 1 | 0.79 | 0.19 | 1.39 |
| 18 | 2 | 0.79 | 0.23 | 1.34 |
| 35 | 1 | 0.79 | 0.16 | 1.40 |

## Reproduce

```bash
modal run modal/partial_prune_sweep_120b_modal.py --layers '0,18,35' --keep-frac 0.5
```

