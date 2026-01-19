# 20B decode throughput (prefill vs decode)

- prompt_len: 1024 | new_tokens: 64 | batch_size: 1

## soft_top_k=0 (no override)

| model | prefill tok/s | decode tok/s | prefill_s | decode_s | peak_alloc_gib | experts | cfg_top_k | applied_top_k | path |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base | 8033 | 18.19 | 0.127 | 3.518 | 13.7 | 32 | 4 | 4 | `openai/gpt-oss-20b` |
| general_50pct_experts | 11566 | 18.09 | 0.089 | 3.539 | 9.0 | 16 | 4 | 4 | `/root/model/artifacts/20b_pruned_models/general_50pct_experts` |
| math_25pct_experts | 11214 | 17.94 | 0.091 | 3.567 | 11.4 | 8 | 4 | 4 | `/root/model/artifacts/20b_pruned_models/math_25pct_experts` |

Deltas vs base:

- general_50pct_experts: decode_tok/s delta=-0.11 (-0.6%)
- math_25pct_experts: decode_tok/s delta=-0.25 (-1.4%)

## soft_top_k=2

| model | prefill tok/s | decode tok/s | prefill_s | decode_s | peak_alloc_gib | experts | cfg_top_k | applied_top_k | path |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base | 11618 | 18.52 | 0.088 | 3.455 | 13.7 | 32 | 4 | 2 | `openai/gpt-oss-20b` |
| general_50pct_experts | 11325 | 17.94 | 0.090 | 3.568 | 9.0 | 16 | 4 | 2 | `/root/model/artifacts/20b_pruned_models/general_50pct_experts` |
| math_25pct_experts | 10288 | 17.48 | 0.100 | 3.662 | 11.4 | 8 | 4 | 2 | `/root/model/artifacts/20b_pruned_models/math_25pct_experts` |

Deltas vs base:

- general_50pct_experts: decode_tok/s delta=-0.59 (-3.2%)
- math_25pct_experts: decode_tok/s delta=-1.05 (-5.6%)

## Reproduce

```bash
modal run modal/benchmark_decode_throughput_20b.py --prompt-len 1024 --new-tokens 64 --batch-size 1 --soft-top-k-values 0,2
```
