# 20B decode throughput: UNION prunes

- prompt_len: 256 | new_tokens: 8192
- batch_sweep: 32
- variants: base_topk4,base_topk2,union50_topk2,unionAgg_topk2

| run | top_k | max_batch | total tok/s @max | per-stream tok/s @max | mem_used_gib @max | model |
|---|---:|---:|---:|---:|---:|---|
| base_topk4 | 4 | 32 | 452.03 | 14.13 | 76.1 | `openai/gpt-oss-20b` |
| base_topk2 | 2 | 32 | 472.03 | 14.75 | 70.0 | `openai/gpt-oss-20b` |
| union50_topk2 | 2 | 32 | 460.45 | 14.39 | 75.8 | `/root/model/artifacts/20b_union_pruned/union50` |
| unionAgg_topk2 | 2 | 32 | 468.72 | 14.65 | 78.1 | `/root/model/artifacts/20b_union_pruned/unionAgg` |

## Per-batch details

### base_topk4

| batch | total tok/s | per-stream tok/s | decode_s | mem_used_gib | status |
|---:|---:|---:|---:|---:|---|
| 32 | 452.03 | 14.13 | 579.930 | 76.11993408203125 | ok |

### base_topk2

| batch | total tok/s | per-stream tok/s | decode_s | mem_used_gib | status |
|---:|---:|---:|---:|---:|---|
| 32 | 472.03 | 14.75 | 555.355 | 69.98321533203125 | ok |

### union50_topk2

| batch | total tok/s | per-stream tok/s | decode_s | mem_used_gib | status |
|---:|---:|---:|---:|---:|---|
| 32 | 460.45 | 14.39 | 569.317 | 75.75469970703125 | ok |

### unionAgg_topk2

| batch | total tok/s | per-stream tok/s | decode_s | mem_used_gib | status |
|---:|---:|---:|---:|---:|---|
| 32 | 468.72 | 14.65 | 559.282 | 78.11016845703125 | ok |

## Reproduce

```bash
modal run modal/benchmark_decode_throughput_union.py --prompt-len 256 --new-tokens 8192 --batch-sizes 32 --variants-csv base_topk4,base_topk2,union50_topk2,unionAgg_topk2
```
