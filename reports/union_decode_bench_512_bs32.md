# 20B decode throughput: UNION prunes

- prompt_len: 256 | new_tokens: 512
- batch_sweep: 32

| run | top_k | max_batch | total tok/s @max | per-stream tok/s @max | mem_used_gib @max | model |
|---|---:|---:|---:|---:|---:|---|
| base_topk4 | 4 | 32 | 581.26 | 18.16 | 18.7 | `openai/gpt-oss-20b` |
| base_topk2 | 2 | 32 | 596.12 | 18.63 | 28.3 | `openai/gpt-oss-20b` |
| union50_topk4 | 4 | 32 | 630.92 | 19.72 | 27.5 | `/root/model/artifacts/20b_union_pruned/union50` |
| union50_topk2 | 2 | 32 | 634.01 | 19.81 | 28.5 | `/root/model/artifacts/20b_union_pruned/union50` |
| unionAgg_topk2 | 2 | 32 | 627.57 | 19.61 | 18.7 | `/root/model/artifacts/20b_union_pruned/unionAgg` |

## Per-batch details

### base_topk4

| batch | total tok/s | per-stream tok/s | decode_s | mem_used_gib | status |
|---:|---:|---:|---:|---:|---|
| 32 | 581.26 | 18.16 | 28.187 | 18.66680908203125 | ok |

### base_topk2

| batch | total tok/s | per-stream tok/s | decode_s | mem_used_gib | status |
|---:|---:|---:|---:|---:|---|
| 32 | 596.12 | 18.63 | 27.484 | 28.29180908203125 | ok |

### union50_topk4

| batch | total tok/s | per-stream tok/s | decode_s | mem_used_gib | status |
|---:|---:|---:|---:|---:|---|
| 32 | 630.92 | 19.72 | 25.968 | 27.53594970703125 | ok |

### union50_topk2

| batch | total tok/s | per-stream tok/s | decode_s | mem_used_gib | status |
|---:|---:|---:|---:|---:|---|
| 32 | 634.01 | 19.81 | 25.842 | 28.54376220703125 | ok |

### unionAgg_topk2

| batch | total tok/s | per-stream tok/s | decode_s | mem_used_gib | status |
|---:|---:|---:|---:|---:|---|
| 32 | 627.57 | 19.61 | 26.107 | 18.74298095703125 | ok |

## Reproduce

```bash
modal run modal/benchmark_decode_throughput_union.py --prompt-len 256 --new-tokens 2048 --batch-sizes 1,2,4,8,16,32
```
