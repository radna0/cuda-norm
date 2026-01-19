# 20B decode throughput: REAP vs frequency

- prompt_len: 256 | new_tokens: 2048
- batch_sweep: 32

| run | top_k | max_batch | total tok/s @max | per-stream tok/s @max | mem_used_gib @max | model |
|---|---:|---:|---:|---:|---:|---|
| base_topk4 | 4 | 32 | 544.04 | 17.00 | 29.5 | `openai/gpt-oss-20b` |
| base_topk2 | 2 | 32 | 561.28 | 17.54 | 39.5 | `openai/gpt-oss-20b` |
| freq50_topk4 | 4 | 32 | 562.32 | 17.57 | 28.7 | `/root/model/artifacts/20b_pruned_models_freq/general_50pct_experts_freq` |
| reap50_topk4 | 4 | 32 | 573.51 | 17.92 | 20.7 | `/root/model/artifacts/20b_pruned_models_reap/general_50pct_experts_reap` |
| reap50_topk2 | 2 | 32 | 591.61 | 18.49 | 25.4 | `/root/model/artifacts/20b_pruned_models_reap/general_50pct_experts_reap` |

## Per-batch details

### base_topk4

| batch | total tok/s | per-stream tok/s | decode_s | mem_used_gib | status |
|---:|---:|---:|---:|---:|---|
| 32 | 544.04 | 17.00 | 120.462 | 29.50665283203125 | ok |

### base_topk2

| batch | total tok/s | per-stream tok/s | decode_s | mem_used_gib | status |
|---:|---:|---:|---:|---:|---|
| 32 | 561.28 | 17.54 | 116.763 | 39.54376220703125 | ok |

### freq50_topk4

| batch | total tok/s | per-stream tok/s | decode_s | mem_used_gib | status |
|---:|---:|---:|---:|---:|---|
| 32 | 562.32 | 17.57 | 116.547 | 28.66290283203125 | ok |

### reap50_topk4

| batch | total tok/s | per-stream tok/s | decode_s | mem_used_gib | status |
|---:|---:|---:|---:|---:|---|
| 32 | 573.51 | 17.92 | 114.271 | 20.65704345703125 | ok |

### reap50_topk2

| batch | total tok/s | per-stream tok/s | decode_s | mem_used_gib | status |
|---:|---:|---:|---:|---:|---|
| 32 | 591.61 | 18.49 | 110.776 | 25.43438720703125 | ok |

## Reproduce

```bash
modal run modal/benchmark_decode_throughput_reap_vs_freq.py --prompt-len 256 --new-tokens 2048 --batch-sizes 1,2,4,8,16,32
```
