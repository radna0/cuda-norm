# 20B decode throughput: REAP vs frequency

- prompt_len: 256 | new_tokens: 512
- batch_sweep: 1,2,4,8,16,32

| run | top_k | max_batch | total tok/s @max | per-stream tok/s @max | mem_used_gib @max | model |
|---|---:|---:|---:|---:|---:|---|
| base_topk4 | 4 | 32 | 602.75 | 18.84 | 20.4 | `openai/gpt-oss-20b` |
| base_topk2 | 2 | 32 | 592.10 | 18.50 | 30.3 | `openai/gpt-oss-20b` |
| freq50_topk4 | 4 | 32 | 579.04 | 18.09 | 29.3 | `/root/model/artifacts/20b_pruned_models_freq/general_50pct_experts_freq` |
| reap50_topk4 | 4 | 32 | 590.27 | 18.45 | 20.3 | `/root/model/artifacts/20b_pruned_models_reap/general_50pct_experts_reap` |
| reap50_topk2 | 2 | 32 | 575.91 | 18.00 | 24.9 | `/root/model/artifacts/20b_pruned_models_reap/general_50pct_experts_reap` |

## Per-batch details

### base_topk4

| batch | total tok/s | per-stream tok/s | decode_s | mem_used_gib | status |
|---:|---:|---:|---:|---:|---|
| 1 | 15.37 | 15.37 | 33.319 | 14.62188720703125 | ok |
| 2 | 36.58 | 18.29 | 27.990 | 14.63751220703125 | ok |
| 4 | 80.04 | 20.01 | 25.587 | 15.03204345703125 | ok |
| 8 | 160.75 | 20.09 | 25.480 | 15.80157470703125 | ok |
| 16 | 316.80 | 19.80 | 25.859 | 17.33673095703125 | ok |
| 32 | 602.75 | 18.84 | 27.182 | 20.40509033203125 | ok |

### base_topk2

| batch | total tok/s | per-stream tok/s | decode_s | mem_used_gib | status |
|---:|---:|---:|---:|---:|---|
| 1 | 17.90 | 17.90 | 28.607 | 24.88751220703125 | ok |
| 2 | 33.48 | 16.74 | 30.589 | 24.90118408203125 | ok |
| 4 | 71.62 | 17.90 | 28.597 | 24.91485595703125 | ok |
| 8 | 137.48 | 17.19 | 29.793 | 25.68243408203125 | ok |
| 16 | 275.96 | 17.25 | 29.685 | 27.21759033203125 | ok |
| 32 | 592.10 | 18.50 | 27.671 | 30.28594970703125 | ok |

### freq50_topk4

| batch | total tok/s | per-stream tok/s | decode_s | mem_used_gib | status |
|---:|---:|---:|---:|---:|---|
| 1 | 16.39 | 16.39 | 31.231 | 24.64532470703125 | ok |
| 2 | 35.40 | 17.70 | 28.928 | 24.64923095703125 | ok |
| 4 | 69.79 | 17.45 | 29.343 | 24.66094970703125 | ok |
| 8 | 134.53 | 16.82 | 30.448 | 24.66094970703125 | ok |
| 16 | 285.47 | 17.84 | 28.696 | 26.19610595703125 | ok |
| 32 | 579.04 | 18.09 | 28.295 | 29.26446533203125 | ok |

### reap50_topk4

| batch | total tok/s | per-stream tok/s | decode_s | mem_used_gib | status |
|---:|---:|---:|---:|---:|---|
| 1 | 18.56 | 18.56 | 27.588 | 15.69219970703125 | ok |
| 2 | 36.94 | 18.47 | 27.724 | 15.70977783203125 | ok |
| 4 | 74.00 | 18.50 | 27.675 | 15.71954345703125 | ok |
| 8 | 146.38 | 18.30 | 27.981 | 15.71954345703125 | ok |
| 16 | 307.12 | 19.20 | 26.673 | 17.25469970703125 | ok |
| 32 | 590.27 | 18.45 | 27.757 | 20.32305908203125 | ok |

### reap50_topk2

| batch | total tok/s | per-stream tok/s | decode_s | mem_used_gib | status |
|---:|---:|---:|---:|---:|---|
| 1 | 17.55 | 17.55 | 29.168 | 20.30548095703125 | ok |
| 2 | 34.90 | 17.45 | 29.342 | 20.30938720703125 | ok |
| 4 | 71.95 | 17.99 | 28.464 | 20.32110595703125 | ok |
| 8 | 134.99 | 16.87 | 30.343 | 20.32110595703125 | ok |
| 16 | 272.16 | 17.01 | 30.100 | 21.85626220703125 | ok |
| 32 | 575.91 | 18.00 | 28.449 | 24.92462158203125 | ok |

## Reproduce

```bash
modal run modal/benchmark_decode_throughput_reap_vs_freq.py --prompt-len 256 --new-tokens 2048 --batch-sizes 1,2,4,8,16,32
```
