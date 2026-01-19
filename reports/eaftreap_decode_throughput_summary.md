# EAFT‑REAP Decode Throughput (Kaggle / SGLang / H100)

Decode benchmark configuration:
- Prompt length: 256
- Temperature: 0.0 (greedy)
- `min_new_tokens = max_new_tokens` and `ignore_eos = true` (force long decode)
- Attention backend: `fa3`

## Results (total tokens/sec)

### `max_new_tokens = 2048`

| batch | base tok/s | EAFT‑REAP‑50 tok/s | delta |
|---:|---:|---:|---:|
| 1 | 216.0 | 249.3 | +15.4% |
| 2 | 512.3 | 517.8 | +1.1% |
| 4 | 1012.3 | 1024.7 | +1.2% |
| 8 | 1967.7 | 2036.2 | +3.5% |
| 16 | 3700.2 | 3804.2 | +2.8% |
| 32 | 6149.6 | 6529.8 | +6.2% |

### `max_new_tokens = 8192`

| batch | base tok/s | EAFT‑REAP‑50 tok/s | delta |
|---:|---:|---:|---:|
| 1 | 292.0 | 292.0 | -0.0% |
| 2 | 508.2 | 515.6 | +1.4% |
| 4 | 989.1 | 998.7 | +1.0% |
| 8 | 1867.1 | 1903.0 | +1.9% |
| 16 | 3337.7 | 3393.6 | +1.7% |
| 32 | 5289.7 | 5472.2 | +3.5% |

## Raw artifacts

- `harmony/cuda-norm/kaggle_runs/decode_fetch_20260116_143119/logs/decode_base_2048_20260116_141403.json`
- `harmony/cuda-norm/kaggle_runs/decode_fetch_20260116_143119/logs/decode_eaftreap50_2048_20260116_142219.json`
- `harmony/cuda-norm/kaggle_runs/decode_fetch_20260116_143808/logs/decode_base_8192_20260116_142856.json`
- `harmony/cuda-norm/kaggle_runs/decode_fetch_20260116_143808/logs/decode_eaftreap50_8192_20260116_143339.json`

