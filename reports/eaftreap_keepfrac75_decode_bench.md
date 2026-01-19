# Decode throughput: 20B base vs EAFT-REAP keep_frac=0.75

- Metric: decode-only total tok/s (SGLang `/generate`, `ignore_eos=true`, `min_new_tokens=max_new_tokens`).
- Prompt len: 256
- top_k: 4 (unchanged)
- Backend: fa3

## Summary

- max_new_tokens=2048: best_total base=5691.1 tok/s @bs=48, pruned=6177.5 tok/s @bs=48; at bs=64 Δ=+0.04%
- max_new_tokens=8192: best_total base=5302.1 tok/s @bs=32, pruned=5283.3 tok/s @bs=32; at bs=32 Δ=-0.35%

## Full sweep (paired rows)

### max_new_tokens=2048

| bs | base tok/s total | pruned tok/s total | Δ% |
|---:|---:|---:|---:|
| 1 | 259.5 | 251.2 | -3.19% |
| 2 | 490.4 | 490.4 | +0.00% |
| 4 | 944.5 | 944.5 | -0.00% |
| 8 | 1525.6 | 1525.6 | +0.00% |
| 16 | 2469.7 | 2469.5 | -0.01% |
| 32 | 2125.6 | 4180.7 | +96.68% |
| 48 | 5691.1 | 6177.5 | +8.55% |
| 64 | 4029.5 | 4031.2 | +0.04% |

### max_new_tokens=8192

| bs | base tok/s total | pruned tok/s total | Δ% |
|---:|---:|---:|---:|
| 1 | 292.3 | 289.1 | -1.07% |
| 2 | 509.3 | 506.9 | -0.48% |
| 4 | 989.7 | 986.4 | -0.34% |
| 8 | 1867.3 | 1862.4 | -0.26% |
| 16 | 3347.7 | 3327.9 | -0.59% |
| 32 | 5302.1 | 5283.3 | -0.35% |

## Notes

- Decode benchmarking is noisy; focus on the shared max batch rows and the best_total summary above.
- The 2048 sweep shows an anomalous slowdown for base at bs=32 in this run; bs=48 and bs=64 look consistent.

## Artifacts

- `harmony/cuda-norm/kaggle_fetch/decode_logs_20260116_163100/decode_base20b_2048_20260116_162803.json`
- `harmony/cuda-norm/kaggle_fetch/decode_logs_20260116_163100/decode_eaftreap75_2048_20260116_162822.json`
- `harmony/cuda-norm/kaggle_fetch/decode_logs_20260116_163347/decode_base20b_8192_20260116_163347.json`
- `harmony/cuda-norm/kaggle_fetch/decode_logs_20260116_163846_r2/decode_eaftreap75_8192_20260116_163846.json`

