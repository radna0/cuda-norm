# 20B prune quality: REAP vs frequency (parity PPL)

- Dataset: `radna0/nemotron-math-v2-harmony-tools` split `high_part00` col `text`
- Blocks: 64 | Batch size: 1

| model | keep_frac | top_k | ppl1024 | ppl2048 | delta vs base (1024/2048) |
|---|---:|---:|---:|---:|---:|
| base | 1.00 | 4 | 2.805 | 2.589 | +0.000 / +0.000 |
| freq_50 | 0.50 | 4 | 6.459 | 5.538 | +3.653 / +2.949 |
| reap_50 | 0.50 | 4 | 3.977 | 3.529 | +1.171 / +0.940 |
| freq_25 | 0.25 | 4 | 4.042 | 3.684 | +1.237 / +1.095 |
| reap_25 | 0.25 | 4 | 3.864 | 3.547 | +1.059 / +0.957 |

## Notes

- This is completion-only PPL on Harmony assistant spans, packed into blocks of seq_len+1.
- `tok_s_pred` (internal) is prefill/scoring throughput and is not decode throughput.

## Reproduce

```bash
modal run modal/eval_prune_quality_reap_vs_freq.py --num-blocks 64 --batch-size 1
```
