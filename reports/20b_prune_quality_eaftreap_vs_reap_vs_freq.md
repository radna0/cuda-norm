# 20B prune quality: EAFT-REAP vs REAP vs frequency (parity PPL)

- Dataset: `radna0/nemotron-math-v2-harmony-tools` split `high_part00` col `text`
- Blocks: 64 | Batch size: 1
- Seq lens: `1024,2048`

| model | keep_frac | top_k | ppl1024 | ppl2048 | delta vs base (1024/2048) |
|---|---:|---:|---:|---:|---:|
| base | 1.00 | 4 | 2.805 | 2.591 | +0.000 / +0.000 |
| freq_50 | 0.50 | 4 | 6.744 | 5.756 | +3.939 / +3.166 |
| reap_50 | 0.50 | 4 | 3.902 | 3.472 | +1.097 / +0.882 |
| eaftreap_50 | 0.50 | 4 | 3.495 | 3.171 | +0.691 / +0.580 |
| freq_25 | 0.25 | 4 | 3.601 | 3.263 | +0.796 / +0.673 |
| reap_25 | 0.25 | 4 | 3.599 | 3.278 | +0.794 / +0.687 |
| eaftreap_25 | 0.25 | 4 | 4.296 | 3.903 | +1.491 / +1.312 |

## Notes

- This is completion-only PPL on Harmony assistant spans, packed into blocks of seq_len+1.
- `tok_s_pred` (internal) is prefill/scoring throughput and is not decode throughput.

## Reproduce

```bash
modal run modal/eval_prune_quality_reap_vs_freq.py --num-blocks 64 --batch-size 1
```
