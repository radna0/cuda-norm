# 20B soft prune eval (inference-only)

## Note (not truth-PPL comparable)

This table is **not** comparable to our “truth PPL” harness (e.g. baseline ~2.9 on packed Harmony blocks):
- It scores per-row `out.loss` over *all tokens* (no completion-only masking).
- It does not use packed-block evaluation (concat+EOS then chunk).

Use this report for **throughput + relative trend only**.
For comparable PPL, use:
- `reports/pruning_eval_parity.md`
- `reports/20b_soft_prune_eval_parity.md`

- Model: `openai/gpt-oss-20b`
- Dataset: `radna0/harmony-nemotron-cpu-artifacts` split `train`
- Eval rows: 256 | Max seq length: 4096

| kept_experts | keep_frac | top_k | ppl | ppl_delta | tokens/s |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.25 | 2 | 552.028 | +505.809 | 10209 |
| 8 | 0.25 | 4 | 177.584 | +131.365 | 9550 |
| 16 | 0.50 | 2 | 176.291 | +130.072 | 10198 |
| 16 | 0.50 | 4 | 58.504 | +12.286 | 9466 |
| 32 | 1.00 | 2 | 115.073 | +68.854 | 5331 |
| 32 | 1.00 | 4 | 46.219 | +0.000 | 4117 |

## Reproduce

```bash
modal run modal/gpt_oss_pruning_track.py --task soft_prune_20b
```
