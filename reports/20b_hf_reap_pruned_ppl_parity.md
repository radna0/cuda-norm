# 20B HF REAP-pruned model: parity PPL

- Dataset: `radna0/nemotron-math-v2-harmony-tools` split `high_part00` col `text`
- Blocks: 64 | Batch size: 1
- Base: `openai/gpt-oss-20b` (cfg_top_k=4)
- Pruned: `sandeshrajx/gpt-oss-20b-reap-0.5-mxfp4` (cfg_top_k=4)

| model | top_k | ppl1024 | ppl2048 | delta vs base_topk4 (1024/2048) |
|---|---:|---:|---:|---:|
| base_topk4 | 4 | 2.805 | 2.589 | +0.000 / +0.000 |
| base_topk2 | 2 | 3.099 | 2.830 | +0.294 / +0.241 |
| pruned_as_is | 4 | 3.327 | 3.034 | +0.522 / +0.445 |
| pruned_topk2 | 2 | 3.646 | 3.291 | +0.841 / +0.701 |
| pruned_topk4 | 4 | 3.327 | 3.034 | +0.522 / +0.445 |

## Notes

- Completion-only PPL on Harmony assistant spans, packed into blocks of seq_len+1.
- `pruned_as_is` uses the pruned model's config routing; `pruned_topk2/4` forces router.top_k at runtime.

## Reproduce

```bash
modal run modal/eval_hf_reap_pruned_20b_parity.py --pruned-model-id sandeshrajx/gpt-oss-20b-reap-0.5-mxfp4 --num-blocks 64 --batch-size 1
```
