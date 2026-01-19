# Union prune quality (parity PPL)

- Models: base=`openai/gpt-oss-20b`, union50=`/root/model/artifacts/20b_union_pruned/union50`, unionAgg=`/root/model/artifacts/20b_union_pruned/unionAgg`
- Blocks: 32 | Batch size: 2
- Domains: general
- Seq lens: 1024, 2048

| domain | seq_len | base | union50 | unionAgg |
|---|---:|---:|---:|---:|
| general | 1024 | 7.208 | 9.674 | 17.110 |
| general | 2048 | 5.017 | 6.498 | 11.020 |

## Notes

- Completion-only PPL on Harmony assistant spans, packed into blocks of seq_len+1.
- If a domain row budget is insufficient (not enough blocks), the entry is `NA` (allow_missing_domains=true).

## Reproduce

```bash
modal run modal/eval_union_prune_quality.py --num-blocks 64 --batch-size 1 --domains-csv math,agentic,general
```
