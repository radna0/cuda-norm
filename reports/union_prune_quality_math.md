# Union prune quality (parity PPL)

- Models: base=`openai/gpt-oss-20b`, union50=`/root/model/artifacts/20b_union_pruned/union50`, unionAgg=`/root/model/artifacts/20b_union_pruned/unionAgg`
- Blocks: 32 | Batch size: 2
- Domains: math
- Seq lens: 1024, 2048

| domain | seq_len | base | union50 | unionAgg |
|---|---:|---:|---:|---:|
| math | 1024 | 2.921 | 3.088 | 3.518 |
| math | 2048 | 2.575 | 2.704 | 3.024 |

## Notes

- Completion-only PPL on Harmony assistant spans, packed into blocks of seq_len+1.
- If a domain row budget is insufficient (not enough blocks), the entry is `NA` (allow_missing_domains=true).

## Reproduce

```bash
modal run modal/eval_union_prune_quality.py --num-blocks 64 --batch-size 1 --domains-csv math,agentic,general
```
