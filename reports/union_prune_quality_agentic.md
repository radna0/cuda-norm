# Union prune quality (parity PPL)

- Models: base=`openai/gpt-oss-20b`, union50=`/root/model/artifacts/20b_union_pruned/union50`, unionAgg=`/root/model/artifacts/20b_union_pruned/unionAgg`
- Blocks: 32 | Batch size: 2
- Domains: agentic
- Seq lens: 1024, 2048

| domain | seq_len | base | union50 | unionAgg |
|---|---:|---:|---:|---:|
| agentic | 1024 | 5.525 | 5.973 | 10.145 |
| agentic | 2048 | 4.828 | 5.125 | 8.794 |

## Notes

- Completion-only PPL on Harmony assistant spans, packed into blocks of seq_len+1.
- If a domain row budget is insufficient (not enough blocks), the entry is `NA` (allow_missing_domains=true).

## Reproduce

```bash
modal run modal/eval_union_prune_quality.py --num-blocks 64 --batch-size 1 --domains-csv math,agentic,general
```
