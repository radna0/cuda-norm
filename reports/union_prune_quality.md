# Union prune quality (parity PPL) — summary

- Base model: `openai/gpt-oss-20b`
- Union models: `/root/model/artifacts/20b_union_pruned/union50`, `/root/model/artifacts/20b_union_pruned/unionAgg`
- Notes: completion-only PPL on Harmony assistant spans, packed into blocks of `seq_len + 1`.
- Slice reports:
  - `reports/union_prune_quality_math.md`
  - `reports/union_prune_quality_agentic.md`
  - `reports/union_prune_quality_general.md`

| domain | seq_len | base | union50 | unionAgg |
|---|---:|---:|---:|---:|
| math | 1024 | 2.921 | 3.088 | 3.518 |
| math | 2048 | 2.575 | 2.704 | 3.024 |
| agentic | 1024 | 5.525 | 5.973 | 10.145 |
| agentic | 2048 | 4.828 | 5.125 | 8.794 |
| general (chat_if) | 1024 | 7.208 | 9.674 | 17.110 |
| general (chat_if) | 2048 | 5.017 | 6.498 | 11.020 |

