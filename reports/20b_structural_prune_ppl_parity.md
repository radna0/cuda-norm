# 20B structural prune PPL (parity harness)

- Dataset: `radna0/nemotron-math-v2-harmony-tools` split `high_part00` col `text`
- seq_len: 1024 | blocks: 64 | batch_size: 1
- rows_seen: 5 | pack_wall_s: 4.5s

| model | ppl | ppl_delta | tok/s(pred) | path |
|---|---:|---:|---:|---|
| base | 2.805128 | +0.0000 | 4245 | `/__modal/volumes/vo-QXnMbFTIRP0PqV5p2edMFh/.hf_cache/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee` |
| general_50pct_experts | 3.920127 | +1.1150 | 9338 | `/root/model/artifacts/20b_pruned_models/general_50pct_experts` |
| math_25pct_experts | 4.693813 | +1.8887 | 8625 | `/root/model/artifacts/20b_pruned_models/math_25pct_experts` |

## Reproduce

```bash
modal run modal/eval_structural_prune_20b_parity.py --seq-len 1024 --num-blocks 64
```
