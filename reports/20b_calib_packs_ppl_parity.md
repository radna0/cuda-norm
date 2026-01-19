# 20B pruning eval: calib packs parity PPL

- Dataset repo: `radna0/harmony-qwen3-calib-packs-v2-20260113`
- Packs: `packs/reasoning_style_10k_v2/reasoning_style_10k_v2.parquet`, `tool_agentic_10k_v6.parquet`, `packs/calib_prompt_10000_v2/calib_prompt_10000_v2.parquet`
- Base: `openai/gpt-oss-20b`
- Pruned: `sandeshrajx/gpt-oss-20b-reap-0.5-mxfp4`
- top_k (forced for both): 4
- blocks per pack: 64 | batch_size: 1

| pack | base ppl1024 | pruned ppl1024 | delta | base ppl2048 | pruned ppl2048 | delta |
|---|---:|---:|---:|---:|---:|---:|
| reasoning_style_10k_v2 | 5.501 | 8.070 | +2.569 | 4.603 | 6.777 | +2.174 |
| tool_agentic_10k_v6 | 5.677 | 7.990 | +2.313 | 4.775 | 6.780 | +2.004 |
| calib_prompt_10000_v2 | 5.965 | 7.994 | +2.030 | 4.545 | 6.063 | +1.518 |
| UNION | 5.226 | 7.466 | +2.240 | 4.722 | 6.550 | +1.829 |

## Notes

- Completion-only PPL on Harmony assistant spans, packed into blocks of seq_len+1.
- Each pack is evaluated independently; UNION is a round-robin interleaving across packs (not dominated by the first pack).
- top_k is forced to keep routing consistent across models.

## Reproduce

```bash
modal run modal/eval_calib_packs_ppl_parity.py \
  --dataset-repo radna0/harmony-qwen3-calib-packs-v2-20260113 \
  --pack-files-csv packs/reasoning_style_10k_v2/reasoning_style_10k_v2.parquet,tool_agentic_10k_v6.parquet,packs/calib_prompt_10000_v2/calib_prompt_10000_v2.parquet \
  --base-model-id openai/gpt-oss-20b \
  --pruned-model-id sandeshrajx/gpt-oss-20b-reap-0.5-mxfp4 \
  --top-k 4 --num-blocks 64 --batch-size 1
```
