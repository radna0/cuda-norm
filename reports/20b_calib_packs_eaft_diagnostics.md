# 20B pruning diagnostics: EAFT-style Confident Conflicts

- Dataset repo: `radna0/harmony-qwen3-calib-packs-v2-20260113`
- Packs: `packs/reasoning_style_10k_v2/reasoning_style_10k_v2.parquet`, `tool_agentic_10k_v6.parquet`, `packs/calib_prompt_10000_v2/calib_prompt_10000_v2.parquet`
- Base: `openai/gpt-oss-20b`
- Pruned: `sandeshrajx/gpt-oss-20b-reap-0.5-mxfp4`
- top_k (forced for both): 4
- entropy_topk: 20 (entropy normalized by ln(K))
- CC definition: bottom 0.15 quantile in BOTH p_t and H_t (completion-only kept tokens)
- blocks per pack per seq_len: 16 | batch_size: 1

## Summary (PPL + CC_rate)

| pack | ppl1024 base | ppl1024 pruned | Δ | CC1024 base | CC1024 pruned | pruned CC@baseThr | ppl2048 base | ppl2048 pruned | Δ | CC2048 base | CC2048 pruned | pruned CC@baseThr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reasoning_style_10k_v2 | 5.695 | 7.651 | +1.955 | 0.0026 | 0.0038 | 0.0028 | 4.327 | 6.622 | +2.295 | 0.0004 | 0.0021 | 0.0006 |
| tool_agentic_10k_v6 | 6.024 | 8.466 | +2.442 | 0.0014 | 0.0029 | 0.0005 | 5.305 | 7.743 | +2.438 | 0.0013 | 0.0031 | 0.0009 |
| calib_prompt_10000_v2 | 4.745 | 5.975 | +1.230 | 0.0001 | 0.0006 | 0.0001 | 4.685 | 6.056 | +1.371 | 0.0003 | 0.0010 | 0.0004 |
| UNION | 5.752 | 7.638 | +1.886 | 0.0019 | 0.0034 | 0.0014 | 4.613 | 6.374 | +1.762 | 0.0014 | 0.0028 | 0.0016 |

## Mean Probability / Entropy

These summarize the EAFT landscape axes on completion-only tokens:
- `mean_prob` = mean reference-token probability `p_t`
- `mean_entropy` = mean normalized Top-K entropy `H_topK / ln(K)`

### seq_len=1024

| pack | base mean_prob | pruned mean_prob | base mean_entropy | pruned mean_entropy |
|---|---:|---:|---:|---:|
| reasoning_style_10k_v2 | 0.4381 | 0.3902 | 0.4318 | 0.4669 |
| tool_agentic_10k_v6 | 0.4826 | 0.4142 | 0.3474 | 0.4275 |
| calib_prompt_10000_v2 | 0.5073 | 0.4678 | 0.3775 | 0.4074 |
| UNION | 0.4462 | 0.3960 | 0.4185 | 0.4599 |

### seq_len=2048

| pack | base mean_prob | pruned mean_prob | base mean_entropy | pruned mean_entropy |
|---|---:|---:|---:|---:|
| reasoning_style_10k_v2 | 0.4963 | 0.4309 | 0.3821 | 0.4277 |
| tool_agentic_10k_v6 | 0.5120 | 0.4345 | 0.3152 | 0.4003 |
| calib_prompt_10000_v2 | 0.4926 | 0.4519 | 0.3893 | 0.4159 |
| UNION | 0.4849 | 0.4288 | 0.3855 | 0.4285 |

## Notes

- `CC_rate` is computed per-model using its own p/H quantile thresholds; `pruned CC@baseThr` applies base thresholds to pruned tokens.
- `H_t` is Top-K entropy (K=entropy_topk) computed on the model's Top-K distribution and normalized by ln(K).
- UNION is round-robin interleaving across packs (not dominated by the first pack).

## Reproduce

```bash
modal run modal/eval_calib_packs_eaft_diagnostics.py \
  --dataset-repo radna0/harmony-qwen3-calib-packs-v2-20260113 \
  --pack-files-csv packs/reasoning_style_10k_v2/reasoning_style_10k_v2.parquet,tool_agentic_10k_v6.parquet,packs/calib_prompt_10000_v2/calib_prompt_10000_v2.parquet \
  --base-model-id openai/gpt-oss-20b \
  --pruned-model-id sandeshrajx/gpt-oss-20b-reap-0.5-mxfp4 \
  --top-k 4 --entropy-topk 20 --cc-quantile 0.15 \
  --num-blocks 16 --batch-size 1
```
