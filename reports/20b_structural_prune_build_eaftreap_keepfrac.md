# 20B structural prune build (EAFT-REAP, calib packs, keep_frac sweep)

- Base model: `openai/gpt-oss-20b`
- Calib repo: `radna0/harmony-qwen3-calib-packs-v2-20260113`
- Packs: packs/reasoning_style_10k_v2/reasoning_style_10k_v2.parquet, tool_agentic_10k_v6.parquet, packs/calib_prompt_10000_v2/calib_prompt_10000_v2.parquet
- Sample rows: 2000 seed=3407 sample_jsonl=`/kaggle/working/pruning_cache/data/calib_packs_samples/calib_packs_sample_1f58729a443affac5fceae125486571f_n2000.jsonl`
- Max seq length: 4096 | Batch size: 1
- EAFT weights: good=1.0 uncertain=0.25 conflict=-2.0

## Variants

- calib_union_keep24of32_k75_eaftreap: keep_frac=None keep_n=24 dir=`/kaggle/working/artifacts/harmony_cuda_norm/20b_pruned_models_eaftreap/calib_union_keep24of32_k75_eaftreap`

## Artifacts

- `artifacts/20b_pruned_models_eaftreap_keepfrac/manifest_eaftreap_keepfrac.json`

## Reproduce (Kaggle/VERSA)

```bash
bash harmony/cuda-norm/scripts/versa_run_pruning_track_kaggle.sh \
  --task build_pruned_20b_eaftreap_keepfrac \
  --model-id-20b openai/gpt-oss-20b \
  --num-rows 2000 --max-seq-length 4096 --batch-size 1 \
  --keep-fracs-csv 0.75
```
