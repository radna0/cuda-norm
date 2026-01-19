# EAFT plots (20B base vs pruned)

Open: `reports/20b_calib_packs_eaft_plots.html`

This dashboard visualizes EAFT’s probability–entropy landscape on completion-only tokens (Harmony-packed blocks) for each calib pack + UNION:

- 2D density heatmaps of `(log10(p_t), H_topK/ln(K))` for base vs pruned
- 1D histograms for `log10(p_t)` and `H`
- Confident Conflict (CC) thresholds (dashed lines) and CC region shading

## Reproduce

```bash
mkdir -p unsloth_logs
ts=$(date +%Y%m%d_%H%M%S)
nohup env MODAL_PROFILE=phamtrinhkien1203 modal run modal/eval_calib_packs_eaft_plots.py \
  --base-model-id openai/gpt-oss-20b \
  --pruned-model-id sandeshrajx/gpt-oss-20b-reap-0.5-mxfp4 \
  --top-k 4 --entropy-topk 20 --cc-quantile 0.15 \
  --num-blocks 32 --batch-size 1 \
  --prob-scale linear \
  --hist-xbins 160 --hist-ybins 120 --logp-min -12 --logp-max 0 \
  > "unsloth_logs/calib_packs_eaft_plots_${ts}.log" 2>&1 &
```
