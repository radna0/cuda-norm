# 120B structural prune build (baseline: first64)

- Base model: `openai/gpt-oss-120b`
- keep_n: 64/128 (all 36 layers)

## Output

- model_dir: `/root/model/artifacts/120b_pruned_models/first64_experts_keepfrac50`

## Validation (CPU)

- wrote: `artifacts/120b_pruned_models_first64/manifest.json`

## Reproduce

```bash
modal run modal/gpt_oss_pruning_track.py --task build_pruned_120b_first64
```
