# 20B structural prune build (frequency baseline)

- Base model: `openai/gpt-oss-20b`
- General dataset: `radna0/harmony-nemotron-cpu-artifacts` split `train` col `text`
- Math dataset: `radna0/nemotron-math-v2-harmony-tools` split `high_part00` col `text`
- Profile rows: 50 | Max seq length: 512

## Variants

- general_50pct_experts_freq: `/root/model/artifacts/20b_pruned_models_freq/general_50pct_experts_freq`
- math_25pct_experts_freq: `/root/model/artifacts/20b_pruned_models_freq/math_25pct_experts_freq`

## Sanity inference

- general ok=True
- math ok=True

## Artifacts

- `artifacts/20b_pruned_models_freq/manifest_freq.json`

## Reproduce

```bash
modal run modal/gpt_oss_pruning_track.py --task build_pruned_20b_freq
```
