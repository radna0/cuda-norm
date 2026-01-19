# 20B structural prune build (REAP-lite ranking)

- Base model: `openai/gpt-oss-20b`
- General dataset: `radna0/harmony-nemotron-cpu-artifacts` split `train` col `text`
- Math dataset: `radna0/nemotron-math-v2-harmony-tools` split `high_part00` col `text`
- Profile rows: 20 | Max seq length: 512 | Batch size: 1

## Variants

- general_50pct_experts_reap: `/root/model/artifacts/20b_pruned_models_reap/general_50pct_experts_reap`
- math_25pct_experts_reap: `/root/model/artifacts/20b_pruned_models_reap/math_25pct_experts_reap`

## Sanity inference

- general ok=True
- math ok=True

## Artifacts

- `artifacts/20b_pruned_models_reap/manifest_reap.json`

## Reproduce

```bash
modal run modal/gpt_oss_pruning_track.py --task build_pruned_20b_reap
```
