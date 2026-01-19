# 20B structural prune build (EAFT-REAP ranking)

- Base model: `openai/gpt-oss-20b`
- General dataset: `radna0/harmony-nemotron-cpu-artifacts` split `train` col `text`
- Math dataset: `radna0/nemotron-math-v2-harmony-tools` split `high_part00` col `text`
- Profile rows: 128 | Max seq length: 4096 | Batch size: 1
- EAFT: cc_q=0.15 uncertain_q=0.85 entropy_topk=20 weights(good/uncertain/conflict)=1.0/0.25/-2.0

## Variants

- general_50pct_experts_eaftreap: `/root/model/artifacts/20b_pruned_models_eaftreap/general_50pct_experts_eaftreap`
- math_25pct_experts_eaftreap: `/root/model/artifacts/20b_pruned_models_eaftreap/math_25pct_experts_eaftreap`

## Sanity inference

- general ok=True
- math ok=True

## Artifacts

- `artifacts/20b_pruned_models_eaftreap/manifest_eaftreap.json`

## Reproduce

```bash
modal run modal/gpt_oss_pruning_track.py --task build_pruned_20b_eaftreap
```
