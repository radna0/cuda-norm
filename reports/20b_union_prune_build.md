# 20B structural prune build (union expert set)

- Base model: `openai/gpt-oss-20b`
- union50: `/root/model/artifacts/20b_union_pruned/union50` ok=True
- unionAgg: `/root/model/artifacts/20b_union_pruned/unionAgg` ok=True

- Manifest: `artifacts/20b_union_pruned/manifest_union.json`

## Reproduce

```bash
modal run modal/gpt_oss_pruning_track.py --task build_pruned_20b_union
```
