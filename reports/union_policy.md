# Union expert-set policy (REAP-lite)

- Domains: agentic, general, math (column `meta_domain`)
- Rows/domain used: 200 (see `artifacts/reap_saliency_by_domain/`)
- Weights: {"agentic": 1.0, "general": 1.0, "math": 1.0, "science": 1.1}

## Outputs

- `artifacts/union_expert_sets/union_cov95.json`
- `artifacts/union_expert_sets/union_cov97.json`
- `artifacts/union_expert_sets/union_cov99.json`
- `artifacts/union_expert_sets/union50_cap16_cov95.json`
- `artifacts/union_expert_sets/unionAgg_cap12_cov97.json`

## Kept-expert counts (per layer)

- union50 (cap16@cov95): {"min": 16, "max": 16, "avg": 16.0, "counts": [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]}
- unionAgg (cap12@cov97): {"min": 12, "max": 12, "avg": 12.0, "counts": [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]}
