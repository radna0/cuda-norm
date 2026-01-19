# EAFT‑REAP Quality Snapshot (curated calib packs)

Eval regime:
- Dataset repo: `radna0/harmony-qwen3-calib-packs-v2-20260113`
- Pack: `UNION`
- Completion-only packed NLL (EAFT harness) with Top‑20 entropy + CC rate at `cc_quantile=0.15`
- Seq lens: 1024, 2048

Near‑lossless gates: `harmony/cuda-norm/pruning/near_lossless_gates.json` (strict).

## Models compared

- Base: `openai/gpt-oss-20b` (Kaggle local model dir)
- HF REAP 0.5: `sandeshrajx/gpt-oss-20b-reap-0.5-mxfp4`
- EAFT‑REAP‑50: structural prune, keep 16/32 experts per layer (EAFT‑REAP ranking)
- EAFT‑REAP‑25: structural prune, keep 8/32 experts per layer (EAFT‑REAP ranking)

## Results (UNION)

### Seq len 1024

| model | ppl | ΔPPL vs base | CC rate | ΔCC vs base | mean p | Δ mean p | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| base | 5.237 | 0.000 | 0.000632 | +0.000000 | 0.4746 | +0.0000 | PASS (baseline) |
| HF REAP 0.5 | 7.649 | +2.412 | 0.001590 | +0.000958 | 0.4158 | -0.0588 | FAIL |
| EAFT‑REAP‑50 | 7.290 | +2.053 | 0.001869 | +0.001237 | 0.4213 | -0.0533 | FAIL |
| EAFT‑REAP‑25 | 18.272 | +13.035 | 0.005183 | +0.004551 | 0.3046 | -0.1700 | FAIL |

### Seq len 2048

| model | ppl | ΔPPL vs base | CC rate | ΔCC vs base | mean p | Δ mean p | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| base | 4.716 | 0.000 | 0.001187 | +0.000000 | 0.4823 | +0.0000 | PASS (baseline) |
| HF REAP 0.5 | 6.574 | +1.857 | 0.002411 | +0.001225 | 0.4285 | -0.0538 | FAIL |
| EAFT‑REAP‑50 | 6.357 | +1.641 | 0.002604 | +0.001418 | 0.4304 | -0.0519 | FAIL |
| EAFT‑REAP‑25 | 15.859 | +11.143 | 0.006315 | +0.005128 | 0.3101 | -0.1722 | FAIL |

## Interpretation

- At **keep_frac=0.50**, EAFT‑REAP‑50 is slightly better than HF REAP 0.5 on PPL, but both are far from “near‑lossless”.
- At **keep_frac=0.25**, both methods collapse badly on the generalist union packs (expected; too aggressive without recovery distill).
- Under the current strict gates, **none of the 50% prunes pass**; near‑lossless likely requires a much higher keep fraction (e.g., 0.8–0.9) or a recovery step.

## Source artifacts

Used to render `harmony/cuda-norm/reports/eaft_dynamic_compare_kaggle_eaftreap.html`:
- `harmony/cuda-norm/kaggle_runs/eaft_models_fetch_20260116_044616/eaft_models/20260116_034608/openai_gpt-oss-20b.json`
- `harmony/cuda-norm/kaggle_runs/eaft_models_fetch_20260116_044616/eaft_models/20260116_034819/sandeshrajx_gpt-oss-20b-reap-0.5-mxfp4.json`
- `harmony/cuda-norm/kaggle_runs/eaft_models_fetch_20260116_044616/eaft_models/20260116_044303/eaftreap_general_50.json`
- `harmony/cuda-norm/kaggle_runs/eaft_models_fetch_20260116_044616/eaft_models/20260116_044539/eaftreap_math_25.json`

