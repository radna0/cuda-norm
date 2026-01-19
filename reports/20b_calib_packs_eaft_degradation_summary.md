# EAFT degradation summary (base vs pruned)

- dataset: `radna0/harmony-qwen3-calib-packs-v2-20260113`
- base: `openai/gpt-oss-20b`
- pruned: `openai/gpt-oss-20b`
- top_k: 4
- entropy_topk: 20

## seq_len=1024

| rank | pack | score_z | score | ΔPPL | ΔNLL | ΔNLL p | CC Δ (pp) | CC Δ CI | JS2D | Δ mean p (pp) | CC z | CC p |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| — | **GLOBAL_AVG** | — | — | +0.000 | — | — | +0.00 | — | 0.0000 | +0.00 | 0.00 | 1.00e+00 |
| 1 | reasoning_style_10k_v2 | +0.00 | +0.00 | +0.000 | +0.0000 | 1.00e+00 | +0.00 | -0.0017–+0.0017 | 0.0000 | +0.00 | 0.00 | 1.00e+00 |
| 2 | tool_agentic_10k_v6 | +0.00 | +0.00 | +0.000 | +0.0000 | 1.00e+00 | +0.00 | -0.0019–+0.0019 | 0.0000 | +0.00 | 0.00 | 1.00e+00 |
| 3 | calib_prompt_10000_v2 | +0.00 | +0.00 | +0.000 | +0.0000 | 1.00e+00 | +0.00 | -0.0023–+0.0023 | 0.0000 | +0.00 | 0.00 | 1.00e+00 |
| 4 | UNION | +0.00 | +0.00 | +0.000 | +0.0000 | 1.00e+00 | +0.00 | -0.0022–+0.0022 | 0.0000 | +0.00 | 0.00 | 1.00e+00 |

Worst pack (by score): `reasoning_style_10k_v2` | score_z=+0.00 | score=+0.00 | ΔPPL=+0.000 | ΔNLL=+0.0000 | CC Δ=+0.00 pp

## seq_len=2048

| rank | pack | score_z | score | ΔPPL | ΔNLL | ΔNLL p | CC Δ (pp) | CC Δ CI | JS2D | Δ mean p (pp) | CC z | CC p |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| — | **GLOBAL_AVG** | — | — | +0.000 | — | — | +0.00 | — | 0.0000 | +0.00 | 0.00 | 1.00e+00 |
| 1 | reasoning_style_10k_v2 | +0.00 | +0.00 | +0.000 | +0.0000 | 1.00e+00 | +0.00 | -0.0010–+0.0010 | 0.0000 | +0.00 | 0.00 | 1.00e+00 |
| 2 | tool_agentic_10k_v6 | +0.00 | +0.00 | +0.000 | +0.0000 | 1.00e+00 | +0.00 | -0.0016–+0.0016 | 0.0000 | +0.00 | 0.00 | 1.00e+00 |
| 3 | calib_prompt_10000_v2 | +0.00 | +0.00 | +0.000 | +0.0000 | 1.00e+00 | +0.00 | -0.0003–+0.0003 | 0.0000 | +0.00 | 0.00 | 1.00e+00 |
| 4 | UNION | +0.00 | +0.00 | +0.000 | +0.0000 | 1.00e+00 | +0.00 | -0.0010–+0.0010 | 0.0000 | +0.00 | 0.00 | 1.00e+00 |

Worst pack (by score): `reasoning_style_10k_v2` | score_z=+0.00 | score=+0.00 | ΔPPL=+0.000 | ΔNLL=+0.0000 | CC Δ=+0.00 pp

## Interpretation

- `score_z` is the z-score of the combined degradation (ΔPPL + CC Δ + JS2D + Δ mean p) across packs.
- `CC p` is a two-sided p-value from the CC z-score (smaller means more significant).
- For quality risk: look for **positive ΔPPL**, **positive CC Δ**, **positive JS2D**, and **mean p drop** together.
