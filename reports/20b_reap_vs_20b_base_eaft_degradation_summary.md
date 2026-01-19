# EAFT degradation summary (base vs pruned)

- dataset: `radna0/harmony-qwen3-calib-packs-v2-20260113`
- base: `openai/gpt-oss-20b`
- pruned: `sandeshrajx/gpt-oss-20b-reap-0.5-mxfp4`
- top_k: 4
- entropy_topk: 20

## seq_len=1024

| rank | pack | score_z | score | ΔPPL | ΔNLL | ΔNLL p | CC Δ (pp) | CC Δ CI | JS2D | Δ mean p (pp) | CC z | CC p |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| — | **GLOBAL_AVG** | — | — | +2.243 | — | — | -0.02 | — | 0.1832 | +5.68 | 0.00 | 9.99e-01 |
| 1 | reasoning_style_10k_v2 | +1.24 | +2.58 | +2.822 | +0.4287 | 9.83e-113 | +0.06 | -0.0000–+0.0012 | 0.1351 | +6.49 | 1.89 | 5.92e-02 |
| 2 | tool_agentic_10k_v6 | +0.67 | +1.39 | +2.299 | +0.3356 | 2.07e-18 | -0.15 | -0.0028–-0.0002 | 0.2955 | +6.92 | -2.20 | 2.77e-02 |
| 3 | UNION | -0.66 | -1.37 | +2.041 | +0.3089 | 2.74e-47 | -0.02 | -0.0011–+0.0006 | 0.1618 | +5.16 | -0.53 | 5.96e-01 |
| 4 | calib_prompt_10000_v2 | -1.25 | -2.61 | +1.808 | +0.2701 | 1.53e-39 | +0.03 | -0.0003–+0.0008 | 0.1405 | +4.15 | 0.85 | 3.96e-01 |

Worst pack (by score): `reasoning_style_10k_v2` | score_z=+1.24 | score=+2.58 | ΔPPL=+2.822 | ΔNLL=+0.4287 | CC Δ=+0.06 pp

## seq_len=2048

| rank | pack | score_z | score | ΔPPL | ΔNLL | ΔNLL p | CC Δ (pp) | CC Δ CI | JS2D | Δ mean p (pp) | CC z | CC p |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| — | **GLOBAL_AVG** | — | — | +1.904 | — | — | +0.00 | — | 0.1166 | +5.91 | 0.31 | 7.59e-01 |
| 1 | tool_agentic_10k_v6 | +1.29 | +3.04 | +2.260 | +0.3743 | 3.48e-41 | -0.06 | -0.0012–-0.0001 | 0.2166 | +7.64 | -2.18 | 2.90e-02 |
| 2 | reasoning_style_10k_v2 | +0.56 | +1.31 | +2.051 | +0.3730 | 4.75e-198 | +0.07 | +0.0003–+0.0011 | 0.0791 | +6.00 | 3.21 | 1.33e-03 |
| 3 | UNION | -0.54 | -1.27 | +1.798 | +0.3486 | 5.38e-140 | -0.01 | -0.0003–+0.0001 | 0.0905 | +5.69 | -0.85 | 3.94e-01 |
| 4 | calib_prompt_10000_v2 | -1.31 | -3.08 | +1.507 | +0.2744 | 1.21e-92 | +0.02 | -0.0002–+0.0005 | 0.0801 | +4.30 | 1.05 | 2.92e-01 |

Worst pack (by score): `tool_agentic_10k_v6` | score_z=+1.29 | score=+3.04 | ΔPPL=+2.260 | ΔNLL=+0.3743 | CC Δ=-0.06 pp

## Interpretation

- `score_z` is the z-score of the combined degradation (ΔPPL + CC Δ + JS2D + Δ mean p) across packs.
- `CC p` is a two-sided p-value from the CC z-score (smaller means more significant).
- For quality risk: look for **positive ΔPPL**, **positive CC Δ**, **positive JS2D**, and **mean p drop** together.
