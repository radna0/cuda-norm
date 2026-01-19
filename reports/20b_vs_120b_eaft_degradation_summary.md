# EAFT degradation summary (base vs pruned)

- dataset: `radna0/harmony-qwen3-calib-packs-v2-20260113`
- base: `openai/gpt-oss-20b`
- pruned: `openai/gpt-oss-120b`
- top_k: 4
- entropy_topk: 20

## seq_len=1024

| rank | pack | score_z | score | ΔPPL | ΔNLL | ΔNLL p | CC Δ (pp) | CC Δ CI | JS2D | Δ mean p (pp) | CC z | CC p |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| — | **GLOBAL_AVG** | — | — | +0.746 | — | — | +0.70 | — | 0.2988 | -1.73 | 3.17 | 1.52e-03 |
| 1 | tool_agentic_10k_v6 | +1.69 | +6.53 | +4.238 | +0.5703 | 2.27e-11 | +1.76 | +0.0115–+0.0236 | 0.3650 | +2.88 | 5.67 | 1.42e-08 |
| 2 | calib_prompt_10000_v2 | -0.21 | -0.82 | -0.285 | -0.0542 | 2.74e-01 | +0.75 | +0.0040–+0.0110 | 0.3150 | -3.69 | 4.17 | 3.00e-05 |
| 3 | UNION | -0.69 | -2.67 | -0.445 | -0.0797 | 4.28e-02 | +0.14 | -0.0010–+0.0038 | 0.2635 | -2.90 | 1.16 | 2.47e-01 |
| 4 | reasoning_style_10k_v2 | -0.79 | -3.05 | -0.525 | -0.0964 | 8.92e-03 | +0.17 | -0.0003–+0.0038 | 0.2516 | -3.23 | 1.68 | 9.29e-02 |

Worst pack (by score): `tool_agentic_10k_v6` | score_z=+1.69 | score=+6.53 | ΔPPL=+4.238 | ΔNLL=+0.5703 | CC Δ=+1.76 pp

## seq_len=2048

| rank | pack | score_z | score | ΔPPL | ΔNLL | ΔNLL p | CC Δ (pp) | CC Δ CI | JS2D | Δ mean p (pp) | CC z | CC p |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| — | **GLOBAL_AVG** | — | — | +0.971 | — | — | +0.70 | — | 0.2109 | -2.56 | 5.49 | 4.07e-08 |
| 1 | tool_agentic_10k_v6 | +1.69 | +6.72 | +4.750 | +0.6090 | 1.76e-22 | +1.75 | +0.0132–+0.0219 | 0.2914 | +2.02 | 7.87 | 3.61e-15 |
| 2 | UNION | -0.24 | -0.96 | +0.030 | +0.0064 | 8.29e-01 | +0.71 | +0.0051–+0.0090 | 0.1991 | -3.35 | 7.08 | 1.41e-12 |
| 3 | calib_prompt_10000_v2 | -0.63 | -2.52 | -0.247 | -0.0655 | 1.47e-02 | +0.11 | +0.0004–+0.0018 | 0.1840 | -3.86 | 3.21 | 1.34e-03 |
| 4 | reasoning_style_10k_v2 | -0.82 | -3.24 | -0.650 | -0.1506 | 2.71e-10 | +0.25 | +0.0012–+0.0038 | 0.1692 | -5.05 | 3.79 | 1.49e-04 |

Worst pack (by score): `tool_agentic_10k_v6` | score_z=+1.69 | score=+6.72 | ΔPPL=+4.750 | ΔNLL=+0.6090 | CC Δ=+1.75 pp

## Interpretation

- `score_z` is the z-score of the combined degradation (ΔPPL + CC Δ + JS2D + Δ mean p) across packs.
- `CC p` is a two-sided p-value from the CC z-score (smaller means more significant).
- For quality risk: look for **positive ΔPPL**, **positive CC Δ**, **positive JS2D**, and **mean p drop** together.
