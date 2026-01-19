# EAFT parity (long-seq tokenbudget ~1M) — keep24/32 EAFT-REAP

- Base: `openai/gpt-oss-20b`
- Pruned: `calib_union_keep24of32_k75_eaftreap`
- Pruned path: `/kaggle/working/artifacts/harmony_cuda_norm/20b_pruned_models_eaftreap/calib_union_keep24of32_k75_eaftreap`
- top_k: 4 (unchanged)

## Matrix results

---

# EAFT parity summary (pair)

- Left: `openai/gpt-oss-20b`
- Right: `calib_union_keep24of32_k75_eaftreap`
- Gates: `near-lossless-v1` (`harmony/cuda-norm/pruning/near_lossless_gates.json`)

## Full Table (Right vs Left)

| pack | seq | ppl_L | ppl_R | Δppl | cc_L | cc_R | Δcc (pp) | mean_p_L | mean_p_R | Δmean_p | JS2D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UNION | 16384 | 3.624 | 3.723 | +0.099 | 0.090 | 0.080 | -0.009 | 0.5440 | 0.5359 | -0.0080 | 0.0030 |
| calib_prompt_10000_v2 | 16384 | 3.522 | 3.612 | +0.090 | 0.071 | 0.063 | -0.008 | 0.5599 | 0.5530 | -0.0069 | 0.0026 |
| reasoning_style_10k_v2 | 16384 | 2.850 | 2.962 | +0.112 | 0.020 | 0.022 | +0.002 | 0.6059 | 0.5965 | -0.0093 | 0.0026 |
| tool_agentic_10k_v6 | 16384 | 3.535 | 3.558 | +0.023 | 0.052 | 0.037 | -0.015 | 0.5646 | 0.5579 | -0.0067 | 0.0057 |

---

# EAFT parity summary (pair)

- Left: `openai/gpt-oss-20b`
- Right: `calib_union_keep24of32_k75_eaftreap`
- Gates: `near-lossless-v1` (`harmony/cuda-norm/pruning/near_lossless_gates.json`)

## Full Table (Right vs Left)

| pack | seq | ppl_L | ppl_R | Δppl | cc_L | cc_R | Δcc (pp) | mean_p_L | mean_p_R | Δmean_p | JS2D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UNION | 4096 | 4.023 | 4.147 | +0.124 | 0.109 | 0.096 | -0.013 | 0.5204 | 0.5115 | -0.0089 | 0.0030 |
| calib_prompt_10000_v2 | 4096 | 3.917 | 4.032 | +0.115 | 0.061 | 0.107 | +0.045 | 0.5357 | 0.5277 | -0.0080 | 0.0028 |
| reasoning_style_10k_v2 | 4096 | 3.181 | 3.317 | +0.136 | 0.017 | 0.017 | -0.000 | 0.5804 | 0.5702 | -0.0101 | 0.0026 |
| tool_agentic_10k_v6 | 4096 | 3.982 | 4.008 | +0.026 | 0.123 | 0.095 | -0.028 | 0.5390 | 0.5318 | -0.0072 | 0.0059 |

---

# EAFT parity summary (pair)

- Left: `openai/gpt-oss-20b`
- Right: `calib_union_keep24of32_k75_eaftreap`
- Gates: `near-lossless-v1` (`harmony/cuda-norm/pruning/near_lossless_gates.json`)

## Full Table (Right vs Left)

| pack | seq | ppl_L | ppl_R | Δppl | cc_L | cc_R | Δcc (pp) | mean_p_L | mean_p_R | Δmean_p | JS2D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UNION | 8192 | 3.758 | 3.865 | +0.107 | 0.080 | 0.070 | -0.009 | 0.5352 | 0.5267 | -0.0084 | 0.0030 |
| calib_prompt_10000_v2 | 8192 | 3.662 | 3.761 | +0.099 | 0.067 | 0.058 | -0.009 | 0.5505 | 0.5429 | -0.0075 | 0.0027 |
| reasoning_style_10k_v2 | 8192 | 2.951 | 3.071 | +0.120 | 0.018 | 0.019 | +0.001 | 0.5970 | 0.5874 | -0.0096 | 0.0026 |
| tool_agentic_10k_v6 | 8192 | 3.698 | 3.720 | +0.022 | 0.058 | 0.090 | +0.032 | 0.5544 | 0.5475 | -0.0070 | 0.0058 |

