# EAFT parity (ctx=65K/131K tokenbudget ~1M) — keep24/32 EAFT-REAP

- Base: `openai/gpt-oss-20b`
- Pruned: `radna0/gptoss20b-keep24of32-k75-eaftreap`
- Pruned path: ``
- top_k: 4 (unchanged)

## Matrix results

---

# EAFT parity summary (pair)

- Left: `openai/gpt-oss-20b`
- Right: `radna0/gptoss20b-keep24of32-k75-eaftreap`
- Gates: `near-lossless-v1` (`harmony/cuda-norm/pruning/near_lossless_gates.json`)

## Full Table (Right vs Left)

| pack | seq | ppl_L | ppl_R | Δppl | cc_L | cc_R | Δcc (pp) | mean_p_L | mean_p_R | Δmean_p | JS2D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UNION | 131072 | 3.526 | 3.615 | +0.089 | 0.057 | 0.099 | +0.042 | 0.5530 | 0.5459 | -0.0071 | 0.0029 |
| calib_prompt_10000_v2 | 131072 | 3.365 | 3.439 | +0.075 | 0.073 | 0.065 | -0.008 | 0.5721 | 0.5668 | -0.0053 | 0.0025 |
| reasoning_style_10k_v2 | 131072 | 2.787 | 2.892 | +0.104 | 0.023 | 0.024 | +0.001 | 0.6139 | 0.6053 | -0.0086 | 0.0025 |
| tool_agentic_10k_v6 | 131072 | 3.305 | 3.335 | +0.030 | 0.037 | 0.033 | -0.004 | 0.5808 | 0.5754 | -0.0054 | 0.0056 |

---

# EAFT parity summary (pair)

- Left: `openai/gpt-oss-20b`
- Right: `radna0/gptoss20b-keep24of32-k75-eaftreap`
- Gates: `near-lossless-v1` (`harmony/cuda-norm/pruning/near_lossless_gates.json`)

## Full Table (Right vs Left)

| pack | seq | ppl_L | ppl_R | Δppl | cc_L | cc_R | Δcc (pp) | mean_p_L | mean_p_R | Δmean_p | JS2D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UNION | 65536 | 3.539 | 3.630 | +0.090 | 0.054 | 0.097 | +0.043 | 0.5514 | 0.5441 | -0.0073 | 0.0029 |
| calib_prompt_10000_v2 | 65536 | 3.396 | 3.473 | +0.077 | 0.073 | 0.065 | -0.008 | 0.5696 | 0.5638 | -0.0058 | 0.0026 |
| reasoning_style_10k_v2 | 65536 | 2.788 | 2.893 | +0.105 | 0.022 | 0.024 | +0.002 | 0.6131 | 0.6044 | -0.0088 | 0.0025 |
| tool_agentic_10k_v6 | 65536 | 3.351 | 3.379 | +0.027 | 0.042 | 0.031 | -0.011 | 0.5775 | 0.5714 | -0.0060 | 0.0056 |

