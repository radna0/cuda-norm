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
