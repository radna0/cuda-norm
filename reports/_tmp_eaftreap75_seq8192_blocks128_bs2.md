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
