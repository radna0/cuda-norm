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
