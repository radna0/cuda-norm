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
