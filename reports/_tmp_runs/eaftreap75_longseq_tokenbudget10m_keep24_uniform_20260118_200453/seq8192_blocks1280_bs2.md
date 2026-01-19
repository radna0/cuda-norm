# EAFT parity summary (pair)

- Left: `openai/gpt-oss-20b`
- Right: `calib_union_keep24of32_k75_eaftreap`
- Gates: `near-lossless-v1` (`harmony/cuda-norm/pruning/near_lossless_gates.json`)

## Full Table (Right vs Left)

| pack | seq | ppl_L | ppl_R | Δppl | cc_L | cc_R | Δcc (pp) | mean_p_L | mean_p_R | Δmean_p | JS2D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UNION | 8192 | 2.998 | 3.112 | +0.114 | 0.046 | 0.042 | -0.004 | 0.6049 | 0.5959 | -0.0090 | 0.0005 |
| calib_prompt_10000_v2 | 8192 | 2.623 | 2.667 | +0.045 | 0.042 | 0.038 | -0.004 | 0.6502 | 0.6455 | -0.0047 | 0.0003 |
| reasoning_style_10k_v2 | 8192 | 2.089 | 2.186 | +0.097 | 0.027 | 0.028 | +0.001 | 0.7101 | 0.7006 | -0.0095 | 0.0005 |
| tool_agentic_10k_v6 | 8192 | 3.718 | 3.800 | +0.082 | 0.051 | 0.049 | -0.002 | 0.5308 | 0.5243 | -0.0065 | 0.0004 |
