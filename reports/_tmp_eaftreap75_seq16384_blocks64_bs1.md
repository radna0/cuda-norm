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
