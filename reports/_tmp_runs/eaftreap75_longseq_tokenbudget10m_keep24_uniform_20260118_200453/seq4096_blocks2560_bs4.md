# EAFT parity summary (pair)

- Left: `openai/gpt-oss-20b`
- Right: `calib_union_keep24of32_k75_eaftreap`
- Gates: `near-lossless-v1` (`harmony/cuda-norm/pruning/near_lossless_gates.json`)

## Full Table (Right vs Left)

| pack | seq | ppl_L | ppl_R | Δppl | cc_L | cc_R | Δcc (pp) | mean_p_L | mean_p_R | Δmean_p | JS2D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UNION | 4096 | 3.189 | 3.321 | +0.132 | 0.041 | 0.037 | -0.004 | 0.5905 | 0.5808 | -0.0097 | 0.0005 |
| calib_prompt_10000_v2 | 4096 | 2.767 | 2.817 | +0.050 | 0.039 | 0.035 | -0.004 | 0.6373 | 0.6324 | -0.0050 | 0.0003 |
| reasoning_style_10k_v2 | 4096 | 2.228 | 2.341 | +0.113 | 0.025 | 0.026 | +0.001 | 0.6944 | 0.6841 | -0.0103 | 0.0005 |
| tool_agentic_10k_v6 | 4096 | 4.004 | 4.097 | +0.093 | 0.082 | 0.076 | -0.006 | 0.5147 | 0.5079 | -0.0068 | 0.0004 |
