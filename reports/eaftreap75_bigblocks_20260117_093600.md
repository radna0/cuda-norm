# EAFT parity summary (pair)

- Left: `openai/gpt-oss-20b`
- Right: `eaftreap_budgeted_keepfrac075`
- Gates: `near-lossless-v1` (`/home/kojoe/harmony/cuda-norm/pruning/near_lossless_gates.json`)

## Hero (UNION)

- seq=1024: ΔPPL=+0.160 | ΔCC=+0.004pp | Δmean_p=-0.0092 | JS2D=0.0059

## Gate Result: PASS

- Rule: |ΔPPL|<=min(abs=0.25, rel=0.05) and |ΔCC|<=0.002 and |Δmean_p|<=0.02 and JS2D<=0.02

## Full Table (Right vs Left)

| pack | seq | ppl_L | ppl_R | Δppl | cc_L | cc_R | Δcc (pp) | mean_p_L | mean_p_R | Δmean_p | JS2D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UNION | 1024 | 5.465 | 5.625 | +0.160 | 0.170 | 0.174 | +0.004 | 0.4615 | 0.4523 | -0.0092 | 0.0059 |
| calib_prompt_10000_v2 | 1024 | 5.244 | 5.368 | +0.124 | 0.153 | 0.163 | +0.010 | 0.4783 | 0.4705 | -0.0078 | 0.0053 |
| reasoning_style_10k_v2 | 1024 | 5.442 | 5.644 | +0.202 | 0.188 | 0.241 | +0.053 | 0.4521 | 0.4418 | -0.0103 | 0.0052 |
| tool_agentic_10k_v6 | 1024 | 5.428 | 5.225 | -0.203 | 0.172 | 0.144 | -0.028 | 0.5090 | 0.5053 | -0.0036 | 0.0215 |
