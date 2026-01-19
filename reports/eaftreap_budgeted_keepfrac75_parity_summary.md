 # EAFT parity summary (pair)

- Left: `openai/gpt-oss-20b`
- Right: `eaftreap_budgeted_keepfrac075`
- Gates: `near-lossless-v1` (`/home/kojoe/harmony/cuda-norm/pruning/near_lossless_gates.json`)

## Hero (UNION)

- seq=1024: ΔPPL=+0.150 | ΔCC=-0.034pp | Δmean_p=-0.0081 | JS2D=0.0233

## Gate Result: FAIL

- Rule: |ΔPPL|<=min(abs=0.25, rel=0.05) and |ΔCC|<=0.002 and |Δmean_p|<=0.02 and JS2D<=0.02
- Failures:
  - seq=1024 ppl_ok=True cc_ok=True mean_p_ok=True js_ok=False

## Full Table (Right vs Left)

| pack | seq | ppl_L | ppl_R | Δppl | cc_L | cc_R | Δcc (pp) | mean_p_L | mean_p_R | Δmean_p | JS2D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UNION | 1024 | 5.725 | 5.874 | +0.150 | 0.200 | 0.166 | -0.034 | 0.4471 | 0.4390 | -0.0081 | 0.0233 |
| calib_prompt_10000_v2 | 1024 | 5.532 | 5.650 | +0.118 | 0.119 | 0.121 | +0.002 | 0.4611 | 0.4545 | -0.0067 | 0.0213 |
| reasoning_style_10k_v2 | 1024 | 5.548 | 5.748 | +0.200 | 0.230 | 0.241 | +0.011 | 0.4452 | 0.4356 | -0.0096 | 0.0196 |
| tool_agentic_10k_v6 | 1024 | 5.503 | 5.327 | -0.177 | 0.267 | 0.232 | -0.034 | 0.5024 | 0.4980 | -0.0044 | 0.0866 |
