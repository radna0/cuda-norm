# EAFT parity summary (pair)

- Left: `openai/gpt-oss-20b`
- Right: `eaftreap_budgeted_keepfrac075`
- Gates: `near-lossless-v1` (`/home/kojoe/harmony/cuda-norm/pruning/near_lossless_gates.json`)

## Hero (UNION)

- seq=1024: ΔPPL=+0.202 | ΔCC=-0.028pp | Δmean_p=-0.0091 | JS2D=0.0238

## Gate Result: FAIL

- Rule: |ΔPPL|<=min(abs=0.25, rel=0.05) and |ΔCC|<=0.002 and |Δmean_p|<=0.02 and JS2D<=0.02
- Failures:
  - seq=1024 ppl_ok=True cc_ok=True mean_p_ok=True js_ok=False

## Full Table (Right vs Left)

| pack | seq | ppl_L | ppl_R | Δppl | cc_L | cc_R | Δcc (pp) | mean_p_L | mean_p_R | Δmean_p | JS2D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UNION | 1024 | 5.725 | 5.927 | +0.202 | 0.200 | 0.172 | -0.028 | 0.4471 | 0.4380 | -0.0091 | 0.0238 |
| calib_prompt_10000_v2 | 1024 | 5.532 | 5.702 | +0.170 | 0.119 | 0.119 | +0.000 | 0.4611 | 0.4535 | -0.0076 | 0.0213 |
| reasoning_style_10k_v2 | 1024 | 5.548 | 5.770 | +0.222 | 0.230 | 0.261 | +0.031 | 0.4452 | 0.4354 | -0.0098 | 0.0195 |
| tool_agentic_10k_v6 | 1024 | 5.503 | 5.522 | +0.018 | 0.267 | 0.248 | -0.019 | 0.5024 | 0.4914 | -0.0110 | 0.0850 |
