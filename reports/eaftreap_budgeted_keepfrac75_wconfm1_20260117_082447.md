# EAFT parity summary (pair)

- Left: `openai/gpt-oss-20b`
- Right: `eaftreap_budgeted_keepfrac075`
- Gates: `near-lossless-v1` (`/home/kojoe/harmony/cuda-norm/pruning/near_lossless_gates.json`)

## Hero (UNION)

- seq=1024: ΔPPL=+0.207 | ΔCC=-0.026pp | Δmean_p=-0.0092 | JS2D=0.0236

## Gate Result: FAIL

- Rule: |ΔPPL|<=min(abs=0.25, rel=0.05) and |ΔCC|<=0.002 and |Δmean_p|<=0.02 and JS2D<=0.02
- Failures:
  - seq=1024 ppl_ok=True cc_ok=True mean_p_ok=True js_ok=False

## Full Table (Right vs Left)

| pack | seq | ppl_L | ppl_R | Δppl | cc_L | cc_R | Δcc (pp) | mean_p_L | mean_p_R | Δmean_p | JS2D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UNION | 1024 | 5.725 | 5.931 | +0.207 | 0.200 | 0.174 | -0.026 | 0.4471 | 0.4379 | -0.0092 | 0.0236 |
| calib_prompt_10000_v2 | 1024 | 5.532 | 5.710 | +0.177 | 0.119 | 0.123 | +0.004 | 0.4611 | 0.4533 | -0.0079 | 0.0212 |
| reasoning_style_10k_v2 | 1024 | 5.548 | 5.775 | +0.227 | 0.230 | 0.256 | +0.026 | 0.4452 | 0.4353 | -0.0099 | 0.0206 |
| tool_agentic_10k_v6 | 1024 | 5.503 | 5.514 | +0.010 | 0.267 | 0.263 | -0.004 | 0.5024 | 0.4923 | -0.0101 | 0.0831 |
