# EAFT parity summary (pair)

- Left: `openai/gpt-oss-20b`
- Right: `eaftreap_budgeted_keepfrac075`
- Gates: `near-lossless-v1` (`/home/kojoe/harmony/cuda-norm/pruning/near_lossless_gates.json`)

## Hero (UNION)

- seq=1024: ΔPPL=+1.051 | ΔCC=+0.068pp | Δmean_p=-0.0232 | JS2D=0.0543

## Gate Result: FAIL

- Rule: |ΔPPL|<=min(abs=0.25, rel=0.05) and |ΔCC|<=0.002 and |Δmean_p|<=0.02 and JS2D<=0.02
- Failures:
  - seq=1024 ppl_ok=False cc_ok=True mean_p_ok=False js_ok=False

## Full Table (Right vs Left)

| pack | seq | ppl_L | ppl_R | Δppl | cc_L | cc_R | Δcc (pp) | mean_p_L | mean_p_R | Δmean_p | JS2D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UNION | 1024 | 5.245 | 6.295 | +1.051 | 0.057 | 0.125 | +0.068 | 0.4748 | 0.4516 | -0.0232 | 0.0543 |
| calib_prompt_10000_v2 | 1024 | 5.932 | 6.963 | +1.031 | 0.140 | 0.263 | +0.123 | 0.4436 | 0.4248 | -0.0189 | 0.0512 |
| reasoning_style_10k_v2 | 1024 | 5.500 | 6.690 | +1.190 | 0.218 | 0.302 | +0.085 | 0.4450 | 0.4197 | -0.0253 | 0.0494 |
| tool_agentic_10k_v6 | 1024 | 5.590 | 6.318 | +0.729 | 0.209 | 0.195 | -0.014 | 0.4966 | 0.4742 | -0.0224 | 0.1515 |
