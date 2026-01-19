# EAFT parity summary (pair)

- Left: `openai/gpt-oss-20b`
- Right: `eaftreap_keep24of32_k75_eaftreap`
- Gates: `near-lossless-v1` (`harmony/cuda-norm/pruning/near_lossless_gates.json`)

## Hero (UNION)

- seq=1024: ΔPPL=+1.235 | ΔCC=+0.043pp | Δmean_p=-0.0284 | JS2D=0.1082
- seq=2048: ΔPPL=+0.990 | ΔCC=+0.027pp | Δmean_p=-0.0290 | JS2D=0.0539

## Gate Result: FAIL

- Rule: |ΔPPL|<=min(abs=0.25, rel=0.05) and |ΔCC|<=0.002 and |Δmean_p|<=0.02 and JS2D<=0.02
- Failures:
  - seq=1024 ppl_ok=False cc_ok=True mean_p_ok=False js_ok=False
  - seq=2048 ppl_ok=False cc_ok=True mean_p_ok=False js_ok=False

## Full Table (Right vs Left)

| pack | seq | ppl_L | ppl_R | Δppl | cc_L | cc_R | Δcc (pp) | mean_p_L | mean_p_R | Δmean_p | JS2D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UNION | 1024 | 5.651 | 6.885 | +1.235 | 0.240 | 0.283 | +0.043 | 0.4473 | 0.4189 | -0.0284 | 0.1082 |
| UNION | 2048 | 4.332 | 5.322 | +0.990 | 0.029 | 0.056 | +0.027 | 0.5103 | 0.4813 | -0.0290 | 0.0539 |
| calib_prompt_10000_v2 | 1024 | 5.821 | 7.180 | +1.359 | 0.101 | 0.164 | +0.063 | 0.4547 | 0.4269 | -0.0278 | 0.0946 |
| calib_prompt_10000_v2 | 2048 | 4.768 | 5.861 | +1.093 | 0.082 | 0.146 | +0.063 | 0.4811 | 0.4536 | -0.0275 | 0.0529 |
| reasoning_style_10k_v2 | 1024 | 5.288 | 6.693 | +1.405 | 0.102 | 0.211 | +0.109 | 0.4604 | 0.4300 | -0.0304 | 0.0872 |
| reasoning_style_10k_v2 | 2048 | 4.545 | 5.775 | +1.229 | 0.083 | 0.230 | +0.148 | 0.4792 | 0.4463 | -0.0329 | 0.0483 |
| tool_agentic_10k_v6 | 1024 | 5.790 | 6.348 | +0.558 | 0.321 | 0.307 | -0.013 | 0.4846 | 0.4630 | -0.0215 | 0.2274 |
| tool_agentic_10k_v6 | 2048 | 4.944 | 5.401 | +0.457 | 0.165 | 0.158 | -0.007 | 0.5280 | 0.5025 | -0.0255 | 0.1557 |
