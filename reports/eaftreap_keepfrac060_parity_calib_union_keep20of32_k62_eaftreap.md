# EAFT parity summary (pair)

- Left: `openai/gpt-oss-20b`
- Right: `calib_union_keep20of32_k62_eaftreap`
- Gates: `near-lossless-v1` (`harmony/cuda-norm/pruning/near_lossless_gates.json`)

## Hero (UNION)

- seq=1024: ΔPPL=+0.485 | ΔCC=+0.053pp | Δmean_p=-0.0210 | JS2D=0.0252
- seq=2048: ΔPPL=+0.383 | ΔCC=+0.065pp | Δmean_p=-0.0216 | JS2D=0.0128

## Gate Result: FAIL

- Rule: |ΔPPL|<=min(abs=0.25, rel=0.05) and |ΔCC|<=0.002 and |Δmean_p|<=0.02 and JS2D<=0.02
- Failures:
  - seq=1024 ppl_ok=False cc_ok=True mean_p_ok=False js_ok=False
  - seq=2048 ppl_ok=False cc_ok=True mean_p_ok=False js_ok=True

## Full Table (Right vs Left)

| pack | seq | ppl_L | ppl_R | Δppl | cc_L | cc_R | Δcc (pp) | mean_p_L | mean_p_R | Δmean_p | JS2D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UNION | 1024 | 5.718 | 6.203 | +0.485 | 0.197 | 0.249 | +0.053 | 0.4468 | 0.4258 | -0.0210 | 0.0252 |
| UNION | 2048 | 4.671 | 5.054 | +0.383 | 0.104 | 0.168 | +0.065 | 0.4868 | 0.4652 | -0.0216 | 0.0128 |
| calib_prompt_10000_v2 | 1024 | 5.513 | 5.957 | +0.444 | 0.115 | 0.158 | +0.043 | 0.4612 | 0.4419 | -0.0192 | 0.0227 |
| calib_prompt_10000_v2 | 2048 | 4.441 | 4.719 | +0.278 | 0.073 | 0.113 | +0.040 | 0.5062 | 0.4891 | -0.0171 | 0.0118 |
| reasoning_style_10k_v2 | 1024 | 5.523 | 6.077 | +0.554 | 0.227 | 0.298 | +0.071 | 0.4455 | 0.4218 | -0.0237 | 0.0212 |
| reasoning_style_10k_v2 | 2048 | 4.468 | 4.932 | +0.464 | 0.096 | 0.186 | +0.090 | 0.4898 | 0.4644 | -0.0253 | 0.0117 |
| tool_agentic_10k_v6 | 1024 | 5.561 | 5.643 | +0.082 | 0.280 | 0.242 | -0.038 | 0.5014 | 0.4780 | -0.0234 | 0.0882 |
| tool_agentic_10k_v6 | 2048 | 4.778 | 4.723 | -0.055 | 0.215 | 0.163 | -0.052 | 0.5374 | 0.5170 | -0.0204 | 0.0460 |
