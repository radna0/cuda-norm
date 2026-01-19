# Near‑Lossless Acceptance Gates (v1)

This project’s pruning goal is **quality preservation first**. “Near‑lossless” means the pruned/modified model must not show measurable regression on our curated evaluation regime.

## Canonical eval regime

- Dataset: `radna0/harmony-qwen3-calib-packs-v2-20260113`
- Packs: `reasoning_style_10k_v2`, `tool_agentic_10k_v6`, `calib_prompt_10000_v2`, and `UNION`
- Metrics are computed in the **packed completion-only** regime:
  - completion-only tokens are the `<|start|>assistant<|message|> ...` spans
  - rows are concatenated with EOS and then chunked into fixed `seq_len` blocks

## Gates (strict, “v1”)

Defined in `harmony/cuda-norm/pruning/near_lossless_gates.json`.

For a candidate model (right) compared against the baseline (left), we require:

- `|ΔPPL| <= max(max_abs_delta_ppl, base_ppl * max_rel_delta_ppl)`
- `|ΔCC_rate| <= max_abs_delta_cc_rate` (CC computed under *baseline* thresholds)
- `|Δmean_prob| <= max_abs_delta_mean_prob`
- `JS2D <= max_js2d` (distribution shift of the 2D p/entropy landscape)

Primary check: `pack=UNION` at `seq_len ∈ {1024, 2048}`.

## Where it is enforced

The dynamic dashboard (`harmony/cuda-norm/reports/eaft_dynamic_compare.html`) displays a **Near‑Lossless Gate PASS/FAIL** badge in the hero row using these exact thresholds.

