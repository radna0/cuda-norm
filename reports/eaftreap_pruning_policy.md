# EAFT‑REAP Pruning Policy (correctness‑aware expert ranking)

Goal: avoid REAP’s blind spot (magnitude ≠ correctness) by conditioning expert saliency on token correctness signals.

## Definitions

Per token (t), on the packed completion-only regime:
- `p_t`: probability of the reference token
- `H_t`: predictive entropy (Top‑K approximation, `K=20`)

Define regions using dataset-driven quantiles:
- **confident conflict**: `p_t <= p_q` and `H_t <= H_q` (default `q=0.15`)
- **uncertain**: `p_t <= p_q` and `H_t >= H_hi` (default `H_hi=0.85`)
- **good**: everything else (or optionally `p_t >= p_hi && H_t <= H_mid`)

## EAFT‑REAP expert saliency (signed)

For each MoE layer and each selected expert `j` (top‑k routing), compute only on selected tokens:

`S_j = mean_t [ w_t * g_{j,t} * ||f_{j}(x_t)||_2 ]`

Where:
- `g_{j,t}` is the router weight/prob for expert `j` on token `t`
- `||f_j(x_t)||_2` is the expert output norm (selected experts only)
- `w_t` depends on the token region:
  - `w_good = +1.0`
  - `w_uncertain = +0.25` (do not penalize hard tokens by default)
  - `w_conflict = -2.0` (penalize experts that drive confident conflicts)

We rank experts per layer by `S_j` (descending) and keep the top‑N.

## Current findings (20B)

At keep_frac=0.50:
- EAFT‑REAP‑50 is slightly better than HF REAP‑0.5 on curated UNION packs, but both fail strict “near‑lossless” gates.
- Decode throughput at batch=32 improves modestly (≈ +3–6%) vs base on Kaggle / SGLang.

Sources:
- Quality: `harmony/cuda-norm/reports/eaftreap_quality_summary.md`
- Decode: `harmony/cuda-norm/reports/eaftreap_decode_throughput_summary.md`

## Recommendation (toward near‑lossless)

1. Treat EAFT‑REAP as the **ranking signal**, but dial back prune aggressiveness:
   - start with keep_frac in {0.80, 0.90, 0.95} (per-layer caps allowed)
2. Evaluate against strict gates on curated packs (UNION @ 1024/2048):
   - accept only candidates that PASS near‑lossless thresholds
3. Only after quality passes, explore `top_k=2` as a compute lever:
   - measure decode throughput at max stable batch
   - re-check EAFT CC‑rate drift (top_k reduction can amplify confident conflicts)

