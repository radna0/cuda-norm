# DFlash success metrics (what “5–6×” means)

We consider DFlash “successful” when it yields sustained **decode throughput** speedups under the real serving regime, not just prefill.

## Primary KPI (what we optimize)

**Decode throughput speedup**
- Metric: `speedup_x = (DFLASH total output tok/s) / (baseline total output tok/s)`
- Workload: `max_new_tokens=2048` (and later 8192/65536/131072), `temperature=0`, `top_k=1`, `top_p=1`
- Concurrency: sweep (1, 2, 4, 8, 16, 32) and report the peak stable point (no OOM/paging/timeouts)

Target:
- **5–6×** speedup at low concurrency (1–4) is the stretch goal.
- At high concurrency, speedups usually compress; we still want a clear win.

## Supporting metrics (must be reported)

**Acceptance rate**
- From server meta: `spec_accept_length_mean / block_size`
- Higher acceptance rate generally ⇒ higher speedup (but depends on verification cost).

**Total output tok/s**
- Total completions tokens across all concurrent requests divided by wall time.

**Per-stream tok/s**
- `total_tok_s / concurrency` (helps detect “only scales with batching” artifacts).

**Correctness / quality guardrails**
- Greedy match sanity: DFlash must not degrade outputs under greedy decoding beyond small numeric noise (for the same target model).
- For trained drafts, we also track PPL/EAFT diagnostics separately (quality dashboards), but DFlash speedup work is gated first on correctness.

## Reference benchmark script

GPU benchmark entrypoint (Kaggle):
- `harmony/cuda-norm/scripts/dflash_gptoss20b_bench_sglang_kaggle.py`

It reports:
- baseline + dflash total tok/s
- `speedup_x`
- `accept_rate_est`

