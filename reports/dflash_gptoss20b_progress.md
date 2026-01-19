# DFlash for GPT‑OSS‑20B — Progress + Next Actions

## What Works Now

- **Target model**: `openai/gpt-oss-20b` (loaded BF16 on `H100:1`).
- **Draft model**: `harmony/cuda-norm/dflash_gptoss/modeling_gptoss_dflash.py` implements a GPT‑OSS‑compatible DFlash draft (non‑causal cross‑attn over `[target_hidden || noise_block]`).
- **Lossless greedy correctness**: speculative decode matches target greedy decode on a smoke prompt.
  - Log: `harmony/cuda-norm/unsloth_logs/dflash_gptoss20b_smoke_20260114_134307.log`

## Checkpoints Produced

- Phase‑A training run (H100):
  - Run dir: `/root/model/dflash_gptoss20b/20260114_143401`
  - Checkpoints: `step_000050`, `step_000100`, `step_000150`, `step_000200`
  - Log: `harmony/cuda-norm/unsloth_logs/dflash_gptoss20b_train2_20260114_143325.log`

## Benchmark Harness (KV‑Cache Baseline)

`harmony/cuda-norm/modal/dflash_gptoss20b_benchmark.py` now benchmarks against an explicit KV‑cache decode loop (not `generate()`), so the “speedup” is comparable to our speculative loop.

### Results (decode to `max_new_tokens=256`, `block_size=8`)

- **Untrained-ish checkpoint (step_000005)**:
  - `speedup_mean ≈ 0.834×` (spec slower, expected)
  - `acceptance_hist = {1: 1014}` (accepts ~1 token per block)
  - JSON: `harmony/cuda-norm/artifacts/dflash_bench/bench_20260114_143216.json`
  - Log: `harmony/cuda-norm/unsloth_logs/dflash_gptoss20b_bench3_20260114_143044.log`

- **Trained checkpoint (step_000200)**:
  - `speedup_mean ≈ 0.819×` (still slower)
  - `acceptance_hist = {1: 1877, 2: 27}` (tiny improvement; still basically 1)
  - JSON: `harmony/cuda-norm/artifacts/dflash_bench/bench_20260114_144320.json`
  - Log: `harmony/cuda-norm/unsloth_logs/dflash_gptoss20b_bench_step200_20260114_144056.log`

## Interpretation (Why No Speedup Yet)

Speculative decoding only wins when the draft’s **accepted prefix length** is >1 for a meaningful fraction of steps. Right now, the draft is still effectively a “1‑token proposer”, so it adds compute without reducing target work.

## Next Actions (to make it ship‑worthy)

1. **Scale training** to reach non‑trivial acceptance (goal: median ≥2, p90 ≥4 at `block_size=8`).
   - Increase steps (1k → 10k), keep seq_len=4096 for now, and log acceptance on a fixed prompt set every N steps.
2. **Tune architecture/target hidden selection**:
   - Increase draft depth (e.g., 8 layers) and/or widen MLP ratio; verify target layer feature extraction matches DFlash assumptions.
3. **Run “real” decode regime**:
   - Benchmark `max_new_tokens=2048` (prefill amortized) and report total tok/s + acceptance histograms.
4. **Optional: B200 path**
   - Once acceptance improves on H100, rerun identical benchmark on B200 to validate speedups in the hardware regime the DFlash repo claims.

