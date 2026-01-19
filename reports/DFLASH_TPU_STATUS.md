# DFlash TPU Status (GPT‑OSS‑20B, EasyDeL)

This report is the current end‑to‑end status of **TPU‑first DFlash draft training + speculative decode** for **GPT‑OSS‑20B** inside **EasyDeL source** (not a one‑off script).

## What Works (Correctness)

### 1) Teacher cache build (TPU)

- Cache dir: `/dev/shm/dflash_cache/build_cache_gptoss20b_ctx1024_b8_k4_n16_roll64_pos0_65k_131k_venvpy2_20260117_203545`
- Teacher snapshot: `/dev/shm/hf/hub/models--unsloth--gpt-oss-20b-BF16/snapshots/cc89b3e7fd423253264883a80a4fa5abc619649f`
- Teacher EasyDeL checkpoint used for cache build: `/dev/shm/easydel_teachers/gptoss20b_bf16_v2`
- Dataset: `radna0/harmony-qwen3-calib-packs-v2-20260113`
  - `packs/reasoning_style_10k_v2/reasoning_style_10k_v2.parquet`
  - `tool_agentic_10k_v6.parquet`
  - `packs/calib_prompt_10000_v2/calib_prompt_10000_v2.parquet`
  - `max_rows_per_pack=2000`
- Cache parameters (from `meta.json`):
  - `ctx_len_full=1024` (stored `ctx_len=1023`, anchor token is separate)
  - `block_size=8` (predict 7 tokens + 1 bonus)
  - `K=4` context feature layers: `[1, 8, 14, 21]`
  - `num_blocks=16`, `rollout_steps=64` → `num_samples=1024`
  - `position_offsets=[0, 65536, 131072]` (positional‑parity training without decoding to 65k/131k)
  - `page_size=128`, `hbm_utilization=0.20`, `prefill_chunk=256`

### 2) Multi‑token verify parity (TPU)

Parity check log is in:
- `harmony/cuda-norm/logs/tpu_dflash/autopipe_cachepos0_65k_131k_venvpy2_rerun_20260117_211718.nohup.log`

Key result (sample 0):
- `match_cached_targets_vs_verify=true`
- `verify_block0_changes_with_future=false` (causality OK)
- `match_1tok_vs_block0=true` on that sample (can be noisy across kernels; not strictly required as long as cached targets match verify)

### 3) Acceptance mechanics sanity (TPU)

We forced draft tokens to equal cache labels (the “gold” verify‑mode labels):
- Log: `harmony/cuda-norm/logs/tpu_dflash/accept_check_run2000_20260117_214859.log`
- Result: `accept_len=7` (max possible for `block_size=8`)

Conclusion: **accept/reject plumbing is correct**; low accept later is not a verifier bug.

## Training (Draft Model)

### Draft model configuration

Draft run config (saved with checkpoint):
- `/dev/shm/dflash-checkpoints/gptoss20b_dflash_sanity_bs32_s200_20260117_212531/run-2000/config.json`
- Key values:
  - `num_layers=8`
  - `hidden_size=2880`
  - `num_context_features=4`
  - `target_layer_ids=[1, 8, 14, 21]`
  - `block_size=8`
  - `qk_norm=true`, `remat=true`

### Training run

- Checkpoints:
  - `/dev/shm/dflash-checkpoints/gptoss20b_dflash_sanity_bs32_s200_20260117_212531/run-200`
  - `/dev/shm/dflash-checkpoints/gptoss20b_dflash_sanity_bs32_s200_20260117_212531/run-500`
  - `/dev/shm/dflash-checkpoints/gptoss20b_dflash_sanity_bs32_s200_20260117_212531/run-2000`
- Cache used: `.../build_cache_gptoss20b_ctx1024_b8_k4_n16_roll64_pos0_65k_131k_venvpy2_20260117_203545`
- Stable setting found:
  - `total_batch_size=32` on `dp=8` (larger `total_batch_size=160` caused silent failures earlier)

## Decode Benchmark (Real Regime: prompt 1024 + decode 2048)

Benchmark log:
- `harmony/cuda-norm/logs/tpu_dflash/esurge_bench_dflash_vs_baseline_run2000_20260117_213632.log`

Settings:
- `prompt_len=1024` (from cache sample 0)
- `max_new_tokens=2048`
- `block_size=8`
- `page_size=128`, `hbm_utilization=0.20`
- Draft: `run-2000`
- Baseline: target greedy decode

Results:
- Baseline `output_toks_per_s=13.97` (wall `146.6s`)
- DFlash `output_toks_per_s=6.87` (wall `298.3s`)
- DFlash acceptance:
  - `accept_rate=0.0428`
  - `accept_len_mean=0.299` (p50=0, p90=1)

Conclusion: **DFlash is currently slower than baseline** because the draft model is not accurate enough; it is proposing mostly wrong tokens, so verification rejects almost everything and we pay the DFlash overhead without acceptance gains.

## Why Speedup Is Not There Yet (Root Cause)

This run has only:
- `1024` cache samples
- `block_size-1 = 7` supervised tokens per sample
- Even at `2000` steps, we only see ~`448k` “visited tokens” in logs.

That is **orders of magnitude too little data** to train a high‑acceptance DFlash draft.

## Next Actions (Quality‑First Path to 5–6×)

1) **Scale cache size** (more samples) while staying within `/dev/shm` budget:
   - target: `4096–8192` samples minimum for next iteration
   - keep `position_offsets=[0,65536,131072]` so long‑position behavior is trained
2) **Train longer** on the larger cache:
   - target: `>= 10k` steps (then re‑bench at fixed checkpoints: 2k/5k/10k)
3) **Then** revisit DFlash speed engineering:
   - faster verify (TARGET_VERIFY‑like block verify kernel / overlap scheduling)
   - higher concurrency and batch decode benchmarks

