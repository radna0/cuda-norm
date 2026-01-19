# TPU DFlash Positional-Parity Baseline (Offline Accept Check)

This captures the **baseline** behavior before any fine-tuning on the new position-offset cache.

## Cache Built (pos offsets)

- Cache dir: `/dev/shm/dflash_cache/gptoss20b_ctx1024_b8_k4_n8_roll8_posoff_v5_20260117_142231`
- Offsets: `0, 4096, 65536, 131072` (saved in `ctx_pos_start_i32.npy`)
- Build log: `harmony/cuda-norm/logs/tpu_dflash/cache_build_posoff_v5_20260117_142231.log`

## Draft Checkpoint Used (good baseline on non-offset cache)

- Draft run: `/dev/shm/dflash-checkpoints/gptoss20b_dflash_ctx1024_b8_k4_bs64_s2000_resume200_v1_20260117_110932/run-1500`

## Offline Eval Harness

- Script: `harmony/cuda-norm/scripts/tpu_dflash_cache_eval_offline.py`
- Metric: `accept_len` = consecutive prefix match length over the `(block_size-1)` drafted tokens.

## Results

### A) Baseline cache (no offsets)

Cache: `/dev/shm/dflash_cache/gptoss20b_ctx1024_b8_k4_n16_roll64_v2_20260117_093347` (pos_start=0 only)

- `accept_len_mean ≈ 6.95` (max is 7 for `block_size=8`)
- `token_acc_mean ≈ 0.998`

### B) Position-offset cache (mixed offsets)

Cache: `/dev/shm/dflash_cache/gptoss20b_ctx1024_b8_k4_n8_roll8_posoff_v5_20260117_142231`

- `accept_len_mean ≈ 0.17`
- `token_acc_mean ≈ 0.04`
- Bucketed by `ctx_pos_start` (n=16 each):
  - `0`: accept_len_mean `0.0`
  - `4096`: accept_len_mean `0.0`
  - `65536`: accept_len_mean `0.0`
  - `131072`: accept_len_mean `0.69`

## Interpretation (what this proves)

- The current draft checkpoint is **not position-parity trained** for the new offset distribution and collapses in offline accept on that cache.
- Next step is to **fine-tune on the offset cache** and re-run the same offline eval to confirm accept_len recovers across offsets.

## After 200-step fine-tune on posoff cache (resume from step 1500 → 1700)

- Fine-tuned draft run:
  - `/dev/shm/dflash-checkpoints/gptoss20b_dflash_ctx1024_b8_k4_posoff_ft_v5_s1700_l8_20260117_145240/run-1700`

### POSOFF cache (64 samples)

- `accept_len_mean ≈ 2.59`, `token_acc_mean ≈ 0.72`
- By offset (n=16 each):
  - `0`: accept_len_mean `4.00`
  - `4096`: accept_len_mean `1.63`
  - `65536`: accept_len_mean `3.69`
  - `131072`: accept_len_mean `1.06`

### BASE cache (128 samples, sanity)

- `accept_len_mean ≈ 6.85`, `token_acc_mean ≈ 0.994` (still near-perfect; no major regression)
