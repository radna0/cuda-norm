# DFlash Draft Training (TPU, EasyDeL, cache-first)

This trains a DFlash draft model **without running the teacher forward during training**:
teacher features are precomputed into a cache, then training is pure TPU draft forward + CE.

## 0) Prereqs

- TPU runtime available (8 devices) and `harmony/cuda-norm/.venv-easydel` created by `harmony/cuda-norm/scripts/setup_tpu_easydel_env.sh`.
- `HF_TOKEN` present in `harmony/cuda-norm/.env` (the training scripts load it).

## 1) Build teacher cache (one-time)

Example cache directory (stored in RAM-disk):

`/dev/shm/dflash_cache/gptoss20b_ctx1024_b8_k4_n512`

Build it:

`harmony/cuda-norm/.venv-easydel/bin/python harmony/cuda-norm/scripts/tpu_dflash_build_teacher_cache.py --out-dir /dev/shm/dflash_cache/gptoss20b_ctx1024_b8_k4_n512`

## 2) Train the draft model (long run)

Start training with logs + pid:

```bash
CACHE_DIR=/dev/shm/dflash_cache/gptoss20b_ctx1024_b8_k4_n512 \
TEACHER_SNAPSHOT=/dev/shm/hf/hub/models--unsloth--gpt-oss-20b-BF16/snapshots/cc89b3e7fd423253264883a80a4fa5abc619649f \
RUN_NAME=gptoss20b_dflash_ctx1024_b8_k4_bs160_s2000 \
TOTAL_BATCH_SIZE=160 MAX_TRAINING_STEPS=2000 SAVE_STEPS=500 \
./harmony/cuda-norm/scripts/run_tpu_dflash_train_logged.sh
```

Monitor:

- Log: `harmony/cuda-norm/logs/tpu_dflash/<RUN_NAME>.log`
- PID: `harmony/cuda-norm/logs/tpu_dflash/<RUN_NAME>.pid`
- Checkpoints + run manifest: `/dev/shm/dflash-checkpoints/<RUN_NAME>/`

Stop:

`kill $(cat harmony/cuda-norm/logs/tpu_dflash/<RUN_NAME>.pid)`

## Notes

- The first 1–2 steps are slow (XLA compile); steady-state steps are much faster.
- If you see `ZstdError` warnings, the persistent compilation cache is corrupted; the launch script clears `JAX_COMPILATION_CACHE_DIR` before long runs.
