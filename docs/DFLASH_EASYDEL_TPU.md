# DFlash draft training on TPU (EasyDeL, cache-first)

## Goals
- Train a DFlash **draft** model for GPT‑OSS without running teacher forward during training.
- Use TPU SPMD: **dp** sharded batch + **tp** sharded `lm_head.weight` for exact full‑vocab CE.
- Put all large caches and compilation artifacts under `/dev/shm` to avoid disk IO stalls.

## One-time setup
```bash
cd /home/kojoe/harmony/cuda-norm
./scripts/setup_tpu_easydel_env.sh
source .venv-easydel/bin/activate
```

Environment:
- Ensure `HF_TOKEN` is set (or in `harmony/cuda-norm/.env`).

## 1) Build the teacher cache (expensive; uses the teacher model)
This runs the teacher forward once per block and writes an mmap‑friendly cache directory:
```bash
python scripts/tpu_dflash_build_teacher_cache.py \
  --model-snapshot-dir /dev/shm/hf/hub/.../snapshots/<SNAPSHOT> \
  --ctx-len 4096 \
  --block-size 8 \
  --num-context-features 4 \
  --num-blocks 4096 \
  --batch-size 4 \
  --out-dir /dev/shm/dflash_cache/gptoss20b_ctx4096_b8_k4
```

Outputs (inside `--out-dir`):
- `meta.json`
- `context_features_u16.npy` (bf16 bitpatterns)
- `anchor_embedding_u16.npy`
- `target_ids.npy`

## 2) Train draft from cache (cheap; no teacher forward)
```bash
python scripts/tpu_dflash_train_with_easydel_trainer.py \
  --cache-dir /dev/shm/dflash_cache/gptoss20b_ctx4096_b8_k4 \
  --teacher-snapshot-dir /dev/shm/hf/hub/.../snapshots/<SNAPSHOT> \
  --save-directory /dev/shm/easydel-checkpoints \
  --model-name gptoss20b-dflash-draft \
  --max-training-steps 2000 \
  --total-batch-size 64 \
  --dp 2 --tp 4 \
  --vocab-chunk-size 8192 \
  --save-steps 500 \
  --log-steps 10
```

Notes:
- `total_batch_size` is the **global** batch per step; it must be divisible by `dp`.
- If memory is low (TPU HBM mostly unused), increase `total_batch_size` first.

## What’s implemented
- `harmony/cuda-norm/dflash_gptoss/easydel_dflash_cache.py`: mmap dataset for cache dirs.
- `harmony/cuda-norm/dflash_gptoss/easydel_dflash_draft_model.py`: NNX DFlash draft model.
- `harmony/cuda-norm/dflash_gptoss/easydel_dflash_trainer.py`: EasyDeL `Trainer` subclass:
  - cache-first (no teacher forward)
  - tp-sharded `lm_head.weight` + chunked CE per shard
  - dp-sharded batch + dp pmean for grads

