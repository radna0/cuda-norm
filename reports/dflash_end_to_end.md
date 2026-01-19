# DFlash end‑to‑end (TPU training → HF export → SGLang GPU benchmark)

## Artifacts

- TPU training logs: `harmony/cuda-norm/logs/tpu_dflash/`
- Exported GPU draft checkpoint (HF+shards): `harmony/cuda-norm/artifacts/dflash_sglang/`
- Packaged tarball for Kaggle upload: `harmony/cuda-norm/artifacts/dflash_sglang/*_sglang.tar.gz`

## Step 1 — Train draft on TPU (EasyDeL)

Use your existing TPU training entrypoint (example):
- `harmony/cuda-norm/scripts/tpu_dflash_train_with_easydel_trainer.py`

This produces an EasyDeL run directory:
- `/dev/shm/dflash-checkpoints/<run_name>/run-<step>/`

## Step 2 — Convert EasyDeL run → HF+safetensors draft checkpoint

```bash
python harmony/cuda-norm/scripts/convert_easydel_dflash_ckpt_to_sglang.py \
  --run-dir /dev/shm/dflash-checkpoints/<run_name>/run-<step> \
  --dst harmony/cuda-norm/artifacts/dflash_sglang/<name> \
  --keep-fc-bias \
  --force
```

Optional (align layer-id semantics to SGLang PR #16818):

```bash
python harmony/cuda-norm/scripts/convert_easydel_dflash_ckpt_to_sglang.py \
  --target-layer-ids-mode afterlayer \
  ...
```

Package for transfer:

```bash
tar -C harmony/cuda-norm/artifacts/dflash_sglang -czf \
  harmony/cuda-norm/artifacts/dflash_sglang/<name>.tar.gz \
  <name>
```

## Step 3 — TPU correctness decode (block verify) + baseline speedup

```bash
export TEACHER_SNAPSHOT_DIR=...
export DRAFT_PARAMS=.../draft_params.msgpack
export ALSO_RUN_BASELINE=1

harmony/cuda-norm/scripts/run_tpu_dflash_decode_logged.sh \
  --max-new-tokens 64 \
  --block-size 8 \
  --draft-layers 8 \
  --num-context-features 4
```

## Step 4 — Kaggle/H100: load + benchmark SGLang DFLASH

Upload the tarball to Kaggle as a dataset, then run:

```bash
export DRAFT_TAR_GZ=/kaggle/input/<dataset>/<name>.tar.gz
export TARGET_MODEL=openai/gpt-oss-20b
export ATTN_BACKEND=fa3

./cuda-norm-sync/scripts/kaggle_sglang_dflash_smoke_from_tar.sh
```

The benchmark prints JSON including:
- `speedup_x`
- `accept_rate_est`

