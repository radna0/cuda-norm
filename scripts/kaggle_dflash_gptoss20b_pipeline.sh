#!/usr/bin/env bash
set -euo pipefail

# End-to-end GPT-OSS-20B DFlash pipeline on Kaggle (H100):
#   1) install deps into /kaggle/working/.pydeps
#   2) overlay our patched SGLang sources (DFLASH + GPT-OSS fixes)
#   3) train HF draft checkpoint (SGLang teacher)
#   4) convert HF draft -> SGLang DFlashDraftModel checkpoint
#   5) run baseline vs DFLASH decode benchmark (max_new_tokens=2048)

cd /kaggle/working

export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export TRANSFORMERS_NO_JAX=1
export HF_HUB_ENABLE_HF_TRANSFER=1

# Load secrets (HF_TOKEN) if present in synced .env.
if [[ -f /kaggle/working/cuda-norm-sync/.env ]]; then
  set -a
  # shellcheck disable=SC1091
  . /kaggle/working/cuda-norm-sync/.env
  set +a
fi

PYDEPS=/kaggle/working/.pydeps
mkdir -p "$PYDEPS"
export PYTHONPATH="/kaggle/working/cuda-norm-sync:$PYDEPS:${PYTHONPATH:-}"

python -V
nvidia-smi -L || true

# Install runtime deps locally (Kaggle site-packages can be read-only).
python -m pip -q install --no-cache-dir -U -t "$PYDEPS" \
  huggingface_hub==0.36.0 hf-transfer \
  safetensors datasets pyarrow requests \
  transformers==4.56.2

# SGLang + kernels (for teacher forward + DFLASH server).
python -m pip -q install --no-cache-dir -U -t "$PYDEPS" "sglang[all]"

python - <<'PY'
import sglang
print("sglang", getattr(sglang, "__version__", "unknown"))
PY

export SGLANG_OVERLAY_SRC=/kaggle/working/cuda-norm-sync/sglang-flashinfer/python/sglang
python /kaggle/working/cuda-norm-sync/scripts/sglang_overlay_install.py

DATASET_REPO="${DATASET_REPO:-radna0/harmony-qwen3-calib-packs-v2-20260113}"
TRAIN_FILES_CSV="${TRAIN_FILES_CSV:-packs/reasoning_style_10k_v2/reasoning_style_10k_v2.parquet,packs/tool_agentic_10k_v6/tool_agentic_10k_v6.parquet,packs/calib_prompt_10000_v2/calib_prompt_10000_v2.parquet}"
TARGET_MODEL="${TARGET_MODEL:-openai/gpt-oss-20b}"
SEQ_LEN="${SEQ_LEN:-4096}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-4}"
MLP_RATIO="${MLP_RATIO:-4.0}"
MAX_STEPS="${MAX_STEPS:-200}"
SAVE_EVERY="${SAVE_EVERY:-200}"
LR="${LR:-2e-4}"
TEACHER_ATTN_BACKEND="${TEACHER_ATTN_BACKEND:-fa3}"
TEACHER_MEM_FRACTION="${TEACHER_MEM_FRACTION:-0.75}"

OUT_ROOT="/kaggle/working/dflash_gptoss20b"

python /kaggle/working/cuda-norm-sync/scripts/dflash_gptoss20b_train_kaggle.py \
  --target-model "$TARGET_MODEL" \
  --dataset-repo "$DATASET_REPO" \
  --train-files-csv "$TRAIN_FILES_CSV" \
  --seq-len "$SEQ_LEN" \
  --block-size "$BLOCK_SIZE" \
  --num-hidden-layers "$NUM_HIDDEN_LAYERS" \
  --mlp-ratio "$MLP_RATIO" \
  --max-steps "$MAX_STEPS" \
  --save-every "$SAVE_EVERY" \
  --lr "$LR" \
  --teacher-attn-backend "$TEACHER_ATTN_BACKEND" \
  --teacher-mem-fraction "$TEACHER_MEM_FRACTION" \
  --out-root "$OUT_ROOT"

# Find the newest checkpoint and convert it.
LATEST_STEP_DIR="$(ls -1dt "$OUT_ROOT"/*/step_* | head -n 1)"
CONVERTED_DIR="${LATEST_STEP_DIR}_sglang"
python /kaggle/working/cuda-norm-sync/scripts/convert_hf_dflash_ckpt_to_sglang.py \
  --src "$LATEST_STEP_DIR" \
  --dst "$CONVERTED_DIR" \
  --force

# Benchmark baseline vs DFLASH at long decode.
python /kaggle/working/cuda-norm-sync/scripts/dflash_gptoss20b_bench_sglang_kaggle.py \
  --target-model "$TARGET_MODEL" \
  --draft-model "$CONVERTED_DIR" \
  --attention-backend "$TEACHER_ATTN_BACKEND" \
  --block-size "$BLOCK_SIZE" \
  --max-new-tokens 2048 \
  --concurrency 4 \
  --num-prompts 8

echo "[+] done"
