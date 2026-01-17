#!/usr/bin/env bash
set -euo pipefail

# Runs TPU DFlash draft training with logs + pid file.
#
# Example:
#   CACHE_DIR=/dev/shm/dflash_cache/gptoss20b_ctx1024_b8_k4_n512 \
#   TEACHER_SNAPSHOT=/dev/shm/hf/hub/models--unsloth--gpt-oss-20b-BF16/snapshots/cc89b3e7fd423253264883a80a4fa5abc619649f \
#   RUN_NAME=gptoss20b_dflash_bs160_s2000 \
#   ./harmony/cuda-norm/scripts/run_tpu_dflash_train_logged.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PY="${ROOT}/harmony/cuda-norm/.venv-easydel/bin/python"
TRAIN_PY="${ROOT}/harmony/cuda-norm/scripts/tpu_dflash_train_with_easydel_trainer.py"

: "${CACHE_DIR:?Set CACHE_DIR=/dev/shm/dflash_cache/...}"
: "${TEACHER_SNAPSHOT:?Set TEACHER_SNAPSHOT=/dev/shm/hf/hub/.../snapshots/<sha>}"
: "${RUN_NAME:=dflash_run}"
: "${MODEL_NAME:=${RUN_NAME}}"
: "${RESUME_PATH:=}"

LOG_DIR="${ROOT}/harmony/cuda-norm/logs/tpu_dflash"
# Always checkpoint to /dev/shm (requested) to avoid root-FS pressure and to
# maximize checkpoint write throughput.
CKPT_DIR="${CKPT_DIR:-/dev/shm/dflash-checkpoints}"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

export HF_HOME="${HF_HOME:-/dev/shm/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/dev/shm/hf/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/dev/shm/xdg}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/dev/shm/jax_compilation_cache_dflash/${RUN_NAME}}"
export TMPDIR="${TMPDIR:-/dev/shm/tmp}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${XDG_CACHE_HOME}" "${JAX_COMPILATION_CACHE_DIR}" "${TMPDIR}"

# Corrupted cache entries can cause zstd warnings; start clean for long runs.
rm -rf "${JAX_COMPILATION_CACHE_DIR:?}/"* || true

LOG_PATH="${LOG_DIR}/${RUN_NAME}.log"
PID_PATH="${LOG_DIR}/${RUN_NAME}.pid"

TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-160}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
LOG_STEPS="${LOG_STEPS:-10}"
REPORT_STEPS="${REPORT_STEPS:-10}"
WORKERS="${WORKERS:-32}"
PREFETCH="${PREFETCH:-256}"
VOCAB_CHUNK_SIZE="${VOCAB_CHUNK_SIZE:-8192}"
DRAFT_LAYERS="${DRAFT_LAYERS:-8}"
MLP_RATIO="${MLP_RATIO:-4.0}"
QK_NORM="${QK_NORM:-true}"
REMAT="${REMAT:-true}"

EXTRA_ARGS=()
if [[ -n "${RESUME_PATH}" ]]; then
  EXTRA_ARGS+=(--resume-path "${RESUME_PATH}")
fi

set -x
nohup "${VENV_PY}" -u "${TRAIN_PY}" \
  --cache-dir "${CACHE_DIR}" \
  --teacher-snapshot-dir "${TEACHER_SNAPSHOT}" \
  --save-directory "${CKPT_DIR}" \
  --model-name "${MODEL_NAME}" \
  "${EXTRA_ARGS[@]}" \
  --max-training-steps "${MAX_TRAINING_STEPS}" \
  --total-batch-size "${TOTAL_BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --draft-layers "${DRAFT_LAYERS}" \
  --mlp-ratio "${MLP_RATIO}" \
  --qk-norm "${QK_NORM}" \
  --remat "${REMAT}" \
  --spmd false \
  --dp 8 --tp 1 \
  --prefetch "${PREFETCH}" \
  --workers "${WORKERS}" \
  --vocab-chunk-size "${VOCAB_CHUNK_SIZE}" \
  --save-steps "${SAVE_STEPS}" \
  --log-steps "${LOG_STEPS}" \
  --report-steps "${REPORT_STEPS}" \
  --disable-wandb \
  >"${LOG_PATH}" 2>&1 &
echo $! > "${PID_PATH}"
set +x

echo "pid=$(cat "${PID_PATH}") log=${LOG_PATH} ckpt_root=${CKPT_DIR}/${RUN_NAME}"
