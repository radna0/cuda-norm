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
# Persistent JAX compilation cache can explode in size and fill /dev/shm.
# Default: disable (still uses in-memory compilation cache).
# To enable explicitly:
#   ENABLE_JAX_PERSISTENT_COMPILATION_CACHE=1 JAX_COMPILATION_CACHE_DIR=/path/...
if [[ "${ENABLE_JAX_PERSISTENT_COMPILATION_CACHE:-0}" == "1" ]]; then
  export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/dev/shm/jax_compilation_cache_dflash/${RUN_NAME}}"
else
  unset JAX_COMPILATION_CACHE_DIR || true
fi
export TMPDIR="${TMPDIR:-/dev/shm/tmp}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${XDG_CACHE_HOME}" "${TMPDIR}"

# If the persistent compilation cache is enabled, keep it across runs to reduce compile cost.
if [[ "${ENABLE_JAX_PERSISTENT_COMPILATION_CACHE:-0}" == "1" ]]; then
  mkdir -p "${JAX_COMPILATION_CACHE_DIR}"
fi

INSTANCE_TAG="${INSTANCE_TAG:-$(date -u +%Y%m%d_%H%M%S)}"

LOG_PATH_REAL="${LOG_DIR}/${RUN_NAME}.${INSTANCE_TAG}.log"
PID_PATH_REAL="${LOG_DIR}/${RUN_NAME}.${INSTANCE_TAG}.pid"

# Stable pointers (for humans + helper scripts).
LOG_PATH="${LOG_DIR}/${RUN_NAME}.log"
PID_PATH="${LOG_DIR}/${RUN_NAME}.pid"

# Point the stable paths at the unique per-run artifacts.
ln -sf "$(basename "${LOG_PATH_REAL}")" "${LOG_PATH}"
ln -sf "$(basename "${PID_PATH_REAL}")" "${PID_PATH}"

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
  >"${LOG_PATH_REAL}" 2>&1 &
echo $! > "${PID_PATH_REAL}"
set +x

echo "pid=$(cat "${PID_PATH}") log=${LOG_PATH} (real=${LOG_PATH_REAL}) ckpt_root=${CKPT_DIR}/${RUN_NAME}"
