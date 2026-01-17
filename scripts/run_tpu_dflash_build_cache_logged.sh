#!/usr/bin/env bash
set -euo pipefail

# Runs TPU teacher-cache build for DFlash with logs + pid file.
#
# Required env:
#   TEACHER_SNAPSHOT=/dev/shm/hf/hub/.../snapshots/<sha>
#   OUT_DIR=/dev/shm/dflash_cache/<name>
#
# Optional env:
#   TEACHER_EASYDEL_DIR=/dev/shm/easydel_teachers/gptoss20b_bf16
#   SAVE_TEACHER_EASYDEL_DIR=/dev/shm/easydel_teachers/gptoss20b_bf16
#   CTX_LEN=1024
#   BLOCK_SIZE=8
#   NUM_CONTEXT_FEATURES=4
#   NUM_BLOCKS=256
#   ROLLOUT_STEPS=1
#   BATCH_SIZE=1
#   PAGE_SIZE=128
#   HBM_UTILIZATION=0.20
#
# Example:
#   TEACHER_SNAPSHOT=... OUT_DIR=/dev/shm/dflash_cache/gptoss20b_ctx1024_b8_k4_n256 \\
#   ./harmony/cuda-norm/scripts/run_tpu_dflash_build_cache_logged.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PY="${ROOT}/harmony/cuda-norm/.venv-easydel/bin/python"
BUILD_PY="${ROOT}/harmony/cuda-norm/scripts/tpu_dflash_build_teacher_cache.py"

: "${TEACHER_SNAPSHOT:?Set TEACHER_SNAPSHOT=/dev/shm/hf/hub/.../snapshots/<sha>}"
: "${OUT_DIR:?Set OUT_DIR=/dev/shm/dflash_cache/<name>}"

LOG_DIR="${ROOT}/harmony/cuda-norm/logs/tpu_dflash"
mkdir -p "${LOG_DIR}"

RUN_NAME="${RUN_NAME:-tpu_dflash_build_cache_$(date -u +%Y%m%d_%H%M%S)}"
LOG_PATH="${LOG_DIR}/${RUN_NAME}.log"
PID_PATH="${LOG_DIR}/${RUN_NAME}.pid"

CTX_LEN="${CTX_LEN:-1024}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
NUM_CONTEXT_FEATURES="${NUM_CONTEXT_FEATURES:-4}"
NUM_BLOCKS="${NUM_BLOCKS:-256}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PAGE_SIZE="${PAGE_SIZE:-128}"
HBM_UTILIZATION="${HBM_UTILIZATION:-0.20}"
PREFILL_CHUNK="${PREFILL_CHUNK:-256}"
MAX_ROWS_PER_PACK="${MAX_ROWS_PER_PACK:-2000}"
TEACHER_EASYDEL_DIR="${TEACHER_EASYDEL_DIR:-}"
SAVE_TEACHER_EASYDEL_DIR="${SAVE_TEACHER_EASYDEL_DIR:-}"

export HF_HOME="${HF_HOME:-/dev/shm/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/dev/shm/hf/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/dev/shm/xdg}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/dev/shm/jax_compilation_cache_dflash/${RUN_NAME}}"
export TMPDIR="${TMPDIR:-/dev/shm/tmp}"
# Crash-safety: forbid verify on-demand token-bucket compilation (can segfault on TPU).
export EASYDEL_VERIFY_DISALLOW_ON_DEMAND_COMPILE="${EASYDEL_VERIFY_DISALLOW_ON_DEMAND_COMPILE:-1}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${XDG_CACHE_HOME}" "${JAX_COMPILATION_CACHE_DIR}" "${TMPDIR}"

set -x
ARGS=()
if [[ -n "${TEACHER_EASYDEL_DIR}" ]]; then
  ARGS+=(--teacher-easydel-dir "${TEACHER_EASYDEL_DIR}")
fi
if [[ -n "${SAVE_TEACHER_EASYDEL_DIR}" ]]; then
  ARGS+=(--save-teacher-easydel-dir "${SAVE_TEACHER_EASYDEL_DIR}")
fi

nohup "${VENV_PY}" -u "${BUILD_PY}" \
  --model-snapshot-dir "${TEACHER_SNAPSHOT}" \
  "${ARGS[@]}" \
  --ctx-len "${CTX_LEN}" \
  --block-size "${BLOCK_SIZE}" \
  --num-context-features "${NUM_CONTEXT_FEATURES}" \
  --num-blocks "${NUM_BLOCKS}" \
  --rollout-steps "${ROLLOUT_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --page-size "${PAGE_SIZE}" \
  --hbm-utilization "${HBM_UTILIZATION}" \
  --prefill-chunk "${PREFILL_CHUNK}" \
  --max-rows-per-pack "${MAX_ROWS_PER_PACK}" \
  --out-dir "${OUT_DIR}" \
  --out "/dev/shm/out/${RUN_NAME}.npz" \
  >"${LOG_PATH}" 2>&1 &
echo $! > "${PID_PATH}"
set +x

echo "pid=$(cat "${PID_PATH}") log=${LOG_PATH} out_dir=${OUT_DIR}"
