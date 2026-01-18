#!/usr/bin/env bash
set -euo pipefail

# Runs TPU teacher-cache build for DFlash draft training with logs + pid file.
#
# Example:
#   MODEL_SNAPSHOT_DIR=/dev/shm/hf/hub/models--unsloth--gpt-oss-20b-BF16/snapshots/cc89b3e7fd423253264883a80a4fa5abc619649f \
#   TEACHER_EASYDEL_DIR=/dev/shm/easydel_teachers/gptoss20b_bf16_cc89b3 \
#   RUN_NAME=cache_ctx1024_b8_k4_n16_roll64_pos0_65k_131k \
#   NUM_BLOCKS=16 ROLLOUT_STEPS=64 CTX_LEN=1024 BLOCK_SIZE=8 NUM_CONTEXT_FEATURES=4 \
#   POSITION_OFFSETS=0,65536,129536 \
#   POSITION_OFFSET_MODE=per_prompt_rollout \
#   ./harmony/cuda-norm/scripts/run_tpu_dflash_build_cache_logged.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV_PY="${ROOT}/harmony/cuda-norm/.venv-easydel/bin/python"
BUILD_PY="${ROOT}/harmony/cuda-norm/scripts/tpu_dflash_build_teacher_cache.py"

: "${MODEL_SNAPSHOT_DIR:?Set MODEL_SNAPSHOT_DIR=/dev/shm/hf/hub/.../snapshots/<sha>}"
: "${RUN_NAME:=dflash_cache_run}"

LOG_DIR="${ROOT}/harmony/cuda-norm/logs/tpu_dflash"
mkdir -p "${LOG_DIR}"

TPU_LOCK_PATH="${TPU_LOCK_PATH:-/dev/shm/tpu.lock}"

export HF_HOME="${HF_HOME:-/dev/shm/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/dev/shm/hf/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/dev/shm/xdg}"
export TMPDIR="${TMPDIR:-/dev/shm/tmp}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${XDG_CACHE_HOME}" "${TMPDIR}"

# Force correctness-first TPU attention path unless explicitly overridden.
export DFLASH_FORCE_RAGGED_V2="${DFLASH_FORCE_RAGGED_V2:-1}"

LOG_PATH="${LOG_DIR}/${RUN_NAME}.log"
PID_PATH="${LOG_DIR}/${RUN_NAME}.pid"

OUT_DIR="${OUT_DIR:-/dev/shm/dflash_cache/${RUN_NAME}}"
# Stream writes directly into out_dir as mmap arrays (lower peak RAM).
STREAM_OUT_DIR="${STREAM_OUT_DIR:-${OUT_DIR}}"
# Default: do NOT also write the legacy .npz for large caches.
WRITE_NPZ="${WRITE_NPZ:-false}"

TEACHER_EASYDEL_DIR="${TEACHER_EASYDEL_DIR:-}"
SAVE_TEACHER_EASYDEL_DIR="${SAVE_TEACHER_EASYDEL_DIR:-}"

CTX_LEN="${CTX_LEN:-1024}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
NUM_CONTEXT_FEATURES="${NUM_CONTEXT_FEATURES:-4}"
NUM_BLOCKS="${NUM_BLOCKS:-16}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-64}"
ROLLOUT_ACCEPT_LEN_MODE="${ROLLOUT_ACCEPT_LEN_MODE:-geometric}"
ROLLOUT_ACCEPT_LEN_P="${ROLLOUT_ACCEPT_LEN_P:-0.35}"
ROLLOUT_STATE_EVOLUTION="${ROLLOUT_STATE_EVOLUTION:-dflash_commit}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SEED:-0}"
PREFILL_CHUNK="${PREFILL_CHUNK:-256}"
PAGE_SIZE="${PAGE_SIZE:-128}"
HBM_UTILIZATION="${HBM_UTILIZATION:-0.20}"
SHARDING_AXIS_DIMS="${SHARDING_AXIS_DIMS:-1,8,1,1,1}"

CALIB_REPO_ID="${CALIB_REPO_ID:-radna0/harmony-qwen3-calib-packs-v2-20260113}"
CALIB_DATA_FILES="${CALIB_DATA_FILES:-packs/reasoning_style_10k_v2/reasoning_style_10k_v2.parquet,tool_agentic_10k_v6.parquet,packs/calib_prompt_10000_v2/calib_prompt_10000_v2.parquet}"
MAX_ROWS_PER_PACK="${MAX_ROWS_PER_PACK:-2000}"
# IMPORTANT:
# - Do not set pos_off=131072 directly: with ctx_len=1023 and rollouts, that
#   exceeds max_position_embeddings=131072 and degrades/invalidates the cache.
# - Use a near-max safe offset (e.g. 129536) so:
#     pos_off + ctx_len + rollout_steps*block_size <= max_position_embeddings - 1
POSITION_OFFSETS="${POSITION_OFFSETS:-0,65536,129536}"
POSITION_OFFSET_MODE="${POSITION_OFFSET_MODE:-per_prompt_rollout}"

EXTRA_ARGS=()
if [[ -n "${TEACHER_EASYDEL_DIR}" ]]; then
  EXTRA_ARGS+=(--teacher-easydel-dir "${TEACHER_EASYDEL_DIR}")
fi
if [[ -n "${SAVE_TEACHER_EASYDEL_DIR}" ]]; then
  EXTRA_ARGS+=(--save-teacher-easydel-dir "${SAVE_TEACHER_EASYDEL_DIR}")
fi

set -x
nohup bash -c 'exec 9>"$1"; flock -x 9; shift; exec "$@"' bash "${TPU_LOCK_PATH}" \
  "${VENV_PY}" -u "${BUILD_PY}" \
  --model-snapshot-dir "${MODEL_SNAPSHOT_DIR}" \
  "${EXTRA_ARGS[@]}" \
  --ctx-len "${CTX_LEN}" \
  --block-size "${BLOCK_SIZE}" \
  --num-context-features "${NUM_CONTEXT_FEATURES}" \
  --num-blocks "${NUM_BLOCKS}" \
  --rollout-steps "${ROLLOUT_STEPS}" \
  --rollout-accept-len-mode "${ROLLOUT_ACCEPT_LEN_MODE}" \
  --rollout-accept-len-p "${ROLLOUT_ACCEPT_LEN_P}" \
  --rollout-state-evolution "${ROLLOUT_STATE_EVOLUTION}" \
  --batch-size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --prefill-chunk "${PREFILL_CHUNK}" \
  --page-size "${PAGE_SIZE}" \
  --hbm-utilization "${HBM_UTILIZATION}" \
  --sharding-axis-dims "${SHARDING_AXIS_DIMS}" \
  --calib-repo-id "${CALIB_REPO_ID}" \
  --calib-data-files "${CALIB_DATA_FILES}" \
  --max-rows-per-pack "${MAX_ROWS_PER_PACK}" \
  --position-offsets "${POSITION_OFFSETS}" \
  --position-offset-mode "${POSITION_OFFSET_MODE}" \
  --out-dir "${OUT_DIR}" \
  --stream-out-dir "${STREAM_OUT_DIR}" \
  --write-npz "${WRITE_NPZ}" \
  >"${LOG_PATH}" 2>&1 &
echo $! > "${PID_PATH}"
set +x

echo "pid=$(cat "${PID_PATH}") log=${LOG_PATH} out_dir=${OUT_DIR} tpu_lock=${TPU_LOCK_PATH}"
