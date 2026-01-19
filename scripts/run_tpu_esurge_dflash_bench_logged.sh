#!/usr/bin/env bash
set -euo pipefail

# Run the TPU eSurge DFLASH benchmark with:
# - TPU lock (prevents JAX distributed collisions)
# - log file in harmony/cuda-norm/logs/tpu_dflash/
#
# Required env:
#   TEACHER_SNAPSHOT_DIR=... (HF snapshot dir)
#   DRAFT_RUN_DIR=...        (EasyDeL run-* dir containing config.json + model/)
#
# Optional env:
#   TEACHER_EASYDEL_DIR=...  (EasyDeL-native teacher checkpoint dir)
#   RUN_NAME=...
#   LOG_DIR=...
#   ALSO_RUN_BASELINE=1
#   CHECK_OUTPUT_MATCH=1
#
# Example (benchmark on training distribution):
#   export TEACHER_SNAPSHOT_DIR=/dev/shm/hf/hub/models--unsloth--gpt-oss-20b-BF16/snapshots/<sha>
#   export TEACHER_EASYDEL_DIR=/dev/shm/easydel_teachers/gptoss20b_bf16_v2
#   export DRAFT_RUN_DIR=/dev/shm/dflash-checkpoints/<run_name>/run-2000
#   export ALSO_RUN_BASELINE=1
#   export CHECK_OUTPUT_MATCH=1
#   export PROMPT_FROM_CACHE_DIR=/dev/shm/dflash_cache/<cache_dir>
#   export CACHE_SAMPLE_IDX=0
#   ./harmony/cuda-norm/scripts/run_tpu_esurge_dflash_bench_logged.sh --max-new-tokens 2048 --block-size 8 --max-model-len 8192

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${ROOT}/harmony/cuda-norm/.venv-easydel/bin/python"
SCRIPT="${ROOT}/harmony/cuda-norm/scripts/tpu_esurge_dflash_bench.py"

: "${TEACHER_SNAPSHOT_DIR:?Set TEACHER_SNAPSHOT_DIR=...}"
: "${DRAFT_RUN_DIR:?Set DRAFT_RUN_DIR=...}"

TPU_LOCK_PATH="${TPU_LOCK_PATH:-/dev/shm/tpu.lock}"
LOG_DIR="${LOG_DIR:-${ROOT}/harmony/cuda-norm/logs/tpu_dflash}"
mkdir -p "${LOG_DIR}"

TS="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-esurge_dflash_bench_${TS}}"
LOG_PATH="${LOG_DIR}/${RUN_NAME}.log"

ARGS=()
if [[ -n "${TEACHER_EASYDEL_DIR:-}" ]]; then
  ARGS+=(--teacher-easydel-dir "${TEACHER_EASYDEL_DIR}")
fi
if [[ -n "${PROMPT_FROM_CACHE_DIR:-}" ]]; then
  ARGS+=(--prompt-from-cache-dir "${PROMPT_FROM_CACHE_DIR}")
  ARGS+=(--cache-sample-idx "${CACHE_SAMPLE_IDX:-0}")
fi
if [[ -n "${ALSO_RUN_BASELINE:-}" ]]; then
  ARGS+=(--also-run-baseline)
fi
if [[ -n "${CHECK_OUTPUT_MATCH:-}" ]]; then
  ARGS+=(--check-output-match)
fi

export EASYDEL_VERIFY_DISALLOW_ON_DEMAND_COMPILE="${EASYDEL_VERIFY_DISALLOW_ON_DEMAND_COMPILE:-1}"
export DFLASH_FORCE_RAGGED_V2="${DFLASH_FORCE_RAGGED_V2:-1}"

set -x
bash -c 'exec 9>"$1"; flock -x 9; shift; exec "$@"' bash "${TPU_LOCK_PATH}" \
  "${PY}" -u "${SCRIPT}" \
  --teacher-snapshot-dir "${TEACHER_SNAPSHOT_DIR}" \
  --draft-run-dir "${DRAFT_RUN_DIR}" \
  "${ARGS[@]}" \
  "$@" \
  2>&1 | tee "${LOG_PATH}"
set +x

echo "[+] log: ${LOG_PATH}" >&2

