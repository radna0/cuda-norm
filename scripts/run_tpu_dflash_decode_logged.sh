#!/usr/bin/env bash
set -euo pipefail

# Run TPU DFlash spec-v1 decode correctness harness with logging.
#
# Example:
#   TEACHER_SNAPSHOT_DIR=/dev/shm/hf/.../snapshots/... \
#   DRAFT_PARAMS=/dev/shm/dflash-checkpoints/.../draft_params.msgpack \
#   ./harmony/cuda-norm/scripts/run_tpu_dflash_decode_logged.sh --max-new-tokens 64

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CUDA_NORM_DIR="${ROOT_DIR}/harmony/cuda-norm"
VENV_PY="${CUDA_NORM_DIR}/.venv-easydel/bin/python"
LOG_DIR="${CUDA_NORM_DIR}/logs/tpu_dflash"
mkdir -p "$LOG_DIR"

TPU_LOCK_PATH="${TPU_LOCK_PATH:-/dev/shm/tpu.lock}"

TS="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-tpu_dflash_decode_v1_${TS}}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

TEACHER_SNAPSHOT_DIR="${TEACHER_SNAPSHOT_DIR:-}"
DRAFT_PARAMS="${DRAFT_PARAMS:-}"
if [[ -z "$TEACHER_SNAPSHOT_DIR" || -z "$DRAFT_PARAMS" ]]; then
  echo "TEACHER_SNAPSHOT_DIR and DRAFT_PARAMS are required env vars" >&2
  exit 2
fi

# Crash-safety: forbid verify on-demand token-bucket compilation (can segfault on TPU).
export EASYDEL_VERIFY_DISALLOW_ON_DEMAND_COMPILE="${EASYDEL_VERIFY_DISALLOW_ON_DEMAND_COMPILE:-1}"

set -x
bash -c 'exec 9>"$1"; flock -x 9; shift; exec "$@"' bash "${TPU_LOCK_PATH}" \
  "${VENV_PY}" -u "${CUDA_NORM_DIR}/scripts/tpu_dflash_spec_decode_v1.py" \
  --teacher-snapshot-dir "${TEACHER_SNAPSHOT_DIR}" \
  --draft-params "${DRAFT_PARAMS}" \
  ${ALSO_RUN_BASELINE:+--also-run-baseline} \
  "$@" \
  2>&1 | tee "${LOG_FILE}"

echo "[+] log: ${LOG_FILE}" >&2
