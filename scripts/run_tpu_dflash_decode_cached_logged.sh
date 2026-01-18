#!/usr/bin/env bash
set -euo pipefail

# Run TPU DFlash decode harness with logging.
#
# Requires:
#   TEACHER_SNAPSHOT_DIR=...
#   DRAFT_RUN_DIR=/dev/shm/dflash-checkpoints/<run_name>/run-<step>
#
# Optional:
#   DFLASH_DECODE_SCRIPT=tpu_dflash_spec_decode_blockverify_v1.py
#
# Example:
#   export ALSO_RUN_BASELINE=1
#   harmony/cuda-norm/scripts/run_tpu_dflash_decode_cached_logged.sh --max-new-tokens 256

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="${SCRIPT_DIR}/../.venv-easydel/bin/python"
LOG_DIR="${SCRIPT_DIR}/../logs/tpu_dflash"
mkdir -p "$LOG_DIR"

TPU_LOCK_PATH="${TPU_LOCK_PATH:-/dev/shm/tpu.lock}"

TS="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-tpu_dflash_decode_cached_v1_${TS}}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

TEACHER_SNAPSHOT_DIR="${TEACHER_SNAPSHOT_DIR:-}"
DRAFT_RUN_DIR="${DRAFT_RUN_DIR:-}"
if [[ -z "$TEACHER_SNAPSHOT_DIR" || -z "$DRAFT_RUN_DIR" ]]; then
  echo "TEACHER_SNAPSHOT_DIR and DRAFT_RUN_DIR are required env vars" >&2
  exit 2
fi

# Crash-safety: forbid verify on-demand token-bucket compilation (can segfault on TPU).
export EASYDEL_VERIFY_DISALLOW_ON_DEMAND_COMPILE="${EASYDEL_VERIFY_DISALLOW_ON_DEMAND_COMPILE:-1}"

ARGS=()
if [[ -n "${ALSO_RUN_BASELINE:-}" ]]; then
  ARGS+=(--also-run-baseline)
fi

set -x
DECODE_SCRIPT="${DFLASH_DECODE_SCRIPT:-tpu_dflash_spec_decode_cached_v1.py}"
bash -c 'exec 9>"$1"; flock -x 9; shift; exec "$@"' bash "${TPU_LOCK_PATH}" \
  "${VENV_PY}" -u "${SCRIPT_DIR}/${DECODE_SCRIPT}" \
  --teacher-snapshot-dir "${TEACHER_SNAPSHOT_DIR}" \
  --draft-run-dir "${DRAFT_RUN_DIR}" \
  "${ARGS[@]}" \
  "$@" \
  2>&1 | tee "${LOG_FILE}"

echo "[+] log: ${LOG_FILE}" >&2
