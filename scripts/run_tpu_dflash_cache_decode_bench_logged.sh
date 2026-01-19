#!/usr/bin/env bash
set -euo pipefail

# Run the cache-distribution DFlash acceptance/throughput probe on TPU with a TPU lock.
#
# Required env:
#   CACHE_DIR=...            (DFlash cache directory with meta.json + .npy files)
#   TEACHER_SNAPSHOT_DIR=... (HF snapshot dir)
#   DRAFT_RUN_DIR=...        (EasyDeL run-* dir)
#
# Optional env:
#   TEACHER_EASYDEL_DIR=...  (EasyDeL-native teacher dir)
#   RUN_NAME=...
#   LOG_DIR=...
#
# Example:
#   export CACHE_DIR=/dev/shm/dflash_cache/build_cache_gptoss20b_ctx1024_b8_k4_n64_roll16_geomp035_perprompt_...
#   export TEACHER_SNAPSHOT_DIR=/dev/shm/hf/hub/models--unsloth--gpt-oss-20b-BF16/snapshots/<sha>
#   export TEACHER_EASYDEL_DIR=/dev/shm/easydel_teachers/gptoss20b_bf16_v2
#   export DRAFT_RUN_DIR=/dev/shm/dflash-checkpoints/<run>/run-500
#   ./harmony/cuda-norm/scripts/run_tpu_dflash_cache_decode_bench_logged.sh --sample-idx 0 --blocks 32 --warmup-blocks 2 --draft-mode direct_window --page-size 128

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${ROOT}/harmony/cuda-norm/.venv-easydel/bin/python"
SCRIPT="${ROOT}/harmony/cuda-norm/scripts/tpu_dflash_cache_decode_bench.py"

: "${CACHE_DIR:?Set CACHE_DIR=...}"
: "${TEACHER_SNAPSHOT_DIR:?Set TEACHER_SNAPSHOT_DIR=...}"
: "${DRAFT_RUN_DIR:?Set DRAFT_RUN_DIR=...}"

TPU_LOCK_PATH="${TPU_LOCK_PATH:-/dev/shm/tpu.lock}"
LOG_DIR="${LOG_DIR:-${ROOT}/harmony/cuda-norm/logs/tpu_dflash}"
mkdir -p "${LOG_DIR}"

TS="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-cache_decode_bench_${TS}}"
LOG_PATH="${LOG_DIR}/${RUN_NAME}.log"

ARGS=()
if [[ -n "${TEACHER_EASYDEL_DIR:-}" ]]; then
  ARGS+=(--teacher-easydel-dir "${TEACHER_EASYDEL_DIR}")
fi

export EASYDEL_VERIFY_DISALLOW_ON_DEMAND_COMPILE="${EASYDEL_VERIFY_DISALLOW_ON_DEMAND_COMPILE:-1}"
export DFLASH_FORCE_RAGGED_V2="${DFLASH_FORCE_RAGGED_V2:-1}"

set -x
bash -c 'exec 9>"$1"; flock -x 9; shift; exec "$@"' bash "${TPU_LOCK_PATH}" \
  "${PY}" -u "${SCRIPT}" \
  --cache-dir "${CACHE_DIR}" \
  --teacher-snapshot-dir "${TEACHER_SNAPSHOT_DIR}" \
  --draft-run-dir "${DRAFT_RUN_DIR}" \
  "${ARGS[@]}" \
  "$@" \
  2>&1 | tee "${LOG_PATH}"
set +x

echo "[+] log: ${LOG_PATH}" >&2

