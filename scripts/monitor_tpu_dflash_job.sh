#!/usr/bin/env bash
set -euo pipefail

# Lightweight monitor for TPU DFlash jobs (cache build / training).
#
# Usage:
#   ./harmony/cuda-norm/scripts/monitor_tpu_dflash_job.sh RUN_NAME
#
# Example:
#   ./harmony/cuda-norm/scripts/monitor_tpu_dflash_job.sh gptoss20b_build_cache_ctx1024_b8_k4_n256_roll8_v1_20260117_042440

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_NAME="${1:?Run name required (matches logs/tpu_dflash/<RUN_NAME>.pid)}"

LOG_DIR="${ROOT}/harmony/cuda-norm/logs/tpu_dflash"
PID_PATH="${LOG_DIR}/${RUN_NAME}.pid"
LOG_PATH="${LOG_DIR}/${RUN_NAME}.log"

if [[ ! -f "${PID_PATH}" ]]; then
  echo "missing pid: ${PID_PATH}" >&2
  exit 1
fi
if [[ ! -f "${LOG_PATH}" ]]; then
  echo "missing log: ${LOG_PATH}" >&2
  exit 1
fi

PID="$(cat "${PID_PATH}" || true)"
if [[ -n "${PID}" ]] && kill -0 "${PID}" 2>/dev/null; then
  echo "RUNNING pid=${PID}"
else
  echo "NOT RUNNING pid=${PID}"
fi

echo "log=${LOG_PATH}"
echo "--- tail (cache/train progress) ---"
rg -n "\\[cache\\]|training process:|Saving checkpoint at step|Traceback|Error|Exception|OOM|out of memory" -S "${LOG_PATH}" | tail -n 20 || true
echo "--- tail (raw) ---"
tail -n 60 "${LOG_PATH}" || true

