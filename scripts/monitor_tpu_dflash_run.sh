#!/usr/bin/env bash
set -euo pipefail

# Monitor a TPU DFlash run started via run_tpu_dflash_train_logged.sh.
#
# Usage:
#   ./harmony/cuda-norm/scripts/monitor_tpu_dflash_run.sh gptoss20b_dflash_ctx1024_b8_k4_bs160_s2000

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_NAME="${1:?Provide RUN_NAME (e.g. gptoss20b_dflash_ctx1024_b8_k4_bs160_s2000)}"
LOG_DIR="${ROOT}/harmony/cuda-norm/logs/tpu_dflash"
LOG_PATH="${LOG_DIR}/${RUN_NAME}.log"
PID_PATH="${LOG_DIR}/${RUN_NAME}.pid"

if [[ ! -f "${LOG_PATH}" ]]; then
  echo "Missing log: ${LOG_PATH}" >&2
  exit 1
fi

if [[ -f "${PID_PATH}" ]]; then
  PID="$(cat "${PID_PATH}")"
  if ps -p "${PID}" >/dev/null 2>&1; then
    echo "RUNNING pid=${PID}"
  else
    echo "NOT RUNNING pid=${PID} (stale pid file?)"
  fi
else
  echo "No pid file at ${PID_PATH}"
fi

echo "Tailing ${LOG_PATH}"
exec tail -n 200 -f "${LOG_PATH}"

