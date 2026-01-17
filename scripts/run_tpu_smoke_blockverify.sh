#!/usr/bin/env bash
set -euo pipefail

# TPU smoke for DFlash decode using block-parallel verify (TARGET_VERIFY-style).
#
# Usage:
#   RUN_NAME=smoke_blockverify ./harmony/cuda-norm/scripts/run_tpu_smoke_blockverify.sh \
#     --teacher-snapshot-dir /path/to/teacher_snapshot \
#     --draft-run-dir /path/to/draft_run_dir
#
# Outputs logs to `harmony/cuda-norm/logs/tpu_dflash/`.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PY="${ROOT}/harmony/cuda-norm/.venv-easydel/bin/python"
LOG_DIR="${ROOT}/harmony/cuda-norm/logs/tpu_dflash"
mkdir -p "${LOG_DIR}"

RUN_NAME="${RUN_NAME:-tpu_dflash_smoke_blockverify}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/${RUN_NAME}_${TS}.log"

echo "[+] Writing log to ${LOG}"
echo "${LOG}" > "${LOG}.path"

nohup "${VENV_PY}" -u "${ROOT}/harmony/cuda-norm/scripts/tpu_dflash_spec_decode_blockverify_v1.py" \
  --max-new-tokens 64 \
  --max-prompt-len 512 \
  --block-size 8 \
  --also-run-baseline \
  "$@" > "${LOG}" 2>&1 & disown

echo "$!" > "${LOG}.pid"
echo "[+] PID $(cat "${LOG}.pid")"
echo "[+] Tail: tail -n 200 -f ${LOG}"
