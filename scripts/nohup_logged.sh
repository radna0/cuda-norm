#!/usr/bin/env bash
set -euo pipefail

# Run a command under nohup, write logs + pid files, and return immediately.
#
# Usage:
#   bash scripts/nohup_logged.sh --name <slug> --log-dir <dir> -- <cmd...>
#
# Example:
#   bash scripts/nohup_logged.sh --name eaft_noise_A --log-dir unsloth_logs -- bash scripts/versa_run_eaft_single_kaggle.sh ...
#

NAME=""
LOG_DIR="unsloth_logs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --) shift; break;;
    -h|--help)
      sed -n '1,120p' "$0"
      exit 0
      ;;
    *)
      echo "[err] Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${NAME}" ]]; then
  echo "[err] --name is required" >&2
  exit 2
fi
if [[ $# -lt 1 ]]; then
  echo "[err] Missing command after --" >&2
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${NAME}_${TS}.log"
PID_PATH="${LOG_PATH}.pid"

nohup "$@" >"${LOG_PATH}" 2>&1 &
echo $! >"${PID_PATH}"

echo "[+] started (nohup)"
echo "    log=${LOG_PATH}"
echo "    pid=$(cat "${PID_PATH}")"

