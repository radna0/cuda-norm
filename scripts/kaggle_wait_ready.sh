#!/usr/bin/env bash
set -euo pipefail

# Wait for KAGGLE_URL in harmony/cuda-norm/.env to become reachable, polling
# /api/sessions every N seconds.
#
# Always run this script under nohup if you want it to keep running.
#
# Usage:
#   bash harmony/cuda-norm/scripts/kaggle_wait_ready.sh --interval-s 120

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL_S="120"
MAX_ITERS="0" # 0 = infinite

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval-s) INTERVAL_S="$2"; shift 2;;
    --max-iters) MAX_ITERS="$2"; shift 2;;
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

i=0
while :; do
  i=$((i+1))
  echo "===== $(date -Is) kaggle_wait_ready iter=${i} ====="
  if bash "${ROOT_DIR}/scripts/kaggle_healthcheck.sh"; then
    echo "[+] kaggle is reachable"
    exit 0
  fi
  if [[ "${MAX_ITERS}" != "0" && "${i}" -ge "${MAX_ITERS}" ]]; then
    echo "[err] max-iters reached without becoming reachable" >&2
    exit 2
  fi
  sleep "${INTERVAL_S}"
done

