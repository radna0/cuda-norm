#!/usr/bin/env bash
set -euo pipefail

# Tail one or more local log files every N seconds (default: 120s).
# Always run under nohup and write output to a monitor log.
#
# Usage:
#   bash scripts/tail_every_2m.sh --interval-s 120 --tail-n 120 -- files...

INTERVAL_S="120"
TAIL_N="120"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval-s) INTERVAL_S="$2"; shift 2;;
    --tail-n) TAIL_N="$2"; shift 2;;
    --) shift; break;;
    -h|--help)
      sed -n '1,120p' "$0"
      exit 0
      ;;
    *)
      # first non-flag starts file list
      break
      ;;
  esac
done

if [[ $# -lt 1 ]]; then
  echo "[err] provide at least 1 log file path" >&2
  exit 2
fi

FILES=("$@")

while :; do
  echo "===== $(date -Is) ====="
  for f in "${FILES[@]}"; do
    echo "----- ${f} (tail ${TAIL_N}) -----"
    if [[ -f "${f}" ]]; then
      tail -n "${TAIL_N}" "${f}" || true
    else
      echo "[missing] ${f}"
    fi
    echo
  done
  sleep "${INTERVAL_S}"
done

