#!/usr/bin/env bash
set -euo pipefail

# Watch a running Kaggle/VERSA pruning job by repeatedly tailing its remote log
# via Versa (reusing the same kernel), and mirror the tail locally.
#
# Required env:
#   REMOTE_JUPYTER_URL="https://.../proxy"
#
# Usage:
#   bash scripts/kaggle_watch_prune_job.sh \
#     --kernel-id <kernel> \
#     --remote-log logs/<job>.log \
#     --local-log logs/<mirror>.log
#
# Notes:
# - This is intentionally "dumb": it does not start/stop the prune job, only
#   monitors it. Use CTRL+C to stop the watcher.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"

if [[ -z "${REMOTE_JUPYTER_URL:-}" ]]; then
  echo "[err] REMOTE_JUPYTER_URL is not set" >&2
  exit 2
fi

KERNEL_ID=""
REMOTE_LOG=""
LOCAL_LOG=""
INTERVAL_S="120"
TAIL_N="40"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel-id) KERNEL_ID="$2"; shift 2;;
    --remote-log) REMOTE_LOG="$2"; shift 2;;
    --local-log) LOCAL_LOG="$2"; shift 2;;
    --interval-s) INTERVAL_S="$2"; shift 2;;
    --tail-n) TAIL_N="$2"; shift 2;;
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

if [[ -z "${KERNEL_ID}" || -z "${REMOTE_LOG}" || -z "${LOCAL_LOG}" ]]; then
  echo "[err] --kernel-id, --remote-log, --local-log are required" >&2
  exit 2
fi

mkdir -p "$(dirname "${LOCAL_LOG}")"
echo "[*] Watching remote log: ${REMOTE_LOG}" | tee -a "${LOCAL_LOG}"
echo "    kernel_id=${KERNEL_ID}" | tee -a "${LOCAL_LOG}"
echo "    interval_s=${INTERVAL_S} tail_n=${TAIL_N}" | tee -a "${LOCAL_LOG}"

while :; do
  TS="$(date +%Y-%m-%dT%H:%M:%S%z)"
  echo "" | tee -a "${LOCAL_LOG}"
  echo "===== ${TS} =====" | tee -a "${LOCAL_LOG}"

  PYTHONPATH="${REPO_ROOT}/third_party/Versa" \
  python -m versa run \
    --backend jupyter \
    --url "${REMOTE_JUPYTER_URL}" \
    ${REMOTE_JUPYTER_TOKEN:+--token "${REMOTE_JUPYTER_TOKEN}"} \
    --kernel-id "${KERNEL_ID}" \
    --cwd "/kaggle/working" \
    "bash -lc \"tail -n ${TAIL_N} ${REMOTE_LOG} || true\"" \
    | tee -a "${LOCAL_LOG}" || true

  # Stop automatically if the job reports completion.
  if rg -n "\\[\\+\\] structural_prune done|\\[\\+\\] Wrote reports/20b_structural_prune_build_eaftreap_budgeted\\.md" -S "${LOCAL_LOG}" >/dev/null 2>&1; then
    echo "[+] Detected completion marker in log. Exiting watcher." | tee -a "${LOCAL_LOG}"
    exit 0
  fi

  sleep "${INTERVAL_S}"
done

