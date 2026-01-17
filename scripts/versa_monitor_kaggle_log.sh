#!/usr/bin/env bash
set -euo pipefail

# Monitor a remote Kaggle Jupyter log file via Versa.
#
# Required env:
#   REMOTE_JUPYTER_URL="https://.../proxy"
# Optional:
#   REMOTE_JUPYTER_TOKEN=""
#
# Usage:
#   bash harmony/cuda-norm/scripts/versa_monitor_kaggle_log.sh \
#     --kernel-id <kernel_id> \
#     --remote-log logs/eaft_single_openai_gpt-oss-20b_YYYYmmdd_HHMMSS.log \
#     --pattern "\\[\\+\\] Wrote " \
#     --interval-s 30

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"

if [[ -z "${REMOTE_JUPYTER_URL:-}" ]]; then
  echo "[err] REMOTE_JUPYTER_URL is not set" >&2
  exit 2
fi

KERNEL_ID=""
REMOTE_LOG=""
PATTERN="\\[\\+\\] Wrote "
INTERVAL_S="30"
TAIL_N="80"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel-id) KERNEL_ID="$2"; shift 2;;
    --remote-log) REMOTE_LOG="$2"; shift 2;;
    --pattern) PATTERN="$2"; shift 2;;
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

if [[ -z "${KERNEL_ID}" || -z "${REMOTE_LOG}" ]]; then
  echo "[err] --kernel-id and --remote-log are required" >&2
  exit 2
fi

while :; do
  echo "===== $(date -Is) ${REMOTE_LOG} ====="
  OUT="$(
    PYTHONPATH="${REPO_ROOT}/third_party/Versa" python -m versa run \
      --backend jupyter \
      --url "${REMOTE_JUPYTER_URL}" \
      ${REMOTE_JUPYTER_TOKEN:+--token "${REMOTE_JUPYTER_TOKEN}"} \
      --kernel-id "${KERNEL_ID}" \
      --cwd "/kaggle/working" \
      bash -lc "tail -n ${TAIL_N} '${REMOTE_LOG}'" \
      | rg -v '^\\[versa\\]' || true
  )"
  printf "%s\n" "${OUT}" | tail -n "${TAIL_N}"
  if printf "%s\n" "${OUT}" | rg -n "${PATTERN}" >/dev/null 2>&1; then
    echo "[+] matched pattern: ${PATTERN}"
    exit 0
  fi
  sleep "${INTERVAL_S}"
done

