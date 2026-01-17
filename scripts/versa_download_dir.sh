#!/usr/bin/env bash
set -euo pipefail

# Download a remote directory from Kaggle Jupyter (/proxy) to local disk via Versa.
#
# Required env:
#   REMOTE_JUPYTER_URL="https://.../proxy"
#
# Usage:
#   bash scripts/versa_download_dir.sh --kernel-id <id> --remote-dir <path> --local-dir <path>
#

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"

if [[ -z "${REMOTE_JUPYTER_URL:-}" ]]; then
  echo "[err] REMOTE_JUPYTER_URL is not set" >&2
  exit 2
fi

KERNEL_ID=""
REMOTE_DIR=""
LOCAL_DIR=""
LOG_PATH="logs/versa_download_dir_$(date +%Y%m%d_%H%M%S).log"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel-id) KERNEL_ID="$2"; shift 2;;
    --remote-dir) REMOTE_DIR="$2"; shift 2;;
    --local-dir) LOCAL_DIR="$2"; shift 2;;
    --log-path) LOG_PATH="$2"; shift 2;;
    -h|--help)
      sed -n '1,140p' "$0"
      exit 0
      ;;
    *)
      echo "[err] Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${KERNEL_ID}" || -z "${REMOTE_DIR}" || -z "${LOCAL_DIR}" ]]; then
  echo "[err] --kernel-id, --remote-dir, and --local-dir are required" >&2
  exit 2
fi

mkdir -p "$(dirname "${LOG_PATH}")" "${LOCAL_DIR}"

# IMPORTANT: Versa's downloader expects a *relative* path for the /files handler.
REMOTE_DIR="${REMOTE_DIR#/}"
REMOTE_DIR="${REMOTE_DIR#/kaggle/working/}"

PYTHONPATH="${REPO_ROOT}/third_party/Versa" \
python -m versa run \
  --backend jupyter \
  --url "${REMOTE_JUPYTER_URL}" \
  ${REMOTE_JUPYTER_TOKEN:+--token "${REMOTE_JUPYTER_TOKEN}"} \
  --kernel-id "${KERNEL_ID}" \
  --cwd "/kaggle/working" \
  --bootstrap-cmd "mkdir -p logs agent_artifacts" \
  --log-path "${LOG_PATH}" \
  --download-remote-dir "${REMOTE_DIR}" \
  --download-local-dir "${LOCAL_DIR}" \
  python -c "print('download ok')"

