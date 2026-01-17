#!/usr/bin/env bash
set -euo pipefail

# Sync the minimal `harmony/cuda-norm` tree (/.versa_sync_min) to a remote Kaggle
# Jupyter server via Versa. This avoids re-installing packages.
#
# Required:
#   export REMOTE_JUPYTER_URL="https://.../proxy"
# Optional:
#   export REMOTE_JUPYTER_TOKEN=""
#
# Example:
#   export REMOTE_JUPYTER_URL="https://.../proxy"
#   bash harmony/cuda-norm/scripts/versa_sync_min_to_kaggle.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"
SYNC_DIR="${ROOT_DIR}/.versa_sync_min"

if [[ -z "${REMOTE_JUPYTER_URL:-}" ]]; then
  echo "REMOTE_JUPYTER_URL is not set" >&2
  exit 2
fi

PYTHONPATH="${REPO_ROOT}/third_party/Versa" \
python -m versa run \
  --backend jupyter \
  --url "${REMOTE_JUPYTER_URL}" \
  ${REMOTE_JUPYTER_TOKEN:+--token "${REMOTE_JUPYTER_TOKEN}"} \
  --sync-local-dir "${SYNC_DIR}" \
  --sync-remote-dir "cuda-norm" \
  --bootstrap-cmd "mkdir -p /kaggle/working/logs" \
  python -c "print('synced cuda-norm/.versa_sync_min')"

