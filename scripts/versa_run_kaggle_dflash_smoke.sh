#!/usr/bin/env bash
set -euo pipefail

# Run the end-to-end DFlash (HF->SGLang) smoke test on a remote Kaggle Jupyter
# server via Versa.
#
# Required:
#   export REMOTE_JUPYTER_URL="https://.../proxy"
# Optional:
#   export REMOTE_JUPYTER_TOKEN=""
#
# This submits ONE remote job which:
#   - syncs `harmony/cuda-norm/` to /kaggle/working/cuda-norm-sync
#   - runs `scripts/kaggle_dflash_smoke.sh` remotely
#
# Output:
#   - Versa local log: `harmony/cuda-norm/logs/versa_dflash_smoke.log`

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"

if [[ -z "${REMOTE_JUPYTER_URL:-}" ]]; then
  echo "REMOTE_JUPYTER_URL is not set"
  exit 2
fi

mkdir -p "${ROOT_DIR}/logs"

SYNC_DIR="$(python "${ROOT_DIR}/scripts/versa_prepare_sync_min.py" --out "${ROOT_DIR}/.versa_sync_min")"

PYTHONPATH="${REPO_ROOT}/third_party/Versa" \
python -m versa run \
  --backend jupyter \
  --url "${REMOTE_JUPYTER_URL}" \
  ${REMOTE_JUPYTER_TOKEN:+--token "${REMOTE_JUPYTER_TOKEN}"} \
  --sync-local-dir "${SYNC_DIR}" \
  --sync-remote-dir "cuda-norm-sync" \
  --log-path "logs/dflash_smoke.log" \
  bash /kaggle/working/cuda-norm-sync/scripts/kaggle_dflash_smoke.sh

echo "submitted. remote log: logs/dflash_smoke.log"
