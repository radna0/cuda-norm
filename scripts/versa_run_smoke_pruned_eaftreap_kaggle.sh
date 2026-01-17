#!/usr/bin/env bash
set -euo pipefail

# Smoke-test pruned EAFT-REAP checkpoints on a remote Kaggle Jupyter server via Versa.
#
# Reads variants from:
# - artifacts/20b_pruned_models_eaftreap_keepfrac/manifest_eaftreap_keepfrac.json (preferred)
# - artifacts/20b_pruned_models_eaftreap/manifest_eaftreap.json (fallback)
#
# Required:
#   export REMOTE_JUPYTER_URL="https://.../proxy"
#
# Optional:
#   export REMOTE_JUPYTER_TOKEN=""
#   export SMOKE_REMOTE_LOG_DIR="logs"
#   export SMOKE_ENV_FILE="/home/kojoe/harmony/cuda-norm/.env"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"
SYNC_DIR="${ROOT_DIR}/.versa_sync_min"

if [[ -z "${REMOTE_JUPYTER_URL:-}" ]]; then
  echo "[err] REMOTE_JUPYTER_URL is not set" >&2
  exit 2
fi

SMOKE_REMOTE_LOG_DIR="${SMOKE_REMOTE_LOG_DIR:-logs}"
SMOKE_ENV_FILE="${SMOKE_ENV_FILE:-${ROOT_DIR}/.env}"

TS="$(date +%Y%m%d_%H%M%S)"
REMOTE_LOG="${SMOKE_REMOTE_LOG_DIR}/smoke_pruned_eaftreap_${TS}.log"

PYTHONPATH="${REPO_ROOT}/third_party/Versa" \
python -m versa run \
  --backend jupyter \
  --url "${REMOTE_JUPYTER_URL}" \
  ${REMOTE_JUPYTER_TOKEN:+--token "${REMOTE_JUPYTER_TOKEN}"} \
  --cwd "/kaggle/working" \
  --sync-local-dir "${SYNC_DIR}" \
  --sync-remote-dir "cuda-norm" \
  --log-path "${REMOTE_LOG}" \
  --detach \
  --bootstrap-cmd "mkdir -p ${SMOKE_REMOTE_LOG_DIR}" \
  --bootstrap-cmd "python -m pip install -q torch transformers safetensors" \
  --env-file "${SMOKE_ENV_FILE}" \
  --env "MODEL_DIR_20B=/kaggle/input/gpt-oss-20b/transformers/default/1" \
  "python -u cuda-norm/scripts/kaggle_smoke_pruned_eaftreap.py"

echo "[+] started smoke"
echo "    remote_log=${REMOTE_LOG}"
