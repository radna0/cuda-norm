#!/usr/bin/env bash
set -euo pipefail

# Versa remote Jupyter smoke test (Kaggle).
#
# Required:
#   export REMOTE_JUPYTER_URL="https://.../proxy"
# Optional:
#   export REMOTE_JUPYTER_TOKEN=""
#
# This will:
#   - create a kernel (if needed)
#   - run a small env/GPU check

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"

if [[ -z "${REMOTE_JUPYTER_URL:-}" ]]; then
  echo "REMOTE_JUPYTER_URL is not set"
  exit 2
fi

PYTHONPATH="${REPO_ROOT}/third_party/Versa" \
python -m versa run \
  --backend jupyter \
  --url "${REMOTE_JUPYTER_URL}" \
  ${REMOTE_JUPYTER_TOKEN:+--token "${REMOTE_JUPYTER_TOKEN}"} \
  --detach \
  --log-path "logs/versa_smoke.log" \
  --bootstrap-cmd "python -V" \
  --bootstrap-cmd "python -c \"import sys; print(sys.executable)\"" \
  --bootstrap-cmd "nvidia-smi || true" \
  --bootstrap-cmd "python -c \"import torch; print('torch', torch.__version__)\" || true" \
  python -c "print('hello from versa remote')"

echo "submitted. tail logs via Versa MCP or check remote logs/versa_smoke.log"
