#!/usr/bin/env bash
set -euo pipefail

# Print the currently active Kaggle Jupyter kernel id for the given KAGGLE_URL.
#
# Uses the Jupyter REST API exposed by Kaggle's /proxy server:
#   GET /api/sessions  -> [{..., kernel: {id: ...}}]
#
# Required:
#   harmony/cuda-norm/.env with KAGGLE_URL=...
#
# Usage:
#   bash harmony/cuda-norm/scripts/kaggle_get_active_kernel_id.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

set -a
source "${ROOT_DIR}/.env"
set +a

if [[ -z "${KAGGLE_URL:-}" ]]; then
  echo "[err] KAGGLE_URL missing in ${ROOT_DIR}/.env" >&2
  exit 2
fi

BASE="${KAGGLE_URL%/}"

json="$(curl -fsL --connect-timeout 5 --max-time 15 "${BASE}/api/sessions" || true)"
if [[ -z "${json}" ]]; then
  echo "[err] could not query /api/sessions (KAGGLE_URL may be expired or the kernel is asleep)" >&2
  exit 2
fi

printf "%s" "${json}" | python -c '
import json, sys
raw = sys.stdin.read()
try:
    data = json.loads(raw)
except Exception as e:
    raise SystemExit(f"[err] invalid JSON from /api/sessions: {e}")
if not isinstance(data, list) or not data:
    raise SystemExit("[err] no active sessions found")
kernel = (data[0].get("kernel") or {}) if isinstance(data[0], dict) else {}
kid = kernel.get("id") if isinstance(kernel, dict) else None
if not kid:
    raise SystemExit("[err] missing kernel.id in /api/sessions payload")
print(kid, end="")
'
