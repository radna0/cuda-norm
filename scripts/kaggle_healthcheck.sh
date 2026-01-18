#!/usr/bin/env bash
set -euo pipefail

# Minimal healthcheck for a Kaggle Jupyter /proxy URL in harmony/cuda-norm/.env.
#
# Returns:
# - 0 if /api/sessions is reachable and returns JSON
# - 2 otherwise
#
# Usage:
#   bash harmony/cuda-norm/scripts/kaggle_healthcheck.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

set -a
source "${ROOT_DIR}/.env"
set +a

if [[ -z "${KAGGLE_URL:-}" ]]; then
  echo "[err] KAGGLE_URL missing in ${ROOT_DIR}/.env" >&2
  exit 2
fi

BASE="${KAGGLE_URL%/}"

echo "[*] probing Kaggle /proxy (URL redacted)" >&2

json="$(curl -fsSL --connect-timeout 5 --max-time 15 "${BASE}/api/sessions" || true)"
if [[ -z "${json}" ]]; then
  echo "[err] /api/sessions timed out/failed (kernel likely asleep or URL expired)" >&2
  exit 2
fi

printf "%s" "${json}" | python - <<'PY'
import json, sys
data=json.loads(sys.stdin.read())
print(f"[+] sessions={len(data) if isinstance(data,list) else '??'}")
if isinstance(data, list) and data:
    k=(data[0].get("kernel") or {}).get("id")
    if k:
        print(f"[+] kernel_id={k}")
PY

exit 0

