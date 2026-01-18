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

payload="$(curl -sS -L --connect-timeout 5 --max-time 15 "${BASE}/api/sessions" || true)"
payload_stripped="$(printf "%s" "${payload}" | tr -d '\r' | sed -E 's/^[[:space:]]+//')"
if [[ -z "${payload_stripped}" ]]; then
  echo "[err] /api/sessions timed out/failed or returned empty (kernel likely asleep or URL expired)" >&2
  exit 2
fi

python - <<'PY' "${payload_stripped}"
import json
import sys

raw = sys.argv[1]
try:
    data = json.loads(raw)
except Exception as e:
    head = raw[:200].replace("\n", " ")
    print(f"[err] /api/sessions returned non-JSON: {e}; head={head!r}", file=sys.stderr)
    raise SystemExit(2)

print(f"[+] sessions={len(data) if isinstance(data, list) else '??'}")
if isinstance(data, list) and data and isinstance(data[0], dict):
    kernel = data[0].get("kernel") or {}
    if isinstance(kernel, dict) and kernel.get("id"):
        print(f"[+] kernel_id={kernel['id']}")
PY

exit 0
