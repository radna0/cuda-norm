#!/usr/bin/env bash
set -euo pipefail

# Download a single file from a Kaggle Jupyter (/proxy) server via its /files handler.
#
# Required env:
#   REMOTE_JUPYTER_URL="https://.../proxy"
#   (also accepts KAGGLE_URL as a fallback)
#
# Usage:
#   bash scripts/kaggle_download_file.sh --remote-path <path> --out <local_path>
#
# Notes:
# - `remote-path` should be a filesystem-relative path as seen from /kaggle/working
#   (e.g., harmony/cuda-norm/artifacts/eaft_models/<run_id>/<name>.json).
#

if [[ -z "${REMOTE_JUPYTER_URL:-}" ]]; then
  export REMOTE_JUPYTER_URL="${KAGGLE_URL:-}"
fi

if [[ -z "${REMOTE_JUPYTER_URL:-}" ]]; then
  echo "[err] REMOTE_JUPYTER_URL/KAGGLE_URL is not set" >&2
  exit 2
fi

REMOTE_PATH=""
OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote-path) REMOTE_PATH="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
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

if [[ -z "${REMOTE_PATH}" || -z "${OUT}" ]]; then
  echo "[err] --remote-path and --out are required" >&2
  exit 2
fi

# Normalize slashes.
REMOTE_PATH="${REMOTE_PATH#/}"
REMOTE_PATH="${REMOTE_PATH#/kaggle/working/}"
REMOTE_PATH="${REMOTE_PATH#kaggle/working/}"

mkdir -p "$(dirname "${OUT}")"

BASE="${REMOTE_JUPYTER_URL%/}"
URL="${BASE}/files/${REMOTE_PATH}"

tmp="${OUT}.part"

# Do not print the full URL (it includes an auth token in KAGGLE_URL).
echo "[*] download remote_path=${REMOTE_PATH}" >&2

# Prefer the lightweight /files handler, but fall back to Jupyter Contents API
# when /files becomes flaky (Kaggle occasionally returns transient 5xx/timeouts).
#
# Contents API returns JSON with {format: "text"|"base64", content: "..."}.
download_via_contents_api() {
  local api_url="${BASE}/api/contents/${REMOTE_PATH}?content=1"
  local json_tmp="${tmp}.json"
  curl -fsSL --retry 8 --retry-all-errors --retry-delay 2 \
    --connect-timeout 10 --max-time 60 \
    -o "${json_tmp}" "${api_url}"
  python - <<'PY' "${json_tmp}" "${tmp}"
import base64
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
data = json.loads(src.read_text(encoding="utf-8", errors="ignore"))
content = data.get("content")
fmt = (data.get("format") or "").lower()
if content is None:
    raise SystemExit("[err] contents API returned no content")
if fmt == "base64":
    dst.write_bytes(base64.b64decode(content))
else:
    # "text" (or unknown) - treat as utf-8 text
    dst.write_text(str(content), encoding="utf-8")
PY
  rm -f "${json_tmp}" || true
}

# Try /files once-ish (fast path). Avoid `--retry-all-errors` here so a 404
# doesn't burn time retrying; we fall back to /api/contents in that case.
if curl -fL --retry 2 --retry-delay 2 \
  --connect-timeout 10 --max-time 60 \
  -C - -o "${tmp}" "${URL}"; then
  :
else
  echo "[warn] /files download failed; retrying via /api/contents" >&2
  download_via_contents_api
fi

mv -f "${tmp}" "${OUT}"
echo "[+] wrote ${OUT}" >&2
