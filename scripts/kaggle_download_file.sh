#!/usr/bin/env bash
set -euo pipefail

# Download a single file from a Kaggle Jupyter (/proxy) server via its /files handler.
#
# Required env:
#   REMOTE_JUPYTER_URL="https://.../proxy"
#
# Usage:
#   bash scripts/kaggle_download_file.sh --remote-path <path> --out <local_path>
#
# Notes:
# - `remote-path` should be a filesystem-relative path as seen from /kaggle/working
#   (e.g., harmony/cuda-norm/artifacts/eaft_models/<run_id>/<name>.json).
#

if [[ -z "${REMOTE_JUPYTER_URL:-}" ]]; then
  echo "[err] REMOTE_JUPYTER_URL is not set" >&2
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

mkdir -p "$(dirname "${OUT}")"

BASE="${REMOTE_JUPYTER_URL%/}"
URL="${BASE}/files/${REMOTE_PATH}"

tmp="${OUT}.part"

echo "[*] GET ${URL}" >&2
curl -fL --retry 8 --retry-all-errors --retry-delay 2 -C - -o "${tmp}" "${URL}"
mv -f "${tmp}" "${OUT}"
echo "[+] wrote ${OUT}" >&2

