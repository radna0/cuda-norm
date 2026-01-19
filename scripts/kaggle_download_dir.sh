#!/usr/bin/env bash
set -euo pipefail

# Recursively download a directory tree from a Kaggle Jupyter (/proxy) server.
#
# Required env:
#   REMOTE_JUPYTER_URL="https://.../proxy"
#   (also accepts KAGGLE_URL as fallback)
#
# Usage:
#   bash harmony/cuda-norm/scripts/kaggle_download_dir.sh \
#     --remote-dir /kaggle/working/<dir> \
#     --out-dir /dev/shm/<dir>
#
# Notes:
# - `remote-dir` can be absolute (/kaggle/working/...) or relative to /kaggle/working.
# - The directory structure is preserved under out-dir.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${REMOTE_JUPYTER_URL:-}" ]]; then
  export REMOTE_JUPYTER_URL="${KAGGLE_URL:-}"
fi
if [[ -z "${REMOTE_JUPYTER_URL:-}" ]]; then
  echo "[err] REMOTE_JUPYTER_URL/KAGGLE_URL is not set" >&2
  exit 2
fi

REMOTE_DIR=""
OUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote-dir) REMOTE_DIR="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    -h|--help)
      sed -n '1,160p' "$0"
      exit 0
      ;;
    *)
      echo "[err] Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${REMOTE_DIR}" || -z "${OUT_DIR}" ]]; then
  echo "[err] --remote-dir and --out-dir are required" >&2
  exit 2
fi

# Normalize to a /kaggle/working-relative path for API calls.
REMOTE_DIR_REL="${REMOTE_DIR#/}"
REMOTE_DIR_REL="${REMOTE_DIR_REL#/kaggle/working/}"
REMOTE_DIR_REL="${REMOTE_DIR_REL#kaggle/working/}"

mkdir -p "${OUT_DIR}"

FILELIST="$(mktemp -t kaggle_dirlist_XXXXXX.txt)"
cleanup_tmp() { rm -f "${FILELIST}" 2>/dev/null || true; }
trap cleanup_tmp EXIT

export _KAGGLE_DL_BASE_URL="${REMOTE_JUPYTER_URL%/}"
export _KAGGLE_DL_REMOTE_DIR_REL="${REMOTE_DIR_REL}"

python - <<'PY' >"${FILELIST}"
import json
import os
import sys
import urllib.parse
import urllib.request

base = os.environ["_KAGGLE_DL_BASE_URL"]
root = os.environ["_KAGGLE_DL_REMOTE_DIR_REL"].strip("/")

def fetch(path: str) -> dict:
    q = urllib.parse.quote(path, safe="/")
    url = f"{base}/api/contents/{q}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8", errors="ignore"))

seen = set()

def walk(path: str) -> None:
    if path in seen:
        return
    seen.add(path)
    data = fetch(path)
    typ = data.get("type")
    if typ == "file":
        print(path)
        return
    if typ != "directory":
        raise SystemExit(f"[err] Unexpected type for {path}: {typ}")
    content = data.get("content") or []
    for item in content:
        if not isinstance(item, dict):
            continue
        p = (item.get("path") or "").strip("/")
        t = item.get("type")
        if not p:
            continue
        if t == "file":
            print(p)
        elif t == "directory":
            walk(p)

walk(root)
PY

count="$(wc -l <"${FILELIST}" | tr -d ' ')"
echo "[*] remote_dir=${REMOTE_DIR_REL}" >&2
echo "[*] files=${count}" >&2

while IFS= read -r remote_path; do
  [[ -n "${remote_path}" ]] || continue
  rel="${remote_path#${REMOTE_DIR_REL}/}"
  out="${OUT_DIR}/${rel}"
  mkdir -p "$(dirname "${out}")"
  bash "${ROOT_DIR}/scripts/kaggle_download_file.sh" --remote-path "${remote_path}" --out "${out}"
done <"${FILELIST}"

echo "[+] wrote_dir ${OUT_DIR}" >&2

