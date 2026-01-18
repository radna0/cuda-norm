#!/usr/bin/env bash
set -euo pipefail

# Poll a Kaggle Jupyter /proxy /files/<remote_path> endpoint until a pattern appears.
#
# Required env:
#   KAGGLE_URL="https://.../proxy" (or pass --base-url)
#
# Usage:
#   bash scripts/kaggle_watch_file.sh --remote-path logs/foo.log --pattern "Wrote reports/" --interval-s 60

BASE_URL="${KAGGLE_URL:-}"
REMOTE_PATH=""
PATTERN=""
FAIL_PATTERN=""
INTERVAL_S="60"
TAIL_N="120"
MAX_ITERS="0" # 0 = infinite

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) BASE_URL="$2"; shift 2;;
    --remote-path) REMOTE_PATH="$2"; shift 2;;
    --pattern) PATTERN="$2"; shift 2;;
    --fail-pattern) FAIL_PATTERN="$2"; shift 2;;
    --interval-s) INTERVAL_S="$2"; shift 2;;
    --tail-n) TAIL_N="$2"; shift 2;;
    --max-iters) MAX_ITERS="$2"; shift 2;;
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

if [[ -z "${BASE_URL}" ]]; then
  echo "[err] KAGGLE_URL/--base-url not set" >&2
  exit 2
fi
if [[ -z "${REMOTE_PATH}" || -z "${PATTERN}" ]]; then
  echo "[err] --remote-path and --pattern are required" >&2
  exit 2
fi

BASE_URL="${BASE_URL%/}"
REMOTE_PATH="${REMOTE_PATH#/}"
REMOTE_PATH="${REMOTE_PATH#/kaggle/working/}"

URL="${BASE_URL}/files/${REMOTE_PATH}"

i=0
while :; do
  i=$((i+1))
  echo "===== $(date -Is) ${REMOTE_PATH} (iter=${i}) ====="

  tmp="$(mktemp)"
  ok="1"
  if ! curl -fsL --retry 6 --retry-all-errors --retry-delay 2 -o "${tmp}" "${URL}"; then
    ok="0"
    echo "[warn] fetch failed: ${URL}" >&2
  fi

  if [[ "${ok}" == "1" ]]; then
    tail -n "${TAIL_N}" "${tmp}" || true
    set +e
    python - <<PY
import re
from pathlib import Path
txt=Path("${tmp}").read_text(encoding="utf-8", errors="ignore")
pat=r'''${PATTERN}'''
if re.search(pat, txt, flags=re.M):
    raise SystemExit(0)
fail=r'''${FAIL_PATTERN}'''
if fail and re.search(fail, txt, flags=re.M):
    raise SystemExit(2)
raise SystemExit(1)
PY
    rc="$?"
    set -e
    if [[ "${rc}" == "0" ]]; then
      echo "[+] matched pattern: ${PATTERN}"
      rm -f "${tmp}"
      exit 0
    fi
    if [[ "${rc}" == "2" ]]; then
      echo "[err] matched fail pattern: ${FAIL_PATTERN}" >&2
      rm -f "${tmp}"
      exit 2
    fi
  fi

  rm -f "${tmp}"
  if [[ "${MAX_ITERS}" != "0" && "${i}" -ge "${MAX_ITERS}" ]]; then
    echo "[err] max-iters reached without match" >&2
    exit 1
  fi
  sleep "${INTERVAL_S}"
done
