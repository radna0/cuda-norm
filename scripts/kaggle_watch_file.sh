#!/usr/bin/env bash
set -euo pipefail

# Poll a Kaggle Jupyter /proxy endpoint until a pattern appears.
#
# Required env:
#   KAGGLE_URL="https://.../proxy" (or pass --base-url)
#   (also accepts REMOTE_JUPYTER_URL as a fallback)
#
# Usage:
#   bash scripts/kaggle_watch_file.sh --remote-path logs/foo.log --pattern "Wrote reports/" --interval-s 60

BASE_URL="${KAGGLE_URL:-${REMOTE_JUPYTER_URL:-}}"
ENV_FILE=""
REMOTE_PATH=""
PATTERN=""
FAIL_PATTERN=""
INTERVAL_S="60"
TAIL_N="120"
MAX_ITERS="0" # 0 = infinite
CONNECT_TIMEOUT_S="10"
MAX_TIME_S="60"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file) ENV_FILE="$2"; shift 2;;
    --base-url) BASE_URL="$2"; shift 2;;
    --remote-path) REMOTE_PATH="$2"; shift 2;;
    --pattern) PATTERN="$2"; shift 2;;
    --fail-pattern) FAIL_PATTERN="$2"; shift 2;;
    --interval-s) INTERVAL_S="$2"; shift 2;;
    --tail-n) TAIL_N="$2"; shift 2;;
    --max-iters) MAX_ITERS="$2"; shift 2;;
    --connect-timeout-s) CONNECT_TIMEOUT_S="$2"; shift 2;;
    --max-time-s) MAX_TIME_S="$2"; shift 2;;
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

if [[ -n "${ENV_FILE}" ]]; then
  if [[ ! -f "${ENV_FILE}" ]]; then
    echo "[err] --env-file does not exist: ${ENV_FILE}" >&2
    exit 2
  fi
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
  BASE_URL="${BASE_URL:-${KAGGLE_URL:-${REMOTE_JUPYTER_URL:-}}}"
fi

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
REMOTE_PATH="${REMOTE_PATH#kaggle/working/}"

FILES_URL="${BASE_URL}/files/${REMOTE_PATH}"
CONTENTS_URL="${BASE_URL}/api/contents/${REMOTE_PATH}?content=1"

fetch_to_tmp() {
  local out_path="$1"
  local json_tmp="${out_path}.json"

  # Prefer /files (fast path), but fall back to Jupyter Contents API when Kaggle
  # /files intermittently 5xx/timeouts.
  # Try /files first, but avoid `--retry-all-errors` so 404s don't spin for a
  # long time; we fall back to /api/contents in that case.
  if curl -fsL --retry 2 --retry-delay 2 \
    --connect-timeout "${CONNECT_TIMEOUT_S}" --max-time "${MAX_TIME_S}" \
    -o "${out_path}" "${FILES_URL}"; then
    return 0
  fi

  if ! curl -fsL --retry 6 --retry-all-errors --retry-delay 2 \
    --connect-timeout "${CONNECT_TIMEOUT_S}" --max-time "${MAX_TIME_S}" \
    -o "${json_tmp}" "${CONTENTS_URL}"; then
    rm -f "${json_tmp}" || true
    return 1
  fi

  python - <<'PY' "${json_tmp}" "${out_path}"
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
    raise SystemExit(1)
if fmt == "base64":
    dst.write_bytes(base64.b64decode(content))
else:
    dst.write_text(str(content), encoding="utf-8")
PY
  rm -f "${json_tmp}" || true
  return 0
}

i=0
while :; do
  i=$((i+1))
  echo "===== $(date -Is) ${REMOTE_PATH} (iter=${i}) ====="

  tmp="$(mktemp)"
  if ! fetch_to_tmp "${tmp}"; then
    # Do not print the full URL (it includes an auth token in KAGGLE_URL).
    echo "[warn] fetch failed: remote_path=${REMOTE_PATH}" >&2
  else
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
