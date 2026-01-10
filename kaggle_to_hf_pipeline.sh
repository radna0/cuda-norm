#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline:
#   Kaggle model instance version -> /dev/shm tarball -> extract -> upload to HF model repo
#
# Notes:
# - Kaggle CLI requires credentials (~/.config/kaggle/kaggle.json or env vars).
# - HF upload requires you to be logged in (`hf auth login`) beforehand.
# - This script never prints or stores HF tokens.
#
# Usage:
#   ./kaggle_to_hf_pipeline.sh [model_instance_version] [hf_repo_id] [out_dir]
#
# Example:
#   ./kaggle_to_hf_pipeline.sh \
#     reyvan14/gpt-oss-120b-math/transformers/default/2 \
#     radna0/gpt-oss-120b-math \
#     /dev/shm/gpt-oss-120b-math-kaggle-v2

MODEL_INSTANCE_VERSION="${1:-reyvan14/gpt-oss-120b-math/transformers/default/2}"
HF_REPO_ID="${2:-radna0/gpt-oss-120b-math}"
OUT_DIR="${3:-/dev/shm/gpt-oss-120b-math-kaggle-v2}"
POLL_SECONDS="${POLL_SECONDS:-60}"
KEEP_TARBALL="${KEEP_TARBALL:-0}"
HF_PRIVATE="${HF_PRIVATE:-0}"
HF_TOKEN="${HF_TOKEN:-}"

if [[ -z "$OUT_DIR" || "$OUT_DIR" == "/" ]]; then
  echo "[err] unsafe OUT_DIR: '$OUT_DIR'" >&2
  exit 2
fi

for cmd in curl tar hf awk stat tee nohup python3; do
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "[err] missing required command: $cmd" >&2
    exit 2
  }
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_SCRIPT="${SCRIPT_DIR}/download_kaggle_model_to_shm.sh"
if [[ ! -x "$DOWNLOAD_SCRIPT" ]]; then
  echo "[err] missing or not executable: $DOWNLOAD_SCRIPT" >&2
  exit 2
fi

TARBALL="${OUT_DIR}/model.tar.gz"
PID_FILE="${OUT_DIR}/download.pid"
DOWNLOAD_LOG="${OUT_DIR}/download.log"
PIPELINE_LOG="${OUT_DIR}/pipeline.log"
EXTRACT_DIR="${OUT_DIR}/extracted"
EXTRACT_OK="${OUT_DIR}/.extracted.ok"

mkdir -p "$OUT_DIR"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

# Log to both stdout and a file (append)
exec > >(tee -a "$PIPELINE_LOG") 2>&1

KAGGLE_URL="https://www.kaggle.com/api/v1/models/${MODEL_INSTANCE_VERSION}/download"

remote_total_bytes() {
  # Best-effort: get the total size from the redirected GCS response.
  # Returns empty string if it can't be determined.
  curl -sS -L -D - -o /dev/null --range 0-0 --max-time 30 --retry 3 --retry-delay 1 "$KAGGLE_URL" \
    | awk 'BEGIN{IGNORECASE=1} /^x-goog-stored-content-length:/ {v=$2} END{gsub(/\r/,"",v); print v}'
}

is_pid_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ -n "${pid}" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

start_or_resume_download() {
  log "[download] starting/resuming -> ${TARBALL}"
  # Keep download log separate; append so resumes are visible.
  nohup "$DOWNLOAD_SCRIPT" "$MODEL_INSTANCE_VERSION" "$OUT_DIR" "$(basename "$TARBALL")" \
    >>"$DOWNLOAD_LOG" 2>&1 &
  echo $! >"$PID_FILE"
  sleep 1
}

hf_is_authed() {
  HF_TOKEN="$HF_TOKEN" python3 - <<'PY' >/dev/null 2>&1
import os
from huggingface_hub import HfApi

token_env = os.environ.get("HF_TOKEN") or ""
token = token_env if token_env else True  # True => use locally cached token
HfApi().whoami(token=token)
PY
}

log "[start] model=${MODEL_INSTANCE_VERSION} repo=${HF_REPO_ID} out=${OUT_DIR}"

already_extracted=0
if [[ -f "$EXTRACT_OK" && -d "$EXTRACT_DIR" && -n "$(ls -A "$EXTRACT_DIR" 2>/dev/null || true)" ]]; then
  already_extracted=1
  log "[skip] already extracted; skipping download"
fi

if [[ "$already_extracted" -eq 0 ]]; then
  TOTAL_BYTES="$(remote_total_bytes || true)"
  if [[ -n "${TOTAL_BYTES}" ]]; then
    log "[info] remote_size_bytes=${TOTAL_BYTES}"
  else
    log "[warn] could not determine remote size; will rely on downloader completion"
  fi

  if is_pid_running; then
    log "[download] already running (pid=$(cat "$PID_FILE"))"
  else
    start_or_resume_download
  fi

  # Initialize speed baseline
  prev_size=0
  if [[ -f "$TARBALL" ]]; then
    prev_size="$(stat -c%s "$TARBALL" 2>/dev/null || echo 0)"
  fi
  prev_ts="$(date +%s)"

  while true; do
    size=0
    if [[ -f "$TARBALL" ]]; then
      size="$(stat -c%s "$TARBALL" 2>/dev/null || echo 0)"
    fi

    # If we know the expected size, consider complete once reached AND downloader exits.
    if [[ -n "${TOTAL_BYTES}" && "$TOTAL_BYTES" -gt 0 && "$size" -ge "$TOTAL_BYTES" ]]; then
      if is_pid_running; then
        log "[download] reached expected size; waiting for downloader to exit"
        while is_pid_running; do
          sleep 5
        done
      fi
      break
    fi

    if ! is_pid_running; then
      # Downloader ended; if we don't know total size, proceed as "done".
      if [[ -z "${TOTAL_BYTES}" && -f "$TARBALL" ]]; then
        log "[download] downloader exited; proceeding without size check"
        break
      fi
      # Otherwise, resume.
      log "[warn] downloader not running but file incomplete; resuming"
      start_or_resume_download
    fi

    now="$(date +%s)"
    dt=$((now - prev_ts))
    if [[ "$dt" -le 0 ]]; then dt=1; fi
    delta=$((size - prev_size))
    if [[ "$delta" -lt 0 ]]; then delta=0; fi
    bps=$((delta / dt))

    size_gib="$(awk -v s="$size" 'BEGIN{printf "%.2f", s/1024/1024/1024}')"
    speed_mib="$(awk -v b="$bps" 'BEGIN{printf "%.2f", b/1024/1024}')"

    if [[ -n "${TOTAL_BYTES}" && "$TOTAL_BYTES" -gt 0 ]]; then
      pct="$(awk -v s="$size" -v t="$TOTAL_BYTES" 'BEGIN{printf "%.2f", (s/t)*100}')"
      eta_min="$(awk -v s="$size" -v t="$TOTAL_BYTES" -v b="$bps" 'BEGIN{rem=t-s; if (b>0) printf "%.1f", rem/b/60; else print "?"}')"
      log "[download] ${size_gib}GiB (${pct}%) ${speed_mib}MiB/s eta=${eta_min}min"
    else
      log "[download] ${size_gib}GiB ${speed_mib}MiB/s"
    fi

    prev_size="$size"
    prev_ts="$now"
    sleep "$POLL_SECONDS"
  done

  log "[download] complete: ${TARBALL}"
fi

# Extract (avoid uploading the tarball/logs)
if [[ -f "$EXTRACT_OK" && -d "$EXTRACT_DIR" && -n "$(ls -A "$EXTRACT_DIR" 2>/dev/null || true)" ]]; then
  log "[extract] already extracted: ${EXTRACT_DIR}"
else
  rm -rf "$EXTRACT_DIR"
  mkdir -p "$EXTRACT_DIR"
  rm -f "$EXTRACT_OK"
  log "[extract] extracting -> ${EXTRACT_DIR}"
  tar -xzf "$TARBALL" -C "$EXTRACT_DIR"
  touch "$EXTRACT_OK"
  log "[extract] done"
fi

if [[ "$KEEP_TARBALL" != "1" && -f "$TARBALL" && -f "$EXTRACT_OK" ]]; then
  log "[cleanup] removing tarball to free space: ${TARBALL}"
  rm -f "$TARBALL"
fi

# Pick upload root: if a single top-level dir exists, upload that.
UPLOAD_DIR="$EXTRACT_DIR"
shopt -s nullglob
entries=("$EXTRACT_DIR"/*)
shopt -u nullglob
if [[ ${#entries[@]} -eq 1 && -d "${entries[0]}" ]]; then
  UPLOAD_DIR="${entries[0]}"
fi
log "[upload] local_dir=${UPLOAD_DIR}"

# Wait for HF auth
until hf_is_authed; do
  if [[ -n "$HF_TOKEN" ]]; then
    log "[wait] HF_TOKEN set but authentication failed (check token/permissions)"
  else
    log "[wait] Hugging Face not logged in. Run: hf auth login"
  fi
  sleep 30
done

log "[hf] ensuring repo exists + visibility: ${HF_REPO_ID} (private=${HF_PRIVATE})"
HF_REPO_ID="$HF_REPO_ID" HF_PRIVATE="$HF_PRIVATE" HF_TOKEN="$HF_TOKEN" python3 - <<'PY'
import os
import sys

from huggingface_hub import HfApi

repo_id = os.environ["HF_REPO_ID"]
private = os.environ.get("HF_PRIVATE", "0") == "1"
token = os.environ.get("HF_TOKEN") or True  # True => use locally cached token

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True, token=token)
try:
    api.update_repo_visibility(repo_id=repo_id, private=private, repo_type="model", token=token)
except Exception as e:
    print(f"[warn] could not update visibility: {e}", file=sys.stderr)
PY

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

log "[hf] uploading (resumable): ${UPLOAD_DIR} -> ${HF_REPO_ID}"
if [[ "$HF_PRIVATE" == "1" ]]; then
  hf upload-large-folder \
    "$HF_REPO_ID" \
    "$UPLOAD_DIR" \
    --repo-type model \
    --private \
    --num-workers "${HF_UPLOAD_WORKERS:-8}" \
    --no-bars
else
  hf upload-large-folder \
    "$HF_REPO_ID" \
    "$UPLOAD_DIR" \
    --repo-type model \
    --num-workers "${HF_UPLOAD_WORKERS:-8}" \
    --no-bars
fi

log "[done] upload complete: https://huggingface.co/${HF_REPO_ID}"
