#!/usr/bin/env bash
set -euo pipefail

# Wrapper orchestrator for keep24 (0.75) bigblocks on Kaggle/VERSA.
#
# This script is intentionally a thin wrapper around:
#   harmony/cuda-norm/scripts/kaggle_orchestrate_keep24_bigblocks_curl.sh
#
# Why:
# - curl /files polling is more robust than Versa "tail" when the Kaggle proxy
#   URL is restarted.
# - still supports a convenience "start build" mode.
#
# Always run via nohup and monitor the produced local log.
#
# Required:
# - harmony/cuda-norm/.env must contain KAGGLE_URL (the Kaggle /proxy url)
#
# Usage (recommended):
#   bash harmony/cuda-norm/scripts/kaggle_keep24_pipeline.sh
#
# Usage (resume; if you already have ids):
#   bash harmony/cuda-norm/scripts/kaggle_orchestrate_keep24_bigblocks.sh \
#     --kernel-id <uuid> \
#     --build-remote-log logs/build_pruned_20b_eaftreap_keepfrac_<ts>.log
#
# Usage (start build + orchestrate):
#   bash harmony/cuda-norm/scripts/kaggle_orchestrate_keep24_bigblocks.sh --start-build

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

KERNEL_ID=""
BUILD_REMOTE_LOG=""
START_BUILD="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel-id) KERNEL_ID="$2"; shift 2;;
    --build-remote-log) BUILD_REMOTE_LOG="$2"; shift 2;;
    --start-build) START_BUILD="1"; shift 1;;
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

if [[ -z "${KERNEL_ID}" ]]; then
  KERNEL_ID="${REMOTE_JUPYTER_KERNEL_ID:-}"
fi

set -a
source "${ROOT_DIR}/.env"
set +a

if [[ -z "${KAGGLE_URL:-}" ]]; then
  echo "[err] KAGGLE_URL is not set (check ${ROOT_DIR}/.env)" >&2
  exit 2
fi

export REMOTE_JUPYTER_URL="${KAGGLE_URL}"

NUM_ROWS="${KEEP24_NUM_ROWS:-2000}"
MAX_SEQ_LENGTH="${KEEP24_MAX_SEQ_LENGTH:-4096}"
BATCH_SIZE="${KEEP24_BATCH_SIZE:-1}"

start_build() {
  bash "${ROOT_DIR}/scripts/versa_run_pruning_track_kaggle.sh" \
    ${KERNEL_ID:+--kernel-id "${KERNEL_ID}"} \
    --task build_pruned_20b_eaftreap_keepfrac \
    --model-id-20b openai/gpt-oss-20b \
    --num-rows "${NUM_ROWS}" --max-seq-length "${MAX_SEQ_LENGTH}" --batch-size "${BATCH_SIZE}" \
    --calib-packs-repo radna0/harmony-qwen3-calib-packs-v2-20260113 \
    --calib-pack-files "packs/reasoning_style_10k_v2/reasoning_style_10k_v2.parquet,tool_agentic_10k_v6.parquet,packs/calib_prompt_10000_v2/calib_prompt_10000_v2.parquet" \
    --calib-pack-sample-strategy per_file \
    --keep-fracs-csv 0.75 \
    --keep-n-round ceil --keep-n-multiple-of 4 \
    --min-keep-per-layer 24 --max-keep-per-layer 24 \
    --core-pos-top-m 4 --core-count-top-m 0 \
    || true
}

if [[ "${START_BUILD}" == "1" || -z "${BUILD_REMOTE_LOG}" ]]; then
  echo "[*] starting prune build (keep_frac=0.75, keep_n=24 uniform)..."
  tmp="$(mktemp -t keep24_orchestrate_build_launcher_XXXXXX.log)"
  echo "[*] build launcher output tee=${tmp}"
  start_build 2>&1 | tee "${tmp}"

  python - <<PY
import json
from pathlib import Path

txt = Path("${tmp}").read_text(encoding="utf-8", errors="ignore")
start = txt.find("{")
end = txt.rfind("}")
if start == -1 or end == -1 or end <= start:
    print("")
    print("")
    raise SystemExit(0)

try:
    obj = json.loads(txt[start : end + 1])
    details = obj.get("details") or {}
    print(details.get("kernel_id") or "")
    print(obj.get("log_path") or "")
except Exception:
    print("")
    print("")
PY
  | {
    read -r parsed_kernel || true
    read -r parsed_logpath || true
    if [[ -z "${KERNEL_ID}" ]]; then KERNEL_ID="${parsed_kernel}"; fi
    if [[ -z "${BUILD_REMOTE_LOG}" ]]; then BUILD_REMOTE_LOG="${parsed_logpath}"; fi
  }
fi

if [[ -z "${KERNEL_ID}" ]]; then
  echo "[err] kernel_id is unknown; pass --kernel-id or use --start-build" >&2
  exit 2
fi
if [[ -z "${BUILD_REMOTE_LOG}" ]]; then
  echo "[err] build remote log is unknown; pass --build-remote-log or use --start-build" >&2
  exit 2
fi

exec bash "${ROOT_DIR}/scripts/kaggle_orchestrate_keep24_bigblocks_curl.sh" \
  --kernel-id "${KERNEL_ID}" \
  --build-remote-log "${BUILD_REMOTE_LOG}"

