#!/usr/bin/env bash
set -euo pipefail

# Keep24 (0.75) end-to-end pipeline on Kaggle:
# 1) Launch the prune build via Versa (returns immediately)
# 2) Extract kernel_id + build_remote_log from the launcher output
# 3) Run the curl-based orchestrator to completion (build -> EAFT base/pruned -> parity md)
#
# Always run this script under nohup (it can take hours).
#
# Usage:
#   bash harmony/cuda-norm/scripts/kaggle_keep24_pipeline.sh
#
# Optional:
#   export KEEP24_NUM_ROWS=2000
#   export KEEP24_MAX_SEQ_LENGTH=4096
#   export KEEP24_BATCH_SIZE=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

set -a
source "${ROOT_DIR}/.env"
set +a

if [[ -z "${KAGGLE_URL:-}" ]]; then
  echo "[err] KAGGLE_URL missing in ${ROOT_DIR}/.env" >&2
  exit 2
fi

export REMOTE_JUPYTER_URL="${KAGGLE_URL}"

NUM_ROWS="${KEEP24_NUM_ROWS:-2000}"
MAX_SEQ_LENGTH="${KEEP24_MAX_SEQ_LENGTH:-4096}"
BATCH_SIZE="${KEEP24_BATCH_SIZE:-1}"

if [[ -n "${REMOTE_JUPYTER_KERNEL_ID:-}" ]]; then
  PREFLIGHT_CLEANUP="${KEEP24_PREFLIGHT_CLEANUP:-1}"
  if [[ "${PREFLIGHT_CLEANUP}" != "0" ]]; then
    echo "[*] preflight: Kaggle GPU cleanup on existing kernel=${REMOTE_JUPYTER_KERNEL_ID}"
    # Best-effort: kill leaked Versa modal_run processes holding VRAM from
    # previous runs, then print nvidia-smi. Hard timeout so we never hang.
    PYTHONPATH="third_party/Versa${PYTHONPATH:+:${PYTHONPATH}}" \
    timeout 180s python "${ROOT_DIR}/scripts/kaggle_gpu_cleanup.py" \
      --kernel-id "${REMOTE_JUPYTER_KERNEL_ID}" \
      --aggressive || echo "[warn] preflight cleanup failed/timeout; continuing"
  else
    echo "[*] preflight: cleanup disabled (KEEP24_PREFLIGHT_CLEANUP=0)"
  fi
fi

echo "[*] launching keep24 build on Kaggle..."
tmp="$(mktemp -t keep24_build_launcher_XXXXXX.log)"
echo "[*] build launcher output tee=${tmp}"

set +e
bash "${ROOT_DIR}/scripts/versa_run_pruning_track_kaggle.sh" \
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
  2>&1 | tee "${tmp}"
rc="${PIPESTATUS[0]}"
set -e

echo "[*] build launcher exitcode=${rc}"

OUT="$(cat "${tmp}" || true)"

KERNEL_ID="$(
  python - <<PY
import json
from pathlib import Path

txt = Path("${tmp}").read_text(encoding="utf-8", errors="ignore")

try:
    start = txt.find("{")
    end = txt.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no json block found")
    obj = json.loads(txt[start : end + 1])
    details = obj.get("details") or {}
    print(details.get("kernel_id") or "")
except Exception:
    print("")
PY
)"
BUILD_REMOTE_LOG="$(
  python - <<PY
import json
from pathlib import Path

txt = Path("${tmp}").read_text(encoding="utf-8", errors="ignore")

try:
    start = txt.find("{")
    end = txt.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no json block found")
    obj = json.loads(txt[start : end + 1])
    print(obj.get("log_path") or "")
except Exception:
    print("")
PY
)"

if [[ -z "${KERNEL_ID}" || -z "${BUILD_REMOTE_LOG}" ]]; then
  echo "[err] failed to parse kernel_id or remote_log from build launcher output" >&2
  echo "[err] kernel_id='${KERNEL_ID}' remote_log='${BUILD_REMOTE_LOG}'" >&2
  exit 2
fi

echo "[+] kernel_id=${KERNEL_ID}"
echo "[+] build_remote_log=${BUILD_REMOTE_LOG}"

exec bash "${ROOT_DIR}/scripts/kaggle_orchestrate_keep24_bigblocks_curl.sh" \
  --kernel-id "${KERNEL_ID}" \
  --build-remote-log "${BUILD_REMOTE_LOG}"
