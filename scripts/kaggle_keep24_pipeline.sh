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
KERNEL_STATE_FILE="${ROOT_DIR}/.kaggle_kernel_id"

set -a
source "${ROOT_DIR}/.env"
set +a

if [[ -z "${KAGGLE_URL:-}" ]]; then
  echo "[err] KAGGLE_URL missing in ${ROOT_DIR}/.env" >&2
  exit 2
fi

export REMOTE_JUPYTER_URL="${KAGGLE_URL}"

# Resolve an active kernel-id if not provided, or if the stored one is stale.
resolve_kernel_id() {
  local kid="${REMOTE_JUPYTER_KERNEL_ID:-}"
  if [[ -n "${kid}" ]]; then
    echo "${kid}"
    return 0
  fi
  if [[ -f "${KERNEL_STATE_FILE}" ]]; then
    kid="$(tr -d '\n\r ' < "${KERNEL_STATE_FILE}" || true)"
  fi
  if [[ -n "${kid}" ]]; then
    # Validate the kernel exists for this KAGGLE_URL.
    local base="${KAGGLE_URL%/}"
    local code
    code="$(curl -s -o /dev/null -w \"%{http_code}\" \"${base}/api/kernels/${kid}\" || true)"
    if [[ "${code}" == "200" ]]; then
      echo "${kid}"
      return 0
    fi
    echo "[warn] stale kernel-id '${kid}' (api/kernels/${kid} -> http ${code}); auto-detecting" >&2
  fi
  bash "${ROOT_DIR}/scripts/kaggle_get_active_kernel_id.sh"
}

# If we have a previously used kernel-id for this server, reuse it to avoid
# leaking VRAM to zombie processes across multiple kernels.
REMOTE_JUPYTER_KERNEL_ID="$(resolve_kernel_id)"
export REMOTE_JUPYTER_KERNEL_ID

NUM_ROWS="${KEEP24_NUM_ROWS:-2000}"
MAX_SEQ_LENGTH="${KEEP24_MAX_SEQ_LENGTH:-4096}"
BATCH_SIZE="${KEEP24_BATCH_SIZE:-1}"

PREFLIGHT_CLEANUP="${KEEP24_PREFLIGHT_CLEANUP:-1}"
if [[ -n "${REMOTE_JUPYTER_KERNEL_ID:-}" && "${PREFLIGHT_CLEANUP}" != "0" ]]; then
  echo "[*] preflight: Kaggle GPU cleanup on kernel=${REMOTE_JUPYTER_KERNEL_ID}"
  # Best-effort: kill leaked Versa modal_run processes holding VRAM from previous
  # runs, then print nvidia-smi. Hard timeout so we never hang.
  PYTHONPATH="third_party/Versa${PYTHONPATH:+:${PYTHONPATH}}" \
  timeout 180s python "${ROOT_DIR}/scripts/kaggle_gpu_cleanup.py" \
    --kernel-id "${REMOTE_JUPYTER_KERNEL_ID}" \
    --aggressive || echo "[warn] preflight cleanup failed/timeout; continuing"
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
  ${REMOTE_JUPYTER_KERNEL_ID:+--kernel-id "${REMOTE_JUPYTER_KERNEL_ID}"} \
  2>&1 | tee "${tmp}"
rc="${PIPESTATUS[0]}"
set -e

echo "[*] build launcher exitcode=${rc}"

OUT="$(cat "${tmp}" || true)"

KERNEL_ID="$(
  python - <<PY
import re
from pathlib import Path

txt = Path("${tmp}").read_text(encoding="utf-8", errors="ignore")

m = re.search(r'\"kernel_id\"\\s*:\\s*\"([0-9a-f\\-]+)\"', txt, flags=re.I)
if m:
    print(m.group(1))
    raise SystemExit(0)
m = re.search(r'\\[\\+\\]\\s*kernel_id=([0-9a-f\\-]+)\\s*$', txt, flags=re.I | re.M)
if m:
    print(m.group(1))
    raise SystemExit(0)
print("")
PY
)"
BUILD_REMOTE_LOG="$(
  python - <<PY
import re
from pathlib import Path

txt = Path("${tmp}").read_text(encoding="utf-8", errors="ignore")

try:
    m = re.search(r'\"log_path\"\\s*:\\s*\"([^\"]+)\"', txt)
    if m:
        print(m.group(1))
    else:
        print("")
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

# Persist the kernel-id so the next run can reuse it for cleanup + stability.
printf "%s" "${KERNEL_ID}" > "${KERNEL_STATE_FILE}"

exec bash "${ROOT_DIR}/scripts/kaggle_orchestrate_keep24_bigblocks_curl.sh" \
  --kernel-id "${KERNEL_ID}" \
  --build-remote-log "${BUILD_REMOTE_LOG}"
