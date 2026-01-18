#!/usr/bin/env bash
set -euo pipefail

# Build the keep_n=24/32 (keep_frac=0.75) EAFT-REAP pruned checkpoint on Kaggle.
#
# Why this exists:
# - Kaggle kernels are ephemeral: /kaggle/working is wiped on restart.
# - Long-seq EAFT eval requires the pruned weights to exist at the manifest path:
#     /kaggle/working/artifacts/harmony_cuda_norm/20b_pruned_models_eaftreap/...
# - This script rebuilds the pruned weights (no finetune), then exits.
#
# Always run under nohup and monitor the local log:
#   mkdir -p harmony/cuda-norm/unsloth_logs
#   nohup bash harmony/cuda-norm/scripts/kaggle_build_keep24_only.sh \
#     > harmony/cuda-norm/unsloth_logs/kaggle_build_keep24_only.log 2>&1 &
#
# Optional env:
#   KEEP24_NUM_ROWS=2000
#   KEEP24_MAX_SEQ_LENGTH=4096
#   KEEP24_BATCH_SIZE=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

set -a
source "${ROOT_DIR}/.env"
set +a

if [[ -z "${KAGGLE_URL:-}" ]]; then
  echo "[err] KAGGLE_URL missing in ${ROOT_DIR}/.env" >&2
  exit 2
fi

export REMOTE_JUPYTER_URL="${KAGGLE_URL}"

KERNEL_ID="$(bash "${ROOT_DIR}/scripts/kaggle_get_active_kernel_id.sh")"
echo "[*] kernel_id=${KERNEL_ID}"

NUM_ROWS="${KEEP24_NUM_ROWS:-2000}"
MAX_SEQ_LENGTH="${KEEP24_MAX_SEQ_LENGTH:-4096}"
BATCH_SIZE="${KEEP24_BATCH_SIZE:-1}"

echo "[*] preflight: Kaggle GPU cleanup"
PYTHONPATH="${ROOT_DIR}/../third_party/Versa${PYTHONPATH:+:${PYTHONPATH}}" \
timeout 180s python "${ROOT_DIR}/scripts/kaggle_gpu_cleanup.py" \
  --kernel-id "${KERNEL_ID}" \
  --env-file "${ROOT_DIR}/.env" \
  --aggressive || echo "[warn] preflight cleanup failed/timeout; continuing"

echo "[*] launching keep24 build (EAFT-REAP keepfrac sweep constrained to keep_n=24)"

LAUNCH_OUT="$(mktemp -t build_keep24_launcher_XXXXXX.log)"
echo "[*] launcher tee=${LAUNCH_OUT}"

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
  --kernel-id "${KERNEL_ID}" \
  2>&1 | tee "${LAUNCH_OUT}"
rc="${PIPESTATUS[0]}"
set -e

echo "[*] launcher exitcode=${rc}"

BUILD_REMOTE_LOG="$(
  python - <<PY
import re
from pathlib import Path
txt = Path("${LAUNCH_OUT}").read_text(encoding="utf-8", errors="ignore")
m = re.search(r'\"log_path\"\\s*:\\s*\"([^\"]+)\"', txt)
print(m.group(1) if m else "")
PY
)"

if [[ -z "${BUILD_REMOTE_LOG}" ]]; then
  echo "[err] could not parse build remote log path from launcher output" >&2
  exit 2
fi
echo "[+] build_remote_log=${BUILD_REMOTE_LOG}"

echo "[*] waiting for build to finish..."
bash "${ROOT_DIR}/scripts/kaggle_watch_file.sh" \
  --remote-path "${BUILD_REMOTE_LOG}" \
  --pattern "\\[\\+\\] Wrote reports/20b_structural_prune_build_eaftreap_keepfrac\\.md" \
  --fail-pattern "Fatal Python error: Segmentation fault|Segmentation fault \\(core dumped\\)|torch\\.OutOfMemoryError|CUDA out of memory|OutOfMemoryError|Traceback \\(most recent call last\\)" \
  --interval-s 120 \
  --tail-n 160

echo "[+] keep24 build finished"
