#!/usr/bin/env bash
set -euo pipefail

# Kaggle/VERSA autopipe:
#   1) wait for KAGGLE_URL to be reachable
#   2) rebuild keep24 (keep_frac=0.75) EAFT-REAP checkpoint (no finetune)
#   3) run the long-seq token-budget matrix (4096/8192/16384)
#
# Always run under nohup and monitor the local log:
#   bash harmony/cuda-norm/scripts/nohup_logged.sh --name kaggle_autopipe_keep24_longseq --log-dir harmony/cuda-norm/unsloth_logs -- \
#     bash harmony/cuda-norm/scripts/kaggle_autopipe_keep24_longseq.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

set -a
source "${ROOT_DIR}/.env"
set +a

if [[ -z "${KAGGLE_URL:-}" ]]; then
  echo "[err] KAGGLE_URL missing in ${ROOT_DIR}/.env" >&2
  exit 2
fi

export REMOTE_JUPYTER_URL="${KAGGLE_URL}"

echo "[*] waiting for Kaggle to be reachable..."
bash "${ROOT_DIR}/scripts/kaggle_wait_ready.sh" --interval-s 120

echo "[*] rebuilding keep24 (keep_frac=0.75, keep_n=24/32, top_k=4 unchanged)..."
bash "${ROOT_DIR}/scripts/kaggle_build_keep24_only.sh"

echo "[*] running long-seq token-budget matrix (base vs keep24)..."
KERNEL_ID="$(bash "${ROOT_DIR}/scripts/kaggle_get_active_kernel_id.sh")"
echo "[*] kernel_id=${KERNEL_ID}"
bash "${ROOT_DIR}/scripts/kaggle_eaftreap75_longseq_tokenbudget1m_keep24_uniform.sh" --kernel-id "${KERNEL_ID}"

echo "[+] autopipe finished"
