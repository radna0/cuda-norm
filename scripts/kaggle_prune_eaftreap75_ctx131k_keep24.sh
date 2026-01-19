#!/usr/bin/env bash
set -euo pipefail

# Run 20B EAFT-REAP pruning calibration at true 131072-token context and build
# the keep_n=24/32 structural prune (uniform across layers).
#
# Notes:
# - Uses Versa remote Jupyter (Kaggle) for GPU work.
# - Reads `KAGGLE_URL` and HF tokens from `harmony/cuda-norm/.env`.
# - Does NOT print the full KAGGLE_URL (it contains an auth token).
#
# Launch this script under nohup and monitor its log.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

set -a
source "${ROOT_DIR}/.env"
set +a

if [[ -z "${KAGGLE_URL:-}" ]]; then
  echo "[err] KAGGLE_URL missing in ${ROOT_DIR}/.env" >&2
  exit 2
fi

export REMOTE_JUPYTER_URL="${KAGGLE_URL}"

KERNEL_ID="$(bash "${ROOT_DIR}/scripts/kaggle_get_active_kernel_id.sh" 2>/dev/null || true)"
if [[ -z "${KERNEL_ID}" ]]; then
  echo "[err] could not infer kernel_id (is the Kaggle kernel running?)" >&2
  exit 2
fi

echo "[*] Starting ctx131k keep24 prune on Kaggle (URL redacted) kernel_id=${KERNEL_ID}" >&2

# Full-token scoring at max context (no 4096 cap).
# NOTE: This is extremely expensive; we lower chunk sizes to reduce peak VRAM.
export EAFT_ATTN_IMPL="${EAFT_ATTN_IMPL:-sdpa}"
export REAP_MAX_TOKENS_PER_BATCH="${REAP_MAX_TOKENS_PER_BATCH:-0}"
export EAFT_CHUNK_SIZE="${EAFT_CHUNK_SIZE:-256}"
export REAP_TOKEN_CHUNK_SIZE="${REAP_TOKEN_CHUNK_SIZE:-1024}"

bash "${ROOT_DIR}/scripts/versa_run_pruning_track_kaggle.sh" \
  --kernel-id "${KERNEL_ID}" \
  --task build_pruned_20b_eaftreap_keepfrac \
  --keep-fracs-csv "0.75" \
  --min-keep-per-layer 24 --max-keep-per-layer 24 \
  --num-rows 2000 --max-seq-length 131072 --batch-size 1 \
  --core-pos-top-m 4 --core-count-top-m 0
