#!/usr/bin/env bash
set -euo pipefail

# Export an EasyDeL TPU DFLASH run dir (tensorstore) into an HF-style
# `DFlashDraftModel` checkpoint directory that `sglang-jax` can load.
#
# Usage:
#   RUN_DIR=/dev/shm/dflash-checkpoints/<run_name>/run-10000 \
#   DST=harmony/cuda-norm/artifacts/dflash_draft_ckpts/sglang_gptoss20b_run10000 \
#   bash harmony/cuda-norm/scripts/tpu_sglang_jax_export_dflash_ckpt.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

RUN_DIR="${RUN_DIR:-}"
DST="${DST:-}"
OUT_DTYPE="${OUT_DTYPE:-float16}" # float16 recommended; draft is small.
MASK_TOKEN="${MASK_TOKEN:-<|MASK|>}"

if [[ -z "${HF_TOKEN:-}" ]] && [[ -f "${REPO_ROOT}/harmony/cuda-norm/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${REPO_ROOT}/harmony/cuda-norm/.env"
  set +a
fi

if [[ -z "${RUN_DIR}" ]]; then
  echo "Missing RUN_DIR env var." >&2
  exit 2
fi
if [[ -z "${DST}" ]]; then
  echo "Missing DST env var." >&2
  exit 2
fi

mkdir -p "$(dirname -- "${DST}")"

python "${REPO_ROOT}/harmony/cuda-norm/scripts/convert_easydel_dflash_ckpt_to_sglang.py" \
  --run-dir "${RUN_DIR}" \
  --dst "${DST}" \
  --mask-token "${MASK_TOKEN}" \
  --dtype "${OUT_DTYPE}" \
  --keep-fc-bias \
  --force

echo "[+] Exported: ${DST}"
