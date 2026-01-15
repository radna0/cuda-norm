#!/usr/bin/env bash
set -euo pipefail

# Run one EAFT single-model collector on a remote Kaggle Jupyter server via Versa.
#
# Required env:
#   REMOTE_JUPYTER_URL="https://.../proxy"
# Optional env:
#   REMOTE_JUPYTER_TOKEN=""
#   GPU_TYPE="H100:1"            # metadata only (Kaggle actual GPU is fixed)
#   EAFT_REMOTE_LOG_DIR="/kaggle/working/logs"
#   EAFT_ENV_FILE="/home/kojoe/harmony/cuda-norm/.env"
#
# Example:
#   export REMOTE_JUPYTER_URL="https://.../proxy"
#   bash harmony/cuda-norm/scripts/versa_run_eaft_single_kaggle.sh \
#     --model-id openai/gpt-oss-20b \
#     --seq-lens-csv 65536,131072 \
#     --num-blocks 8 --batch-size 1 --sample-points 10000

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"

if [[ -z "${REMOTE_JUPYTER_URL:-}" ]]; then
  echo "[err] REMOTE_JUPYTER_URL is not set" >&2
  exit 2
fi

EAFT_REMOTE_LOG_DIR="${EAFT_REMOTE_LOG_DIR:-/kaggle/working/logs}"
EAFT_ENV_FILE="${EAFT_ENV_FILE:-${ROOT_DIR}/.env}"
GPU_TYPE="${GPU_TYPE:-H100:1}"

MODEL_ID=""
SEQ_LENS_CSV="65536,131072"
NUM_BLOCKS="8"
BATCH_SIZE="1"
SAMPLE_POINTS="10000"
TOP_K="4"
ENTROPY_TOPK="20"
CC_QUANTILE="0.15"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-id) MODEL_ID="$2"; shift 2;;
    --seq-lens-csv) SEQ_LENS_CSV="$2"; shift 2;;
    --num-blocks) NUM_BLOCKS="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --sample-points) SAMPLE_POINTS="$2"; shift 2;;
    --top-k) TOP_K="$2"; shift 2;;
    --entropy-topk) ENTROPY_TOPK="$2"; shift 2;;
    --cc-quantile) CC_QUANTILE="$2"; shift 2;;
    -h|--help)
      sed -n '1,120p' "$0"
      exit 0
      ;;
    *)
      echo "[err] Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${MODEL_ID}" ]]; then
  echo "[err] --model-id is required" >&2
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
SLUG="$(echo "${MODEL_ID}" | tr '/:' '__')"
REMOTE_LOG="${EAFT_REMOTE_LOG_DIR}/eaft_single_${SLUG}_${TS}.log"

PYTHONPATH="${REPO_ROOT}/third_party/Versa" \
python -m versa run \
  --backend jupyter \
  --url "${REMOTE_JUPYTER_URL}" \
  ${REMOTE_JUPYTER_TOKEN:+--token "${REMOTE_JUPYTER_TOKEN}"} \
  --cwd "/kaggle/working" \
  --log-path "${REMOTE_LOG}" \
  --detach \
  --bootstrap-cmd "mkdir -p ${EAFT_REMOTE_LOG_DIR}" \
  --env-file "${EAFT_ENV_FILE}" \
  --env "GPU_TYPE=${GPU_TYPE}" \
  --env "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1" \
  "${ROOT_DIR}/modal/collect_calib_packs_eaft_single.py::main" -- \
    --model-id "${MODEL_ID}" \
    --seq-lens-csv "${SEQ_LENS_CSV}" \
    --num-blocks "${NUM_BLOCKS}" \
    --batch-size "${BATCH_SIZE}" \
    --sample-points "${SAMPLE_POINTS}" \
    --top-k "${TOP_K}" \
    --entropy-topk "${ENTROPY_TOPK}" \
    --cc-quantile "${CC_QUANTILE}"

echo "[+] started"
echo "    model_id=${MODEL_ID}"
echo "    remote_log=${REMOTE_LOG}"
