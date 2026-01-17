#!/usr/bin/env bash
set -euo pipefail

# Run SGLang decode-only throughput benchmark on a remote Kaggle Jupyter server via Versa.
#
# Required:
#   export REMOTE_JUPYTER_URL="https://.../proxy"
#
# Optional:
#   export REMOTE_JUPYTER_TOKEN=""
#   export DECODE_REMOTE_LOG_DIR="logs"
#   export DECODE_ENV_FILE="/home/kojoe/harmony/cuda-norm/.env"
#   export ATTENTION_BACKEND="fa3"   # fa3|flashinfer|trtllm
#
# Example:
#   bash harmony/cuda-norm/scripts/versa_run_decode_bench_sglang_kaggle.sh \
#     --name base20b \
#     --model-path /kaggle/input/gpt-oss-20b/transformers/default/1 \
#     --prompt-len 256 --max-new-tokens 8192 --batch-sizes 1,2,4,8,16,32

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"
SYNC_DIR="${ROOT_DIR}/.versa_sync_min"

if [[ -z "${REMOTE_JUPYTER_URL:-}" ]]; then
  echo "[err] REMOTE_JUPYTER_URL is not set" >&2
  exit 2
fi

DECODE_REMOTE_LOG_DIR="${DECODE_REMOTE_LOG_DIR:-logs}"
DECODE_ENV_FILE="${DECODE_ENV_FILE:-${ROOT_DIR}/.env}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-fa3}"

NAME=""
MODEL_PATH=""
PROMPT_LEN="256"
MAX_NEW_TOKENS="2048"
BATCH_SIZES="1,2,4,8,16,32"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2;;
    --model-path) MODEL_PATH="$2"; shift 2;;
    --prompt-len) PROMPT_LEN="$2"; shift 2;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2;;
    --batch-sizes) BATCH_SIZES="$2"; shift 2;;
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

if [[ -z "${NAME}" ]]; then
  echo "[err] --name is required" >&2
  exit 2
fi
if [[ -z "${MODEL_PATH}" ]]; then
  echo "[err] --model-path is required" >&2
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
REMOTE_LOG="${DECODE_REMOTE_LOG_DIR}/decode_${NAME}_${MAX_NEW_TOKENS}_${TS}.log"
REMOTE_JSON="${DECODE_REMOTE_LOG_DIR}/decode_${NAME}_${MAX_NEW_TOKENS}_${TS}.json"
PORT="${PORT:-$((30000 + (RANDOM % 2000)))}"

PYTHONPATH="${REPO_ROOT}/third_party/Versa" \
python -m versa run \
  --backend jupyter \
  --url "${REMOTE_JUPYTER_URL}" \
  ${REMOTE_JUPYTER_TOKEN:+--token "${REMOTE_JUPYTER_TOKEN}"} \
  --cwd "/kaggle/working" \
  --sync-local-dir "${SYNC_DIR}" \
  --sync-remote-dir "cuda-norm" \
  --log-path "${REMOTE_LOG}" \
  --bootstrap-cmd "mkdir -p ${DECODE_REMOTE_LOG_DIR}" \
  --bootstrap-cmd "python -m pip install -q sglang[all] requests" \
  --env-file "${DECODE_ENV_FILE}" \
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  "python -u cuda-norm/scripts/kaggle_decode_bench_sglang.py --model-path ${MODEL_PATH} --attention-backend ${ATTENTION_BACKEND} --prompt-len ${PROMPT_LEN} --max-new-tokens ${MAX_NEW_TOKENS} --batch-sizes ${BATCH_SIZES} --port ${PORT} --out-json ${REMOTE_JSON}"

echo "[+] started decode bench"
echo "    name=${NAME}"
echo "    port=${PORT}"
echo "    remote_log=${REMOTE_LOG}"
echo "    remote_json=${REMOTE_JSON}"
