#!/usr/bin/env bash
set -euo pipefail

# Run one EAFT single-model collector on a remote Kaggle Jupyter server via Versa.
#
# Required env:
#   REMOTE_JUPYTER_URL="https://.../proxy"
# Optional env:
#   REMOTE_JUPYTER_TOKEN=""
#   REMOTE_JUPYTER_KERNEL_ID="" # reuse an existing kernel
#   GPU_TYPE="H100:1"            # metadata only (Kaggle actual GPU is fixed)
#   EAFT_REMOTE_LOG_DIR="logs"
#   EAFT_ENV_FILE="/home/kojoe/harmony/cuda-norm/.env"
#
# Example:
#   export REMOTE_JUPYTER_URL="https://.../proxy"
#   bash harmony/cuda-norm/scripts/versa_run_eaft_single_kaggle.sh \
#     --model-id openai/gpt-oss-20b \
#     --model-path /kaggle/input/gpt-oss-20b/transformers/default/1 \
#     --seq-lens-csv 65536,131072 \
#     --num-blocks 8 --batch-size 1 --sample-points 10000

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"

if [[ -z "${REMOTE_JUPYTER_URL:-}" ]]; then
  echo "[err] REMOTE_JUPYTER_URL is not set" >&2
  exit 2
fi

EAFT_REMOTE_LOG_DIR="${EAFT_REMOTE_LOG_DIR:-logs}"
EAFT_ENV_FILE="${EAFT_ENV_FILE:-${ROOT_DIR}/.env}"
GPU_TYPE="${GPU_TYPE:-H100:1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
TRITON_VERSION="${TRITON_VERSION:-3.4.0}"
KERNELS_VERSION="${KERNELS_VERSION:-0.11.7}"

MODEL_ID=""
MODEL_PATH=""
SEQ_LENS_CSV="65536,131072"
NUM_BLOCKS="8"
BATCH_SIZE="1"
SAMPLE_POINTS="10000"
TOP_K="4"
ENTROPY_TOPK="20"
CC_QUANTILE="0.15"
PROGRESS_EVERY_S=""
MAX_NEW_TOKENS=""
SKIP_PREDOWNLOAD="0"
DETACH="1"
KERNEL_ID="${REMOTE_JUPYTER_KERNEL_ID:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel-id) KERNEL_ID="$2"; shift 2;;
    --model-id) MODEL_ID="$2"; shift 2;;
    --model-path) MODEL_PATH="$2"; shift 2;;
    --seq-lens-csv) SEQ_LENS_CSV="$2"; shift 2;;
    --num-blocks) NUM_BLOCKS="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --sample-points) SAMPLE_POINTS="$2"; shift 2;;
    --top-k) TOP_K="$2"; shift 2;;
    --entropy-topk) ENTROPY_TOPK="$2"; shift 2;;
    --cc-quantile) CC_QUANTILE="$2"; shift 2;;
    --progress-every-s) PROGRESS_EVERY_S="$2"; shift 2;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2;;
    --skip-predownload) SKIP_PREDOWNLOAD="1"; shift 1;;
    --no-detach) DETACH=""; shift 1;;
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

EXTRA_ARGS=()
if [[ "${SKIP_PREDOWNLOAD}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-predownload)
fi

EXTRA_ENV=()
if [[ -n "${PROGRESS_EVERY_S}" ]]; then
  EXTRA_ENV+=(--env "EAFT_PROGRESS_EVERY_S=${PROGRESS_EVERY_S}")
fi
if [[ -n "${MAX_NEW_TOKENS}" ]]; then
  EXTRA_ENV+=(--env "EAFT_MAX_NEW_TOKENS=${MAX_NEW_TOKENS}")
fi

# Do not leak KAGGLE_URL (auth token) into the remote environment or Versa logs.
# The remote kernel does not need KAGGLE_URL; it's only used locally for /files downloads.
ENV_TMP="$(mktemp -t eaft_env_XXXXXX)"
trap 'rm -f "${ENV_TMP}"' EXIT
if [[ -f "${EAFT_ENV_FILE}" ]]; then
  rg -v '^KAGGLE_URL=' "${EAFT_ENV_FILE}" > "${ENV_TMP}" || true
else
  : > "${ENV_TMP}"
fi

PYTHONPATH="${REPO_ROOT}/third_party/Versa" \
bash -lc "
  set -euo pipefail
  cd \"${ROOT_DIR}\"
  python -m versa run \
    --backend jupyter \
    --url \"${REMOTE_JUPYTER_URL}\" \
    ${REMOTE_JUPYTER_TOKEN:+--token \"${REMOTE_JUPYTER_TOKEN}\"} \
    ${KERNEL_ID:+--kernel-id \"${KERNEL_ID}\"} \
    --repo-root \"${ROOT_DIR}\" \
    --log-path \"${REMOTE_LOG}\" \
    ${DETACH:+--detach} \
    --bootstrap-cmd \"mkdir -p ${EAFT_REMOTE_LOG_DIR}\" \
    --bootstrap-cmd \"mkdir -p eaft_cache\" \
    --bootstrap-cmd \"mkdir -p artifacts\" \
    --bootstrap-cmd \"python -m pip install -U pip\" \
    --bootstrap-cmd \"python -m pip install -q modal datasets transformers==4.56.2 tokenizers safetensors pyarrow pandas accelerate huggingface-hub hf_transfer\" \
    --bootstrap-cmd \"python -m pip uninstall -y torchvision || true\" \
    --bootstrap-cmd \"python -m pip install -q triton==${TRITON_VERSION} kernels==${KERNELS_VERSION}\" \
    --bootstrap-cmd \"python -c \\\"import triton, kernels; ver=tuple(int(x) for x in triton.__version__.split('.')[:2]); assert ver >= (3,4), f'triton too old: {triton.__version__}'; print('[bootstrap] triton', triton.__version__, 'kernels OK')\\\"\" \
    --bootstrap-cmd \"python -m pip install -q 'sglang[all]'\" \
    --bootstrap-cmd \"python -c 'import torch; print(torch.__version__)' || python -m pip install -q torch --index-url ${TORCH_INDEX_URL}\" \
    --env-file \"${ENV_TMP}\" \
    --env \"GPU_TYPE=${GPU_TYPE}\" \
    --env \"EAFT_LOCAL_MODE=1\" \
    --env \"EAFT_ARTIFACTS_DIR=/kaggle/working/artifacts/eaft_models\" \
    --env \"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1\" \
    --env \"PYTHONFAULTHANDLER=1\" \
    --env \"TORCH_SHOW_CPP_STACKTRACES=1\" \
    --env \"TRANSFORMERS_NO_TORCHVISION=1\" \
    ${EXTRA_ENV[*]:+${EXTRA_ENV[*]}} \
    modal/collect_calib_packs_eaft_single.py::main -- \
      --model-id \"${MODEL_ID}\" \
      ${MODEL_PATH:+--model-path \"${MODEL_PATH}\"} \
      --seq-lens-csv \"${SEQ_LENS_CSV}\" \
      --num-blocks \"${NUM_BLOCKS}\" \
      --batch-size \"${BATCH_SIZE}\" \
      --sample-points \"${SAMPLE_POINTS}\" \
      --top-k \"${TOP_K}\" \
      --entropy-topk \"${ENTROPY_TOPK}\" \
      --cc-quantile \"${CC_QUANTILE}\" \
      ${EXTRA_ARGS[*]:+${EXTRA_ARGS[*]}}
"

echo "[+] started"
echo "    model_id=${MODEL_ID}"
echo "    model_path=${MODEL_PATH:-<hub>}"
echo "    remote_log=${REMOTE_LOG}"
