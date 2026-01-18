#!/usr/bin/env bash
set -euo pipefail

# Run pruning-track tasks on a remote Kaggle Jupyter server via Versa.
#
# This runs the code *in the Kaggle kernel* (PRUNING_LOCAL_MODE=1), so it does
# not consume Modal GPU time and does not submit Modal jobs.
#
# Required:
#   export REMOTE_JUPYTER_URL="https://.../proxy"
#
# Optional:
#   export REMOTE_JUPYTER_TOKEN=""
#   export REMOTE_JUPYTER_KERNEL_ID="" # reuse an existing kernel
#   export PRUNING_ENV_FILE="/home/kojoe/harmony/cuda-norm/.env"
#   export PRUNING_REMOTE_LOG_DIR="logs"
#
# Examples:
#   # EAFT-REAP saliency (math pruning dataset)
#   bash harmony/cuda-norm/scripts/versa_run_pruning_track_kaggle.sh \
#     --task eaftreap_saliency_20b \
#     --model-id-20b openai/gpt-oss-20b \
#     --dataset-id radna0/nemotron-math-v2-harmony-tools --dataset-split high_part00 \
#     --num-rows 500 --max-seq-length 4096 --batch-size 1
#
#   # Build EAFT-REAP structural prunes (general 50% + math 25%)
#   bash harmony/cuda-norm/scripts/versa_run_pruning_track_kaggle.sh \
#     --task build_pruned_20b_eaftreap \
#     --model-id-20b openai/gpt-oss-20b \
#     --dataset-id radna0/harmony-nemotron-cpu-artifacts --dataset-split train \
#     --num-rows 500 --max-seq-length 4096 --batch-size 1
#
#   # Build EAFT-REAP keep_frac sweep on curated calib packs (recommended)
#   bash harmony/cuda-norm/scripts/versa_run_pruning_track_kaggle.sh \
#     --task build_pruned_20b_eaftreap_keepfrac \
#     --model-id-20b openai/gpt-oss-20b \
#     --calib-packs-repo radna0/harmony-qwen3-calib-packs-v2-20260113 \
#     --calib-pack-files "packs/reasoning_style_10k_v2/reasoning_style_10k_v2.parquet,tool_agentic_10k_v6.parquet,packs/calib_prompt_10000_v2/calib_prompt_10000_v2.parquet" \
#     --calib-pack-sample-strategy "per_file" \
#     --keep-fracs-csv "0.75,0.60" \
#     --keep-n-round "ceil" --keep-n-multiple-of 4 \
#     --num-rows 30000 --max-seq-length 4096 --batch-size 1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"
SYNC_DIR="${ROOT_DIR}/.versa_sync_min"

if [[ -z "${REMOTE_JUPYTER_URL:-}" ]]; then
  echo "[err] REMOTE_JUPYTER_URL is not set" >&2
  exit 2
fi

PRUNING_REMOTE_LOG_DIR="${PRUNING_REMOTE_LOG_DIR:-logs}"
PRUNING_ENV_FILE="${PRUNING_ENV_FILE:-${ROOT_DIR}/.env}"
DETACH="1"
KERNEL_ID="${REMOTE_JUPYTER_KERNEL_ID:-}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
TRITON_VERSION="${TRITON_VERSION:-3.4.0}"
KERNELS_VERSION="${KERNELS_VERSION:-0.11.7}"

TASK=""
MODEL_ID_20B="openai/gpt-oss-20b"
DATASET_ID="radna0/harmony-nemotron-cpu-artifacts"
DATASET_SPLIT="train"
TEXT_COLUMN="text"
DOMAIN=""
DOMAIN_COLUMN="meta_domain"
MATH_DATASET_ID="radna0/nemotron-math-v2-harmony-tools"
MATH_DATASET_SPLIT="high_part00"
MATH_TEXT_COLUMN="text"
NUM_ROWS="500"
MAX_SEQ_LENGTH="4096"
BATCH_SIZE="1"

# Calib packs (EAFT-REAP keep_frac sweep)
CALIB_PACKS_REPO="radna0/harmony-qwen3-calib-packs-v2-20260113"
CALIB_PACK_FILES="packs/reasoning_style_10k_v2/reasoning_style_10k_v2.parquet,tool_agentic_10k_v6.parquet,packs/calib_prompt_10000_v2/calib_prompt_10000_v2.parquet"
CALIB_PACK_SAMPLE_STRATEGY="per_file"
CALIB_PACK_WEIGHTS_CSV=""
KEEP_FRACS_CSV="0.75"
KEEP_FRAC="0.75"
KEEP_N_ROUND="ceil"
KEEP_N_MULTIPLE_OF="4"
MIN_KEEP_PER_LAYER="16"
MAX_KEEP_PER_LAYER="32"
CORE_POS_TOP_M="4"
CORE_COUNT_TOP_M="0"

# EAFT-REAP knobs
EAFT_CC_Q="0.15"
EAFT_UNCERTAIN_Q="0.85"
EAFT_ENTROPY_TOPK="20"
EAFT_W_GOOD="1.0"
EAFT_W_UNCERTAIN="0.25"
EAFT_W_CONFLICT="-2.0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2;;
    --model-id-20b) MODEL_ID_20B="$2"; shift 2;;
    --dataset-id) DATASET_ID="$2"; shift 2;;
    --dataset-split) DATASET_SPLIT="$2"; shift 2;;
    --text-column) TEXT_COLUMN="$2"; shift 2;;
    --domain) DOMAIN="$2"; shift 2;;
    --domain-column) DOMAIN_COLUMN="$2"; shift 2;;
    --math-dataset-id) MATH_DATASET_ID="$2"; shift 2;;
    --math-dataset-split) MATH_DATASET_SPLIT="$2"; shift 2;;
    --math-text-column) MATH_TEXT_COLUMN="$2"; shift 2;;
    --num-rows) NUM_ROWS="$2"; shift 2;;
    --max-seq-length) MAX_SEQ_LENGTH="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --calib-packs-repo) CALIB_PACKS_REPO="$2"; shift 2;;
    --calib-pack-files) CALIB_PACK_FILES="$2"; shift 2;;
    --calib-pack-sample-strategy) CALIB_PACK_SAMPLE_STRATEGY="$2"; shift 2;;
    --calib-pack-weights-csv) CALIB_PACK_WEIGHTS_CSV="$2"; shift 2;;
    --keep-fracs-csv) KEEP_FRACS_CSV="$2"; shift 2;;
    --keep-frac) KEEP_FRAC="$2"; shift 2;;
    --keep-n-round) KEEP_N_ROUND="$2"; shift 2;;
    --keep-n-multiple-of) KEEP_N_MULTIPLE_OF="$2"; shift 2;;
    --min-keep-per-layer) MIN_KEEP_PER_LAYER="$2"; shift 2;;
    --max-keep-per-layer) MAX_KEEP_PER_LAYER="$2"; shift 2;;
    --core-pos-top-m) CORE_POS_TOP_M="$2"; shift 2;;
    --core-count-top-m) CORE_COUNT_TOP_M="$2"; shift 2;;
    --eaft-cc-quantile) EAFT_CC_Q="$2"; shift 2;;
    --eaft-uncertain-quantile) EAFT_UNCERTAIN_Q="$2"; shift 2;;
    --eaft-entropy-topk) EAFT_ENTROPY_TOPK="$2"; shift 2;;
    --eaft-w-good) EAFT_W_GOOD="$2"; shift 2;;
    --eaft-w-uncertain) EAFT_W_UNCERTAIN="$2"; shift 2;;
    --eaft-w-conflict) EAFT_W_CONFLICT="$2"; shift 2;;
    --kernel-id) KERNEL_ID="$2"; shift 2;;
    --no-detach) DETACH=""; shift 1;;
    -h|--help)
      sed -n '1,140p' "$0"
      exit 0
      ;;
    *)
      echo "[err] Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${TASK}" ]]; then
  echo "[err] --task is required" >&2
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
REMOTE_LOG="${PRUNING_REMOTE_LOG_DIR}/${TASK}_${TS}.log"

python "${ROOT_DIR}/scripts/versa_prepare_sync_min.py" --out "${SYNC_DIR}" >/dev/null

PYTHONPATH="${REPO_ROOT}/third_party/Versa" \
python -m versa run \
  --backend jupyter \
  --url "${REMOTE_JUPYTER_URL}" \
  ${REMOTE_JUPYTER_TOKEN:+--token "${REMOTE_JUPYTER_TOKEN}"} \
  ${KERNEL_ID:+--kernel-id "${KERNEL_ID}"} \
  --cwd "/kaggle/working" \
  --sync-local-dir "${SYNC_DIR}" \
  --sync-remote-dir "harmony/cuda-norm" \
  --log-path "${REMOTE_LOG}" \
  ${DETACH:+--detach} \
  --bootstrap-cmd "mkdir -p /kaggle/working/${PRUNING_REMOTE_LOG_DIR}" \
  --bootstrap-cmd "python -m pip install -U pip" \
  --bootstrap-cmd "python -m pip install -q modal datasets transformers==4.56.2 tokenizers safetensors pyarrow pandas accelerate huggingface-hub hf_transfer" \
  --bootstrap-cmd "python -m pip uninstall -y torchvision || true" \
  --bootstrap-cmd "python -m pip install -q triton==${TRITON_VERSION} kernels==${KERNELS_VERSION}" \
  --bootstrap-cmd "python -c \"import triton, kernels; ver=tuple(int(x) for x in triton.__version__.split('.')[:2]); assert ver >= (3,4), f'triton too old: {triton.__version__}'; print('[bootstrap] triton', triton.__version__, 'kernels OK')\"" \
  --bootstrap-cmd "python -c 'import torch; print(torch.__version__)' || python -m pip install -q torch --index-url ${TORCH_INDEX_URL}" \
  --env-file "${PRUNING_ENV_FILE}" \
  --env "PYTHONFAULTHANDLER=1" \
  --env "TORCH_SHOW_CPP_STACKTRACES=1" \
  --env "TRANSFORMERS_NO_TORCHVISION=1" \
  --env "PRUNING_LOCAL_MODE=1" \
  --env "PRUNING_CACHE_ROOT=/kaggle/working/pruning_cache" \
  --env "PRUNING_MODEL_DIR=/kaggle/working/pruning_cache/model" \
  --env "PRUNING_DATA_DIR=/kaggle/working/pruning_cache/data" \
  --env "PRUNING_HF_HOME=/kaggle/working/pruning_cache/hf_cache" \
  --env "PRUNING_ARTIFACTS_DIR=/kaggle/working/artifacts/harmony_cuda_norm" \
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  --env "MODEL_DIR_20B=/kaggle/input/gpt-oss-20b/transformers/default/1" \
  "${ROOT_DIR}/modal/gpt_oss_pruning_track.py::main" -- \
    --task "${TASK}" \
    --model-id-20b "${MODEL_ID_20B}" \
    --dataset-id "${DATASET_ID}" --dataset-split "${DATASET_SPLIT}" --text-column "${TEXT_COLUMN}" \
    --domain "${DOMAIN}" --domain-column "${DOMAIN_COLUMN}" \
    --math-dataset-id "${MATH_DATASET_ID}" --math-dataset-split "${MATH_DATASET_SPLIT}" --math-text-column "${MATH_TEXT_COLUMN}" \
    --num-rows "${NUM_ROWS}" --max-seq-length "${MAX_SEQ_LENGTH}" --batch-size "${BATCH_SIZE}" \
    --eaft-cc-quantile "${EAFT_CC_Q}" --eaft-uncertain-quantile "${EAFT_UNCERTAIN_Q}" --eaft-entropy-topk "${EAFT_ENTROPY_TOPK}" \
    --eaft-w-good "${EAFT_W_GOOD}" --eaft-w-uncertain "${EAFT_W_UNCERTAIN}" --eaft-w-conflict "${EAFT_W_CONFLICT}" \
    --calib-packs-repo "${CALIB_PACKS_REPO}" --calib-pack-files-csv "${CALIB_PACK_FILES}" --calib-pack-sample-strategy "${CALIB_PACK_SAMPLE_STRATEGY}" \
    --calib-pack-weights-csv "${CALIB_PACK_WEIGHTS_CSV}" \
    --keep-fracs-csv "${KEEP_FRACS_CSV}" \
    --keep-n-round "${KEEP_N_ROUND}" --keep-n-multiple-of "${KEEP_N_MULTIPLE_OF}" \
    --keep-frac "${KEEP_FRAC}" --min-keep-per-layer "${MIN_KEEP_PER_LAYER}" --max-keep-per-layer "${MAX_KEEP_PER_LAYER}" \
    --core-pos-top-m "${CORE_POS_TOP_M}" --core-count-top-m "${CORE_COUNT_TOP_M}"

echo "[+] started pruning task"
echo "    task=${TASK}"
echo "    remote_log=${REMOTE_LOG}"
