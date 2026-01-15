#!/usr/bin/env bash
set -euo pipefail

# Run the GPT-OSS-20B DFlash pipeline on a Kaggle Jupyter server via Versa.
#
# Required:
#   export REMOTE_JUPYTER_URL="https://.../proxy"
# Optional:
#   export REMOTE_JUPYTER_TOKEN=""
#
# Optional overrides (forwarded to remote via env):
#   TARGET_MODEL, DATASET_REPO, TRAIN_FILES_CSV, SEQ_LEN, BLOCK_SIZE,
#   NUM_HIDDEN_LAYERS, MLP_RATIO, MAX_STEPS, SAVE_EVERY, LR,
#   TEACHER_ATTN_BACKEND, TEACHER_MEM_FRACTION
#
# Output:
#   harmony/cuda-norm/logs/versa_dflash_gptoss20b_pipeline.log

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"

if [[ -z "${REMOTE_JUPYTER_URL:-}" ]]; then
  echo "REMOTE_JUPYTER_URL is not set"
  exit 2
fi

mkdir -p "${ROOT_DIR}/logs"

SYNC_DIR="$(python "${ROOT_DIR}/scripts/versa_prepare_sync_min.py" --out "${ROOT_DIR}/.versa_sync_min")"

EXTRA_ENVS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  EXTRA_ENVS+=(--env "HF_TOKEN=${HF_TOKEN}")
fi

PYTHONPATH="${REPO_ROOT}/third_party/Versa" \
python -m versa run \
  --backend jupyter \
  --url "${REMOTE_JUPYTER_URL}" \
  ${REMOTE_JUPYTER_TOKEN:+--token "${REMOTE_JUPYTER_TOKEN}"} \
  --sync-local-dir "${SYNC_DIR}" \
  --sync-remote-dir "cuda-norm-sync" \
  --log-path "logs/dflash_gptoss20b_pipeline.log" \
  --env TARGET_MODEL="${TARGET_MODEL:-openai/gpt-oss-20b}" \
  --env DATASET_REPO="${DATASET_REPO:-radna0/harmony-qwen3-calib-packs-v2-20260113}" \
  --env TRAIN_FILES_CSV="${TRAIN_FILES_CSV:-packs/reasoning_style_10k_v2/reasoning_style_10k_v2.parquet,packs/tool_agentic_10k_v6/tool_agentic_10k_v6.parquet,packs/calib_prompt_10000_v2/calib_prompt_10000_v2.parquet}" \
  --env SEQ_LEN="${SEQ_LEN:-4096}" \
  --env BLOCK_SIZE="${BLOCK_SIZE:-8}" \
  --env NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-4}" \
  --env MLP_RATIO="${MLP_RATIO:-4.0}" \
  --env MAX_STEPS="${MAX_STEPS:-200}" \
  --env SAVE_EVERY="${SAVE_EVERY:-200}" \
  --env LR="${LR:-2e-4}" \
  --env TEACHER_ATTN_BACKEND="${TEACHER_ATTN_BACKEND:-fa3}" \
  --env TEACHER_MEM_FRACTION="${TEACHER_MEM_FRACTION:-0.75}" \
  "${EXTRA_ENVS[@]}" \
  bash /kaggle/working/cuda-norm-sync/scripts/kaggle_dflash_gptoss20b_pipeline.sh

echo "submitted. remote log: logs/dflash_gptoss20b_pipeline.log"
