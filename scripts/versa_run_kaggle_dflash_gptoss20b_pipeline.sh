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
#   BENCHMARK_STEPS_CSV, BENCH_MAX_NEW_TOKENS, BENCH_CONCURRENCY, BENCH_NUM_PROMPTS
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

TS="$(date +%Y%m%d_%H%M%S)"
REMOTE_LOG_PATH="logs/dflash_gptoss20b_pipeline_${TS}.log"

PYTHONPATH="${REPO_ROOT}/third_party/Versa" \
python -m versa run \
  --backend jupyter \
  --url "${REMOTE_JUPYTER_URL}" \
  ${REMOTE_JUPYTER_TOKEN:+--token "${REMOTE_JUPYTER_TOKEN}"} \
  --sync-local-dir "${SYNC_DIR}" \
  --sync-remote-dir "cuda-norm-sync" \
  --log-path "${REMOTE_LOG_PATH}" \
  --detach \
  --env TARGET_MODEL="${TARGET_MODEL:-openai/gpt-oss-20b}" \
  --env DATASET_REPO="${DATASET_REPO:-radna0/harmony-qwen3-calib-packs-v2-20260113}" \
  --env TRAIN_FILES_CSV="${TRAIN_FILES_CSV:-packs/reasoning_style_10k_v2/reasoning_style_10k_v2.parquet,tool_agentic_10k_v6.parquet,packs/calib_prompt_10000_v2/calib_prompt_10000_v2.parquet}" \
  --env SEQ_LEN="${SEQ_LEN:-4096}" \
  --env BLOCK_SIZE="${BLOCK_SIZE:-8}" \
  --env NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-4}" \
  --env MLP_RATIO="${MLP_RATIO:-4.0}" \
  --env MAX_STEPS="${MAX_STEPS:-200}" \
  --env SAVE_EVERY="${SAVE_EVERY:-200}" \
  --env LR="${LR:-2e-4}" \
  --env TEACHER_ATTN_BACKEND="${TEACHER_ATTN_BACKEND:-fa3}" \
  --env TEACHER_MEM_FRACTION="${TEACHER_MEM_FRACTION:-0.75}" \
  ${BENCHMARK_STEPS_CSV:+--env BENCHMARK_STEPS_CSV="${BENCHMARK_STEPS_CSV}"} \
  ${BENCH_MAX_NEW_TOKENS:+--env BENCH_MAX_NEW_TOKENS="${BENCH_MAX_NEW_TOKENS}"} \
  ${BENCH_CONCURRENCY:+--env BENCH_CONCURRENCY="${BENCH_CONCURRENCY}"} \
  ${BENCH_NUM_PROMPTS:+--env BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS}"} \
  bash /kaggle/working/cuda-norm-sync/scripts/kaggle_dflash_gptoss20b_pipeline.sh

echo "submitted. remote log: ${REMOTE_LOG_PATH}"
