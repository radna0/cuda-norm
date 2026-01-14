#!/usr/bin/env bash
set -euo pipefail

# DP embedding runner for the 300k deep-reasoning excerpt pool.
#
# Requirements:
# - `modal` CLI configured (uses current MODAL_PROFILE).
# - HF_TOKEN available via `harmony/cuda-norm/.env` (not printed).
#
# This launches one Modal job per shard (tp_size=1), writing logs to:
#   harmony/cuda-norm/modal_parallel_logs/
#
# Usage:
#   ./run_modal_excerpt300k_embed_dp8.sh <shard_from> <shard_to>
#
# Example:
#   ./run_modal_excerpt300k_embed_dp8.sh 0 5
#   ./run_modal_excerpt300k_embed_dp8.sh 6 7

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <shard_from> <shard_to>" >&2
  exit 2
fi

SHARD_FROM="$1"
SHARD_TO="$2"

mkdir -p harmony/cuda-norm/modal_parallel_logs

LOG_TS_FILE="harmony/cuda-norm/modal_parallel_logs/LATEST_EXCERPT300K_LOG_TS.txt"
RUN_GROUP_FILE="harmony/cuda-norm/modal_parallel_logs/LATEST_EXCERPT300K_RUN_GROUP.txt"

# Important: for multi-wave runs (0-5 then 6-7), we must keep a stable
# `log_ts` so the watcher can locate logs consistently.
#
# Resolution order:
# 1) LOG_TS_OVERRIDE env var
# 2) existing LATEST_EXCERPT300K_LOG_TS.txt (if valid)
# 3) current wall-clock time
TS="${LOG_TS_OVERRIDE:-}"
if [[ -z "${TS}" && -f "${LOG_TS_FILE}" ]]; then
  TS="$(tr -d ' \t\r\n' <"${LOG_TS_FILE}" || true)"
fi
if [[ -z "${TS}" || ! "${TS}" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
fi

RUN_GROUP="${RUN_GROUP:-excerpt300k_${TS}}"

# Persist the run group + log_ts for watcher/automation.
echo "${RUN_GROUP}" > "${RUN_GROUP_FILE}"
echo "${TS}" > "${LOG_TS_FILE}"
CANDIDATE_DATASET_ID_DEFAULT="radna0/harmony-qwen3-reasoning-excerpt-candidates-v2-300k"
CANDIDATE_DATASET_ID="${CANDIDATE_DATASET_ID_OVERRIDE:-$CANDIDATE_DATASET_ID_DEFAULT}"
OUT_DATASET_ID_DEFAULT="radna0/harmony-qwen3-reasoning-excerpt-embeddings-v2-300k"
OUT_DATASET_ID="${OUT_DATASET_ID_OVERRIDE:-$OUT_DATASET_ID_DEFAULT}"

for ((i=SHARD_FROM; i<=SHARD_TO; i++)); do
  LOG="harmony/cuda-norm/modal_parallel_logs/excerpt300k_embed_shard${i}_${TS}.log"
  PIDFILE="${LOG}.pid"
  nohup bash -lc '
    set -euo pipefail
    set -a
    source harmony/cuda-norm/.env
    set +a
    export CANDIDATE_DATASET_ID='"$CANDIDATE_DATASET_ID"'
    export CANDIDATE_SUBDIR=
    export CANDIDATE_MOUNT_DIR=/root/data/__disabled
    export MODEL_ID=Qwen/Qwen3-Embedding-8B
    export TRUST_REMOTE_CODE=1
    export OUT_DATASET_ID='"$OUT_DATASET_ID"'
    export OUT_SUBDIR=embeddings/'"$RUN_GROUP"'/shard_'"$i"'

    export TEXT_COLUMN=embed_text
    export OUT_DIM=256
    export MAX_TOKENS=2048
    export BATCH_SIZE=256
    export MAX_RECORDS=0
    export ROWS_PER_SHARD=200000
    export LOG_EVERY_S=10

    export USE_INPUT_IDS_COLUMN=1
    export INPUT_IDS_COLUMN=input_ids
    export LOCAL_FILES_ONLY=1
    export SKIP_PREFETCH=1
    export PREFETCH_ONLY=0

    export SGLANG_ATTENTION_BACKEND=trtllm_mha
    export SGLANG_KV_CACHE_DTYPE=bf16
    export SGLANG_DISABLE_CUDA_GRAPH=0
    export SGLANG_TP_SIZE=1

    export FILE_SHARD_COUNT=8
    export FILE_SHARD_INDEX='"$i"'

    modal run harmony/cuda-norm/modal/qwen_embedding_sglang_scoring.py
  ' >"$LOG" 2>&1 &
  echo $! >"$PIDFILE"
  echo "[ok] shard=$i pid=$(cat "$PIDFILE") log=$LOG"
done
