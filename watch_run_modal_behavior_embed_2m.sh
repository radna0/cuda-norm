#!/usr/bin/env bash
set -euo pipefail

PACKED_DIR="${1:?usage: $0 <packed_candidates_dir> <remote_tag>}"
REMOTE_TAG="${2:?usage: $0 <packed_candidates_dir> <remote_tag>}"
PACKED_BASE="$(basename "$PACKED_DIR")"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/modal_parallel_logs}"
CPU_LOG_DIR="${CPU_LOG_DIR:-$ROOT_DIR/cpu_logs}"
mkdir -p "$LOG_DIR" "$CPU_LOG_DIR"

# Concurrency cap (manager requirement).
# Default to 6 (manager requirement) to keep GPU utilization high without overloading the host.
MAX_PARALLEL="${MAX_PARALLEL:-6}"
if ! [[ "${MAX_PARALLEL}" =~ ^[0-9]+$ ]]; then
  echo "[err] MAX_PARALLEL must be an integer, got: ${MAX_PARALLEL}" >&2
  exit 2
fi
if [[ "${MAX_PARALLEL}" -gt 6 ]]; then
  MAX_PARALLEL=6
fi
FILE_SHARD_COUNT="${FILE_SHARD_COUNT:-8}"

# Speed: avoid re-uploading candidates / re-prefetching model if already cached in Modal volumes.
SKIP_CANDIDATE_UPLOAD="${SKIP_CANDIDATE_UPLOAD:-1}"
SKIP_MODEL_PREFETCH="${SKIP_MODEL_PREFETCH:-1}"

# Embedding config (behavior view).
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Embedding-8B}"
MODEL_DIR_NAME="${MODEL_ID//\//__}"
OUT_DIM="${OUT_DIM:-256}"
MAX_TOKENS="${MAX_TOKENS:-512}"
BATCH_SIZE="${BATCH_SIZE:-512}"
GPU_SPEC="${GPU_SPEC:-B200:1}"
ATTN_BACKEND="${ATTN_BACKEND:-trtllm_mha}"

# HF output (private-by-default in the Modal job itself).
OUT_DATASET_ID="${OUT_DATASET_ID:-radna0/harmony-qwen3-embeddings-2m}"
OUT_SUBDIR_BASE="${OUT_SUBDIR_BASE:-behavior_v2}"

ts="$(date +%Y%m%d_%H%M%S)"
master_log="$CPU_LOG_DIR/run_modal_behavior_embed_2m_${ts}.log"
pidfile="$master_log.pid"

run_one() {
  local shard_idx="$1"
  local run_tag="qwen3_${REMOTE_TAG}_behavior_shard$(printf '%02d' "$shard_idx")_${ts}"
  local log_path="$LOG_DIR/${run_tag}.log"

  count_running() {
    local running=0
    shopt -s nullglob
    for pf in "$LOG_DIR"/*.pid; do
      local pid
      pid="$(cat "$pf" 2>/dev/null || true)"
      if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        running=$((running + 1))
      else
        rm -f "$pf" || true
      fi
    done
    shopt -u nullglob
    echo "$running"
  }

  # Throttle to MAX_PARALLEL concurrent Modal CLIs.
  while true; do
    local running
    running="$(count_running)"
    if [[ "$running" -lt "$MAX_PARALLEL" ]]; then
      break
    fi
    sleep 15
  done

  echo "[*] starting shard ${shard_idx}/${FILE_SHARD_COUNT} run_tag=${run_tag}" | tee -a "$master_log"

  nohup bash -lc "
    set -euo pipefail;
    set -a; source \"$ROOT_DIR/.env\"; set +a;
    export PYTHONUNBUFFERED=1;
    export QWEN_EMBED_GPU=\"$GPU_SPEC\";
    export MODEL_ID=\"$MODEL_ID\";
    export OUT_DIM=\"$OUT_DIM\";
    export MAX_TOKENS=\"$MAX_TOKENS\";
    export BATCH_SIZE=\"$BATCH_SIZE\";
    export SGLANG_ATTENTION_BACKEND=\"$ATTN_BACKEND\";
    export SGLANG_KV_CACHE_DTYPE=bf16;
    export SGLANG_DISABLE_CUDA_GRAPH=0;
    export USE_INPUT_IDS_COLUMN=1;
    export INPUT_IDS_COLUMN=input_ids;
    export TEXT_COLUMN=embed_text;
    export CANDIDATE_DATASET_ID=\"\";
    export CANDIDATE_SUBDIR=\"\";
    export CANDIDATE_MOUNT_DIR=\"/root/data/candidates/$REMOTE_TAG/$PACKED_BASE\";
    export FILE_SHARD_COUNT=\"$FILE_SHARD_COUNT\";
    export FILE_SHARD_INDEX=\"$shard_idx\";
    export RUN_TAG=\"$run_tag\";
    export OUT_DATASET_ID=\"$OUT_DATASET_ID\";
    export OUT_SUBDIR=\"$OUT_SUBDIR_BASE/$run_tag\";
    export SKIP_PREFETCH=1;
    modal run \"$ROOT_DIR/modal/qwen_embedding_sglang_scoring.py\";
  " >"$log_path" 2>&1 &

  echo $! >"${log_path}.pid"
  echo "[ok] pid=$(cat "${log_path}.pid") log=${log_path}" | tee -a "$master_log"
}

echo "[*] waiting for packed manifest: $PACKED_DIR/bucket_manifest.json" | tee -a "$master_log"
while [[ ! -f "$PACKED_DIR/bucket_manifest.json" ]]; do
  sleep 30
done
echo "[*] waiting for behavior QA report: $PACKED_DIR/behavior_signature_report.md" | tee -a "$master_log"
while [[ ! -f "$PACKED_DIR/behavior_signature_report.md" ]]; do
  sleep 30
done
echo "[ok] packed candidates ready: $PACKED_DIR" | tee -a "$master_log"

uniq_sigs="$(awk -F'\`' '/^- unique_signatures: / {print $2; exit}' "$PACKED_DIR/behavior_signature_report.md" 2>/dev/null || true)"
uniq_keysets="$(awk -F'\`' '/^- unique_tool_output_keysets: / {print $2; exit}' "$PACKED_DIR/behavior_signature_report.md" 2>/dev/null || true)"
echo "[*] QA metrics: unique_signatures=${uniq_sigs:-?} unique_tool_output_keysets=${uniq_keysets:-?}" | tee -a "$master_log"
if [[ "$uniq_sigs" =~ ^[0-9]+$ ]] && [[ "$uniq_sigs" -lt 5000 ]]; then
  echo "[err] behavior signatures collapsed (unique_signatures=$uniq_sigs)" | tee -a "$master_log"
  exit 3
fi
if [[ "$uniq_keysets" =~ ^[0-9]+$ ]] && [[ "$uniq_keysets" -lt 200 ]]; then
  echo "[err] tool output keysets too low (unique_tool_output_keysets=$uniq_keysets)" | tee -a "$master_log"
  exit 3
fi

num_files="$(find "$PACKED_DIR" -name '*.parquet' 2>/dev/null | wc -l | tr -d ' ')"
if [[ "${num_files}" -le 0 ]]; then
  echo "[err] no parquet files under $PACKED_DIR" | tee -a "$master_log"
  exit 2
fi
if [[ "${FILE_SHARD_COUNT}" -gt "${num_files}" ]]; then
  echo "[warn] FILE_SHARD_COUNT=${FILE_SHARD_COUNT} > parquet_files=${num_files}; reducing shard count" | tee -a "$master_log"
  FILE_SHARD_COUNT="${num_files}"
fi

remote_candidates_dir="/candidates/$REMOTE_TAG/$PACKED_BASE"
if [[ "${SKIP_CANDIDATE_UPLOAD}" == "1" ]] && modal volume ls harmony-embed-data "$remote_candidates_dir" >/dev/null 2>&1; then
  echo "[ok] candidates already present in Modal volume at $remote_candidates_dir; skipping upload" | tee -a "$master_log"
else
  echo "[*] uploading packed candidates -> Modal volume harmony-embed-data:/candidates/$REMOTE_TAG/" | tee -a "$master_log"
  modal volume put --force harmony-embed-data "$PACKED_DIR" "/candidates/$REMOTE_TAG/" >>"$master_log" 2>&1
  echo "[ok] upload complete" | tee -a "$master_log"
fi

if [[ "${SKIP_MODEL_PREFETCH}" == "1" ]] && modal volume ls qwen-embed-model-weights "/$MODEL_DIR_NAME" >/dev/null 2>&1; then
  echo "[ok] model already cached in qwen-embed-model-weights: /$MODEL_DIR_NAME; skipping prefetch" | tee -a "$master_log"
else
  prefetch_log="$LOG_DIR/qwen3_${REMOTE_TAG}_prefetch_model_${ts}.log"
  echo "[*] prefetching model into /models volume (once) log=$prefetch_log" | tee -a "$master_log"
  nohup bash -lc "
    set -euo pipefail;
    set -a; source \"$ROOT_DIR/.env\"; set +a;
    export PYTHONUNBUFFERED=1;
    export PREFETCH_ONLY=1;
    export SKIP_PREFETCH=0;
    export MODEL_ID=\"$MODEL_ID\";
    modal run \"$ROOT_DIR/modal/qwen_embedding_sglang_scoring.py\";
  " >\"$prefetch_log\" 2>&1
  echo "[ok] model prefetch done" | tee -a "$master_log"
fi

echo "[*] launching DP shards: FILE_SHARD_COUNT=$FILE_SHARD_COUNT MAX_PARALLEL=$MAX_PARALLEL" | tee -a "$master_log"
for i in $(seq 0 $((FILE_SHARD_COUNT - 1))); do
  run_one "$i"
done

echo "[ok] launched all shard runs (they continue in background); logs in $LOG_DIR" | tee -a "$master_log"
