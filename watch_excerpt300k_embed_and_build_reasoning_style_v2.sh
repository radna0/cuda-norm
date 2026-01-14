#!/usr/bin/env bash
set -euo pipefail

# End-to-end watcher:
# 1) waits for excerpt300k DP embed shards 0-5 to finish
# 2) launches shards 6-7 (same RUN_GROUP)
# 3) waits for shards 6-7 to finish
# 4) downloads embeddings
# 5) selects reasoning_style_10k_v2 with per-domain quotas
# 6) exports JSONL/Parquet pack from normalized shards
# 7) uploads pack into the v2 calib packs repo (private)
#
# Logs:
# - Start this via nohup into harmony/cuda-norm/cpu_logs/

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

set -a
source harmony/cuda-norm/.env
set +a

mkdir -p harmony/cuda-norm/cpu_logs harmony/cuda-norm/hf_upload_logs harmony/cuda-norm/modal_parallel_logs

RUN_GROUP_FILE="harmony/cuda-norm/modal_parallel_logs/LATEST_EXCERPT300K_RUN_GROUP.txt"
if [[ ! -f "$RUN_GROUP_FILE" ]]; then
  echo "[err] missing $RUN_GROUP_FILE (run group not set)" >&2
  exit 2
fi
RUN_GROUP="$(cat "$RUN_GROUP_FILE")"
if [[ -z "$RUN_GROUP" ]]; then
  echo "[err] empty run group in $RUN_GROUP_FILE" >&2
  exit 2
fi

log_ts=""
log_ts_file="harmony/cuda-norm/modal_parallel_logs/LATEST_EXCERPT300K_LOG_TS.txt"
if [[ -f "$log_ts_file" ]]; then
  log_ts="$(tr -d ' \t\r\n' <"$log_ts_file" || true)"
fi
if [[ -z "$log_ts" || ! "$log_ts" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  latest_shard0_log="$(ls -t harmony/cuda-norm/modal_parallel_logs/excerpt300k_embed_shard0_*.log | head -n 1)"
  latest_shard0_base="$(basename "$latest_shard0_log")"
  log_ts="${latest_shard0_base#excerpt300k_embed_shard0_}"
  log_ts="${log_ts%.log}"
  if [[ -z "$log_ts" || "$log_ts" == "$latest_shard0_base" || ! "$log_ts" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
    echo "[err] could not parse log timestamp from $latest_shard0_log" >&2
    exit 2
  fi
fi

if [[ ! -f "harmony/cuda-norm/modal_parallel_logs/excerpt300k_embed_shard0_${log_ts}.log" ]]; then
  echo "[err] expected shard0 log missing for log_ts=$log_ts" >&2
  exit 2
fi

echo "[*] RUN_GROUP=$RUN_GROUP log_ts=$log_ts"

shard_log() {
  local shard="$1"
  echo "harmony/cuda-norm/modal_parallel_logs/excerpt300k_embed_shard${shard}_${log_ts}.log"
}

wait_shards_done() {
  local from="$1"
  local to="$2"
  echo "[*] waiting for shards ${from}-${to} to complete..."
  while true; do
    local all_done="1"
    for ((i=from; i<=to; i++)); do
      local log
      log="$(shard_log "$i")"
      if [[ ! -f "$log" ]]; then
        all_done="0"
        continue
      fi
      if ! rg -n "\\[ok\\] upload complete|✓ App completed\\.|Stopping app - local entrypoint completed\\." "$log" >/dev/null 2>&1; then
        all_done="0"
      fi
    done
    if [[ "$all_done" == "1" ]]; then
      echo "[ok] shards ${from}-${to} complete"
      break
    fi
    sleep 20
  done
}

wait_shards_done 0 5

echo "[*] launching wave2 shards 6-7 (DP, same RUN_GROUP)..."
RUN_GROUP="$RUN_GROUP" ./harmony/cuda-norm/run_modal_excerpt300k_embed_dp8.sh 6 7

wait_shards_done 6 7

echo "[*] downloading embeddings from HF..."
EMBED_REPO="radna0/harmony-qwen3-reasoning-excerpt-embeddings-v2-300k"
LOCAL_EMBED_ROOT="/dev/shm/hf_${RUN_GROUP}_excerpt_embeddings"
python - <<PY
from huggingface_hub import snapshot_download
import os

repo_id = "$EMBED_REPO"
run_group = "$RUN_GROUP"
local_dir = "$LOCAL_EMBED_ROOT"

allow = [f"embeddings/{run_group}/**"]
snap = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=allow,
    local_dir=local_dir,
)
print("[ok] snapshot_download", snap)
PY

IN_DIR="$LOCAL_EMBED_ROOT/embeddings/$RUN_GROUP"
if [[ ! -d "$IN_DIR" ]]; then
  echo "[err] missing downloaded embeddings dir: $IN_DIR" >&2
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
SEL_DIR="harmony/cuda-norm/artifacts/selection/reasoning_style_10k_v2_${TS}"
mkdir -p "$SEL_DIR"

echo "[*] selecting reasoning_style_10k_v2 (per-domain quotas)..."
python harmony/cuda-norm/cpu_select_reasoning_style_ids_from_excerpt_embeddings_by_domain.py \
  --in_dir "$IN_DIR" \
  --out_dir "$SEL_DIR" \
  --target_k 10000 \
  --domain_quota_json '{"math":2000,"proof":2000,"science":2000,"agentic":2000,"chat_if":2000}' \
  --bits 18 \
  --seed 0 \
  --dense_fraction 0.30

SELECTED_META="$(ls -t "$SEL_DIR"/*selected_meta*.parquet | head -n 1)"
if [[ ! -f "$SELECTED_META" ]]; then
  echo "[err] selection did not produce selected_meta parquet in $SEL_DIR" >&2
  exit 2
fi

NORM_ROOT="/dev/shm/harmony_artifacts_norm_2m_20260112_164801/normalized"
if [[ ! -d "$NORM_ROOT" ]]; then
  echo "[err] missing normalized_root: $NORM_ROOT" >&2
  exit 2
fi

PACK_DIR="harmony/cuda-norm/artifacts/calib_packs/reasoning_style_10k_v2_${TS}"
mkdir -p "$PACK_DIR"

echo "[*] exporting Harmony pack from normalized shards..."
python harmony/cuda-norm/cpu_export_calib_pack_from_selected_meta.py \
  --normalized_root "$NORM_ROOT" \
  --selected_meta "$SELECTED_META" \
  --out_dir "$PACK_DIR" \
  --pack_name "reasoning_style_10k_v2" \
  --num_workers 180

echo "[*] staging for HF upload (avoid filename collisions)..."
PUBLISH_DIR="harmony/cuda-norm/artifacts/calib_packs_publish/reasoning_style_10k_v2_${TS}"
mkdir -p "$PUBLISH_DIR/packs/reasoning_style_10k_v2"
cp -al "$PACK_DIR"/* "$PUBLISH_DIR/packs/reasoning_style_10k_v2/"

echo "[*] uploading pack to HF v2 repo (private)..."
UPLOAD_LOG="harmony/cuda-norm/hf_upload_logs/reasoning_style_10k_v2_upload_${TS}.log"
./harmony/cuda-norm/upload_folder_to_hf_dataset_repo.sh radna0/harmony-qwen3-calib-packs-v2-20260113 "$PUBLISH_DIR" >"$UPLOAD_LOG" 2>&1
echo "[ok] upload complete log=$UPLOAD_LOG"

echo "[ok] done: selection=$SEL_DIR pack=$PACK_DIR run_group=$RUN_GROUP"
