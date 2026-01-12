#!/usr/bin/env bash
set -euo pipefail

PREFETCH_LOG="${1:?usage: $0 <prefetch_log> <artifacts_dir> <ids_dir> <out_dir>}"
ARTIFACTS_DIR="${2:?usage: $0 <prefetch_log> <artifacts_dir> <ids_dir> <out_dir>}"
IDS_DIR="${3:?usage: $0 <prefetch_log> <artifacts_dir> <ids_dir> <out_dir>}"
OUT_DIR="${4:?usage: $0 <prefetch_log> <artifacts_dir> <ids_dir> <out_dir>}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-Embedding-8B}"
MAX_TOKENS="${MAX_TOKENS:-512}"
NUM_WORKERS="${NUM_WORKERS:-180}"
TARGET_ROWS_PER_TASK="${TARGET_ROWS_PER_TASK:-100000}"

LOG_DIR="${LOG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cpu_logs}"
mkdir -p "$LOG_DIR"

ts="$(date +%Y%m%d_%H%M%S)"
log="$LOG_DIR/behavior2m_refresh_and_qa_${ts}.log"
pidfile="$log.pid"

cmd=(
  bash -lc
  "set -a; source .env; set +a;
   export PYTHONUNBUFFERED=1;
   echo \"[*] waiting for prefetch completion in: $PREFETCH_LOG\";
   while true; do
     if rg -n \"\\[ok\\] snapshot_download\" \"$PREFETCH_LOG\" >/dev/null 2>&1; then
       break;
     fi;
     sleep 30;
   done;
   echo \"[ok] prefetch done\";
   python cpu_make_embedding_candidates.py \\
     --in_dir \"$ARTIFACTS_DIR/normalized\" \\
     --out_dir \"$OUT_DIR\" \\
     --view behavior \\
     --ids_file \"$IDS_DIR/ids.txt\" \\
     --mix_group_map \"$IDS_DIR/id_mix_group.parquet\" \\
     --fail_on_missing_mix_group \\
     --require_valid_harmony \\
     --require_completion_nonempty \\
     --require_valid_tool_schema \\
     --num_workers $NUM_WORKERS \\
     --target_rows_per_task $TARGET_ROWS_PER_TASK \\
     --tokenize_model_id \"$MODEL_ID\" \\
     --tokenize_max_tokens $MAX_TOKENS \\
     ;
   python cpu_analyze_behavior_signatures.py \\
     --in_dir \"$OUT_DIR\" \\
     --max_rows 1000000 \\
     --out \"$OUT_DIR/behavior_signature_report.md\" \\
     ;
   echo \"[ok] wrote $OUT_DIR/behavior_signature_report.md\";
  "
)

echo "[*] starting watcher+builder"
echo "[*] log: $log"
nohup "${cmd[@]}" >"$log" 2>&1 &
echo $! >"$pidfile"
echo "[ok] pid=$(cat "$pidfile")"
echo "[ok] tail -f $log"

