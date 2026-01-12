#!/usr/bin/env bash
set -euo pipefail

IN_DIR="${1:?usage: $0 <behavior_candidates_dir> <packed_out_dir>}"
OUT_DIR="${2:?usage: $0 <behavior_candidates_dir> <packed_out_dir>}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/cpu_logs}"
mkdir -p "$LOG_DIR"

BUCKET_EDGES="${BUCKET_EDGES:-0,128,256,512,inf}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-200000}"

ts="$(date +%Y%m%d_%H%M%S)"
log="$LOG_DIR/pack_behavior_candidates_${ts}.log"
pidfile="$log.pid"

cmd=(
  bash -lc
  "set -euo pipefail;
   export PYTHONUNBUFFERED=1;
   echo \"[*] waiting for candidates_manifest.json in: $IN_DIR\";
   while true; do
     if [[ -f \"$IN_DIR/candidates_manifest.json\" ]]; then
       break;
     fi;
     sleep 30;
   done;
   echo \"[ok] found $IN_DIR/candidates_manifest.json\";
   python \"$ROOT_DIR/cpu_pack_candidates_by_tok_len.py\" \\
     --in_dir \"$IN_DIR\" \\
     --out_dir \"$OUT_DIR\" \\
     --bucket_edges \"$BUCKET_EDGES\" \\
     --rows_per_shard $ROWS_PER_SHARD \\
     --dedup_by_id \\
     ;
   python \"$ROOT_DIR/cpu_analyze_behavior_signatures.py\" \\
     --in_dir \"$OUT_DIR\" \\
     --max_rows 1000000 \\
     --out \"$OUT_DIR/behavior_signature_report.md\" \\
     ;
   echo \"[ok] packed + wrote $OUT_DIR/behavior_signature_report.md\";
  "
)

echo "[*] starting pack watcher"
echo "[*] log: $log"
nohup "${cmd[@]}" >"$log" 2>&1 &
echo $! >"$pidfile"
echo "[ok] pid=$(cat "$pidfile")"
echo "[ok] tail -f $log"

