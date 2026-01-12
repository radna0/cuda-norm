#!/usr/bin/env bash
set -euo pipefail

IN_DIR="${1:?usage: $0 <in_dir> <out_dir> [ids_file] }"
OUT_DIR="${2:?usage: $0 <in_dir> <out_dir> [ids_file] }"
IDS_FILE="${3:-}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/cpu_logs}"
NUM_WORKERS="${NUM_WORKERS:-180}"

mkdir -p "$LOG_DIR"

ts="$(date +%Y%m%d_%H%M%S)"
log="$LOG_DIR/behavior_candidates_${ts}.log"
pidfile="$log.pid"

cmd=(
  python "$ROOT_DIR/cpu_make_embedding_candidates.py"
  --in_dir "$IN_DIR"
  --out_dir "$OUT_DIR"
  --view behavior
  --require_valid_harmony
  --require_completion_nonempty
  --require_valid_tool_schema
  --num_workers "$NUM_WORKERS"
)

if [[ -n "$IDS_FILE" ]]; then
  cmd+=(--ids_file "$IDS_FILE")
fi

echo "[*] starting: ${cmd[*]}"
echo "[*] log: $log"
nohup "${cmd[@]}" >"$log" 2>&1 &
echo $! >"$pidfile"
echo "[ok] pid=$(cat "$pidfile")"
