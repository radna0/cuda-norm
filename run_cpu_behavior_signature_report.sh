#!/usr/bin/env bash
set -euo pipefail

IN_DIR="${1:?usage: $0 <candidates_dir> [max_rows]}"
MAX_ROWS="${2:-200000}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/cpu_logs}"

mkdir -p "$LOG_DIR"

ts="$(date +%Y%m%d_%H%M%S)"
log="$LOG_DIR/behavior_signature_report_${ts}.log"
out="$LOG_DIR/behavior_signature_report_${ts}.md"
pidfile="$log.pid"

cmd=(
  python "$ROOT_DIR/cpu_analyze_behavior_signatures.py"
  --in_dir "$IN_DIR"
  --max_rows "$MAX_ROWS"
  --out "$out"
)

echo "[*] starting: ${cmd[*]}"
echo "[*] log: $log"
nohup "${cmd[@]}" >"$log" 2>&1 &
echo $! >"$pidfile"
echo "[ok] pid=$(cat "$pidfile")"
echo "[ok] report_path=$out"
