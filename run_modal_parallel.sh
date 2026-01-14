#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CMD_FILE="${1:-}"
if [[ -z "$CMD_FILE" ]] || [[ "$CMD_FILE" == "-h" ]] || [[ "$CMD_FILE" == "--help" ]]; then
  cat <<'EOF'
usage: run_modal_parallel.sh <commands.txt>

Runs up to 6 concurrent `modal run ...` CLI jobs, each via nohup and with a per-job .log + .pid under:
  $LOG_DIR (default: ./modal_parallel_logs)

commands.txt format (one per line):
  - Blank lines and lines starting with # are ignored.
  - Optional tag prefix:
      my_tag: modal run modal/some_job.py --arg 123
    Otherwise the tag is derived from the command.

Env:
  MAX_PARALLEL  Max concurrent Modal CLI jobs (default: 6, hard-capped to 6).
  LOG_DIR       Log output directory (default: ./modal_parallel_logs).
EOF
  exit 2
fi

if [[ ! -f "$CMD_FILE" ]]; then
  echo "[err] commands file not found: $CMD_FILE" >&2
  exit 2
fi

LOG_DIR="${LOG_DIR:-$ROOT_DIR/modal_parallel_logs}"
mkdir -p "$LOG_DIR"

MAX_PARALLEL="${MAX_PARALLEL:-6}"
if ! [[ "$MAX_PARALLEL" =~ ^[0-9]+$ ]]; then
  echo "[err] MAX_PARALLEL must be an integer, got: $MAX_PARALLEL" >&2
  exit 2
fi
if [[ "$MAX_PARALLEL" -gt 6 ]]; then
  MAX_PARALLEL=6
fi
if [[ "$MAX_PARALLEL" -le 0 ]]; then
  echo "[err] MAX_PARALLEL must be > 0" >&2
  exit 2
fi

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

sanitize_tag() {
  local s="$1"
  s="${s//\//_}"
  s="${s// /_}"
  s="${s//[^A-Za-z0-9_.-]/_}"
  s="${s#__}"
  s="${s%%__}"
  echo "${s:0:120}"
}

derive_tag() {
  local cmd="$1"
  local s="$cmd"
  s="${s#modal run }"
  s="${s#python }"
  s="${s#bash }"
  s="${s%% *}"
  if [[ -z "$s" ]]; then
    s="modal_job"
  fi
  sanitize_tag "$s"
}

ts="$(date +%Y%m%d_%H%M%S)"
echo "[*] LOG_DIR=$LOG_DIR MAX_PARALLEL=$MAX_PARALLEL ts=$ts"

started=0
while IFS= read -r raw || [[ -n "$raw" ]]; do
  line="$(echo "$raw" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  if [[ -z "$line" ]] || [[ "$line" == \#* ]]; then
    continue
  fi

  tag=""
  cmd="$line"
  if [[ "$line" =~ ^([^:]{1,80}):[[:space:]]+(.+)$ ]]; then
    tag="$(sanitize_tag "${BASH_REMATCH[1]}")"
    cmd="${BASH_REMATCH[2]}"
  else
    tag="$(derive_tag "$cmd")"
  fi

  # Throttle to MAX_PARALLEL concurrent Modal CLIs.
  while true; do
    running="$(count_running)"
    if [[ "$running" -lt "$MAX_PARALLEL" ]]; then
      break
    fi
    sleep 5
  done

  rid="$(python -c 'import os; print(f"{os.getpid()}_{os.urandom(3).hex()}")')"
  log_path="$LOG_DIR/${tag}_${ts}_${rid}.log"
  cmd_path="$LOG_DIR/${tag}_${ts}_${rid}.cmd"

  printf '%s\n' "$cmd" >"$cmd_path"
  echo "[*] launch tag=$tag log=$log_path"

  nohup bash -lc "
    set -euo pipefail;
    if [[ -f \"$ROOT_DIR/.env\" ]]; then set -a; source \"$ROOT_DIR/.env\"; set +a; fi;
    export PYTHONUNBUFFERED=1;
    $cmd
  " >"$log_path" 2>&1 &
  echo $! >"${log_path}.pid"
  started=$((started + 1))
done <"$CMD_FILE"

echo "[ok] launched $started job(s); logs in $LOG_DIR"
