#!/usr/bin/env bash
set -euo pipefail

# Run a small matrix of cached DFlash decode benchmarks on TPU.
#
# Required env:
#   TEACHER_SNAPSHOT_DIR
#   DRAFT_RUN_DIR
#
# Optional env:
#   ALSO_RUN_BASELINE=1
#   DFLASH_DECODE_SCRIPT=tpu_dflash_spec_decode_blockverify_v1.py
#   BLOCK_SIZES="8,16"
#   MAX_NEW_TOKENS_LIST="256,1024"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../logs/tpu_dflash"
mkdir -p "$LOG_DIR"

TEACHER_SNAPSHOT_DIR="${TEACHER_SNAPSHOT_DIR:-}"
DRAFT_RUN_DIR="${DRAFT_RUN_DIR:-}"
if [[ -z "$TEACHER_SNAPSHOT_DIR" || -z "$DRAFT_RUN_DIR" ]]; then
  echo "TEACHER_SNAPSHOT_DIR and DRAFT_RUN_DIR are required env vars" >&2
  exit 2
fi

BLOCK_SIZES="${BLOCK_SIZES:-8}"
MAX_NEW_TOKENS_LIST="${MAX_NEW_TOKENS_LIST:-256}"

IFS=',' read -ra BS <<<"$BLOCK_SIZES"
IFS=',' read -ra MNT <<<"$MAX_NEW_TOKENS_LIST"

for b in "${BS[@]}"; do
  b="${b//[[:space:]]/}"
  [[ -z "$b" ]] && continue
  for n in "${MNT[@]}"; do
    n="${n//[[:space:]]/}"
    [[ -z "$n" ]] && continue
    TS="$(date -u +%Y%m%d_%H%M%S)"
    export RUN_NAME="tpu_bench_cached_b${b}_n${n}_${TS}"
    "${SCRIPT_DIR}/run_tpu_dflash_decode_cached_logged.sh" \
      --platform tpu \
      --max-prompt-len 512 \
      --max-new-tokens "$n" \
      --block-size "$b"
  done
done

echo "[+] logs: ${LOG_DIR}" >&2
