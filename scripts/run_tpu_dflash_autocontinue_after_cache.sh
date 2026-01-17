#!/usr/bin/env bash
set -euo pipefail

# Auto-continue pipeline:
# 1) Wait for a cache builder log to emit "[done] wrote cache_dir=...".
# 2) Verify cache parity vs current verify kernel.
# 3) Run a 200-step training sanity job.
# 4) Optionally start a longer training run, then decode-benchmark.
#
# This script is TPU-safe: it does no TPU work until the cache build finishes.
#
# Usage:
#   CACHE_LOG=... TEACHER_SNAPSHOT_DIR=... ./harmony/cuda-norm/scripts/run_tpu_dflash_autocontinue_after_cache.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPTS="${ROOT}/harmony/cuda-norm/scripts"
VENV_PY="${ROOT}/harmony/cuda-norm/.venv-easydel/bin/python"

: "${CACHE_LOG:?Set CACHE_LOG=harmony/cuda-norm/logs/tpu_dflash/<cache_build>.log}"
: "${TEACHER_SNAPSHOT_DIR:?Set TEACHER_SNAPSHOT_DIR=/dev/shm/easydel_teachers/<teacher> (or HF snapshot dir)}"

LOG_DIR="${ROOT}/harmony/cuda-norm/logs/tpu_dflash"
mkdir -p "${LOG_DIR}"

TS="$(date -u +%Y%m%d_%H%M%S)"
PIPE_NAME="${PIPE_NAME:-tpu_dflash_autocontinue_${TS}}"
PIPE_LOG="${LOG_DIR}/${PIPE_NAME}.log"
PID_FILE="${LOG_DIR}/${PIPE_NAME}.pid"

AUTO_LONG_RUN="${AUTO_LONG_RUN:-1}"          # 1 = start long run after 200-step sanity
LONG_STEPS="${LONG_STEPS:-2000}"
SANITY_STEPS="${SANITY_STEPS:-200}"
SANITY_SAVE_STEPS="${SANITY_SAVE_STEPS:-200}"
LONG_SAVE_STEPS="${LONG_SAVE_STEPS:-500}"

echo "pid=$$" > "${PID_FILE}"

{
  echo "[*] cache_log=${CACHE_LOG}"
  echo "[*] teacher_snapshot_dir=${TEACHER_SNAPSHOT_DIR}"
  echo "[*] auto_long_run=${AUTO_LONG_RUN} sanity_steps=${SANITY_STEPS} long_steps=${LONG_STEPS}"

  echo "[*] waiting for cache build to finish..."
  cache_dir=""
  while true; do
    line="$(rg -n "\\[done\\] wrote cache_dir=" -S "${CACHE_LOG}" | tail -n 1 || true)"
    if [[ -n "${line}" ]]; then
      cache_dir="$(echo "${line}" | sed -E 's/.*wrote cache_dir=//')"
      break
    fi
    sleep 30
  done
  echo "[+] cache_dir=${cache_dir}"

  echo "[*] running cache parity check (multitoken verify vs cached targets)..."
  "${VENV_PY}" -u "${SCRIPTS}/tpu_verify_multitoken_parity.py" \
    --cache-dir "${cache_dir}" \
    --teacher-snapshot-dir "${TEACHER_SNAPSHOT_DIR}" \
    --sample-idx 0 \
    --block-size 8 \
    --page-size 32 \
    --hbm-utilization 0.20 \
    --prefill-chunk 256

  echo "[*] starting training sanity run (${SANITY_STEPS} steps)..."
  export CACHE_DIR="${cache_dir}"
  export TEACHER_SNAPSHOT="${TEACHER_SNAPSHOT_DIR}"
  export RUN_NAME="gptoss20b_dflash_sanity_${TS}"
  export MODEL_NAME="${RUN_NAME}"
  export MAX_TRAINING_STEPS="${SANITY_STEPS}"
  export SAVE_STEPS="${SANITY_SAVE_STEPS}"
  export LOG_STEPS="${LOG_STEPS:-10}"
  export REPORT_STEPS="${REPORT_STEPS:-10}"
  "${SCRIPTS}/run_tpu_dflash_train_logged.sh"

  sanity_pid="$(cat "${LOG_DIR}/${RUN_NAME}.pid")"
  sanity_log="${LOG_DIR}/${RUN_NAME}.log"
  echo "[*] sanity training pid=${sanity_pid} log=${sanity_log}"
  while ps -p "${sanity_pid}" >/dev/null 2>&1; do
    tail -n 20 "${sanity_log}" || true
    sleep 60
  done
  echo "[+] sanity training finished"

  if [[ "${AUTO_LONG_RUN}" == "1" ]]; then
    echo "[*] starting long training run (${LONG_STEPS} steps)..."
    export RUN_NAME="gptoss20b_dflash_long_${TS}"
    export MODEL_NAME="${RUN_NAME}"
    export MAX_TRAINING_STEPS="${LONG_STEPS}"
    export SAVE_STEPS="${LONG_SAVE_STEPS}"
    "${SCRIPTS}/run_tpu_dflash_train_logged.sh"
    long_pid="$(cat "${LOG_DIR}/${RUN_NAME}.pid")"
    long_log="${LOG_DIR}/${RUN_NAME}.log"
    echo "[*] long training pid=${long_pid} log=${long_log}"
    while ps -p "${long_pid}" >/dev/null 2>&1; do
      tail -n 30 "${long_log}" || true
      sleep 120
    done
    echo "[+] long training finished"

    # Best-effort: locate most recent run directory.
    ckpt_root="${CKPT_DIR:-/dev/shm/dflash-checkpoints}/${RUN_NAME}"
    draft_run_dir="$(ls -1dt "${ckpt_root}"/run-* 2>/dev/null | head -n 1 || true)"
    if [[ -z "${draft_run_dir}" ]]; then
      echo "[!] could not find draft run dir under ${ckpt_root}; skipping decode bench" >&2
      exit 0
    fi

    echo "[*] running cached decode benchmark (blockverify spec-v1) ..."
    export TEACHER_SNAPSHOT_DIR="${TEACHER_SNAPSHOT_DIR}"
    export DRAFT_RUN_DIR="${draft_run_dir}"
    export ALSO_RUN_BASELINE=1
    "${SCRIPTS}/run_tpu_dflash_decode_cached_logged.sh" --max-new-tokens 256 --block-size 8 --page-size 32 --hbm-utilization 0.20
    echo "[+] decode bench done (see logs in ${LOG_DIR})"
  fi
} 2>&1 | tee "${PIPE_LOG}"

echo "[+] pipeline log: ${PIPE_LOG}" >&2
