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

TPU_LOCK_PATH="${TPU_LOCK_PATH:-/dev/shm/tpu.lock}"

TS="$(date -u +%Y%m%d_%H%M%S)"
PIPE_NAME="${PIPE_NAME:-tpu_dflash_autocontinue_${TS}}"
PIPE_LOG="${LOG_DIR}/${PIPE_NAME}.log"
PID_FILE="${LOG_DIR}/${PIPE_NAME}.pid"

AUTO_LONG_RUN="${AUTO_LONG_RUN:-1}"          # 1 = start long run after 200-step sanity
LONG_STEPS="${LONG_STEPS:-2000}"
SANITY_STEPS="${SANITY_STEPS:-200}"
SANITY_SAVE_STEPS="${SANITY_SAVE_STEPS:-200}"
LONG_SAVE_STEPS="${LONG_SAVE_STEPS:-500}"
BENCH_STEPS="${BENCH_STEPS:-1000,2000}"

# Runtime knobs (keep consistent with cache build).
BLOCK_SIZE="${BLOCK_SIZE:-}"
PAGE_SIZE="${PAGE_SIZE:-}"
HBM_UTILIZATION="${HBM_UTILIZATION:-}"
PREFILL_CHUNK="${PREFILL_CHUNK:-}"

# Optional: speed up teacher load in benches by using an EasyDeL-native teacher dir.
TEACHER_EASYDEL_DIR="${TEACHER_EASYDEL_DIR:-}"

echo "pid=$$" > "${PID_FILE}"

tpu_exec() {
  bash -c 'exec 9>"$1"; flock -x 9; shift; exec "$@"' bash "${TPU_LOCK_PATH}" "$@"
}

{
  echo "[*] cache_log=${CACHE_LOG}"
  echo "[*] teacher_snapshot_dir=${TEACHER_SNAPSHOT_DIR}"
  echo "[*] auto_long_run=${AUTO_LONG_RUN} sanity_steps=${SANITY_STEPS} long_steps=${LONG_STEPS}"
  echo "[*] tpu_lock=${TPU_LOCK_PATH}"

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

  if [[ -f "${cache_dir}/meta.json" ]]; then
    meta_block_size="$("${VENV_PY}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("block_size",""))' "${cache_dir}/meta.json")"
    meta_page_size="$("${VENV_PY}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("page_size",""))' "${cache_dir}/meta.json")"
    meta_hbm_util="$("${VENV_PY}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("hbm_utilization",""))' "${cache_dir}/meta.json")"
    meta_prefill_chunk="$("${VENV_PY}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("prefill_chunk",""))' "${cache_dir}/meta.json")"
    if [[ -z "${BLOCK_SIZE}" && -n "${meta_block_size}" ]]; then BLOCK_SIZE="${meta_block_size}"; fi
    if [[ -z "${PAGE_SIZE}" && -n "${meta_page_size}" ]]; then PAGE_SIZE="${meta_page_size}"; fi
    if [[ -z "${HBM_UTILIZATION}" && -n "${meta_hbm_util}" ]]; then HBM_UTILIZATION="${meta_hbm_util}"; fi
    if [[ -z "${PREFILL_CHUNK}" && -n "${meta_prefill_chunk}" ]]; then PREFILL_CHUNK="${meta_prefill_chunk}"; fi
  fi

  BLOCK_SIZE="${BLOCK_SIZE:-8}"
  PAGE_SIZE="${PAGE_SIZE:-128}"
  HBM_UTILIZATION="${HBM_UTILIZATION:-0.20}"
  PREFILL_CHUNK="${PREFILL_CHUNK:-256}"

  echo "[*] running cache parity check (multitoken verify vs cached targets)..."
  tpu_exec "${VENV_PY}" -u "${SCRIPTS}/tpu_verify_multitoken_parity.py" \
    --cache-dir "${cache_dir}" \
    --teacher-snapshot-dir "${TEACHER_SNAPSHOT_DIR}" \
    ${TEACHER_EASYDEL_DIR:+--teacher-easydel-dir "${TEACHER_EASYDEL_DIR}"} \
    --sample-idx 0 \
    --block-size "${BLOCK_SIZE}" \
    --page-size "${PAGE_SIZE}" \
    --hbm-utilization "${HBM_UTILIZATION}" \
    --prefill-chunk "${PREFILL_CHUNK}"

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
    latest_run_dir="$(ls -1dt "${ckpt_root}"/run-* 2>/dev/null | head -n 1 || true)"
    if [[ -z "${latest_run_dir}" ]]; then
      echo "[!] could not find draft run dir under ${ckpt_root}; skipping decode bench" >&2
      exit 0
    fi

    echo "[*] running cached decode benchmark(s) (blockverify spec-v1) ..."
    # NOTE: use the eSurge-native benchmark harness (no HF-style use_cache).
    # Run a small set of checkpoints (e.g. 1000,2000) to measure speedup trajectory
    # without interrupting training.
    IFS=',' read -r -a bench_steps_arr <<< "${BENCH_STEPS}"
    for s in "${bench_steps_arr[@]}"; do
      s_trim="$(echo "${s}" | tr -d '[:space:]')"
      [[ -z "${s_trim}" ]] && continue
      step_dir="${ckpt_root}/run-${s_trim}"
      if [[ ! -d "${step_dir}" ]]; then
        echo "[*] bench: missing ${step_dir}, skipping"
        continue
      fi
      bench_log="${LOG_DIR}/esurge_bench_${RUN_NAME}_run${s_trim}_${TS}.log"
      tpu_exec "${VENV_PY}" -u "${SCRIPTS}/tpu_esurge_dflash_bench.py" \
        --teacher-snapshot-dir "${TEACHER_SNAPSHOT_DIR}" \
        ${TEACHER_EASYDEL_DIR:+--teacher-easydel-dir "${TEACHER_EASYDEL_DIR}"} \
        --draft-run-dir "${step_dir}" \
        --block-size "${BLOCK_SIZE}" \
        --max-new-tokens 2048 \
        --max-model-len 4096 \
        --page-size "${PAGE_SIZE}" \
        --hbm-utilization "${HBM_UTILIZATION}" \
        --prompt-from-cache-dir "${cache_dir}" \
        --cache-sample-idx 0 \
        --also-run-baseline \
        2>&1 | tee "${bench_log}"
      echo "[+] decode bench done (log: ${bench_log})"
    done

    # Always bench the latest checkpoint too (in case BENCH_STEPS misses it).
    bench_log="${LOG_DIR}/esurge_bench_${RUN_NAME}_latest_${TS}.log"
    tpu_exec "${VENV_PY}" -u "${SCRIPTS}/tpu_esurge_dflash_bench.py" \
      --teacher-snapshot-dir "${TEACHER_SNAPSHOT_DIR}" \
      ${TEACHER_EASYDEL_DIR:+--teacher-easydel-dir "${TEACHER_EASYDEL_DIR}"} \
      --draft-run-dir "${latest_run_dir}" \
      --block-size "${BLOCK_SIZE}" \
      --max-new-tokens 2048 \
      --max-model-len 4096 \
      --page-size "${PAGE_SIZE}" \
      --hbm-utilization "${HBM_UTILIZATION}" \
      --prompt-from-cache-dir "${cache_dir}" \
      --cache-sample-idx 0 \
      --also-run-baseline \
      2>&1 | tee "${bench_log}"
    echo "[+] decode bench done (log: ${bench_log})"
  fi
} 2>&1 | tee "${PIPE_LOG}"

echo "[+] pipeline log: ${PIPE_LOG}" >&2
