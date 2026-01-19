#!/usr/bin/env bash
set -euo pipefail

# Run baseline vs DFLASH throughput on TPU via `sglang-jax` and save logs/results.
#
# Requirements:
# - `harmony/cuda-norm/external/sglang-jax` is present.
# - Target model is available via HF cache or local path.
# - Draft checkpoint is an HF-style DFlashDraftModel directory (see export script).
#
# Example:
#   source harmony/cuda-norm/.env
#   MODEL_PATH=/dev/shm/hf/hub/models--unsloth--gpt-oss-20b-BF16/snapshots/<sha> \
#   DRAFT_PATH=harmony/cuda-norm/artifacts/dflash_draft_ckpts/sglang_gptoss20b_run10000 \
#   PROMPT_LEN=1024 MAX_NEW_TOKENS=2048 NUM_PROMPTS=32 \
#   bash harmony/cuda-norm/scripts/tpu_sglang_jax_bench_dflash_vs_base.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
SGLJAX_ROOT="${REPO_ROOT}/harmony/cuda-norm/external/sglang-jax"
SGLJAX_PY="${SGLJAX_PY:-}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/harmony/cuda-norm/logs/tpu_dflash}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-sglang_jax_dflash_vs_base_${TS}}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"
SKIP_DFLASH="${SKIP_DFLASH:-0}"
TPU_LOCK_PATH="${TPU_LOCK_PATH:-/dev/shm/tpu.lock}"

MODEL_PATH="${MODEL_PATH:-}"
DRAFT_PATH="${DRAFT_PATH:-}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"

PROMPT_LEN="${PROMPT_LEN:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
NUM_PROMPTS="${NUM_PROMPTS:-32}"
SEED="${SEED:-1}"

MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.70}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-}"

# Optional: override JAX precompile buckets to avoid huge compile/OOM during
# speculative decoding. Space- or comma-separated lists.
PRECOMPILE_BS_PADDINGS="${PRECOMPILE_BS_PADDINGS:-}"
PRECOMPILE_TOKEN_PADDINGS="${PRECOMPILE_TOKEN_PADDINGS:-}"

# TPU-first default (override if needed).
export JAX_PLATFORMS="${JAX_PLATFORMS:-tpu}"

if [[ -z "${HF_TOKEN:-}" ]] && [[ -f "${REPO_ROOT}/harmony/cuda-norm/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${REPO_ROOT}/harmony/cuda-norm/.env"
  set +a
fi

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Missing MODEL_PATH env var." >&2
  exit 2
fi
if [[ -z "${DRAFT_PATH}" ]]; then
  echo "Missing DRAFT_PATH env var." >&2
  exit 2
fi

MODEL_PATH="$(readlink -f -- "${MODEL_PATH}")"
DRAFT_PATH="$(readlink -f -- "${DRAFT_PATH}")"

mkdir -p "${LOG_DIR}"

BASE_LOG="${LOG_DIR}/${RUN_NAME}.baseline.log"
DFLASH_LOG="${LOG_DIR}/${RUN_NAME}.dflash.log"

if [[ -z "${SGLJAX_PY}" ]] && [[ -x "${SGLJAX_ROOT}/.venv/bin/python" ]]; then
  SGLJAX_PY="${SGLJAX_ROOT}/.venv/bin/python"
fi
if [[ -z "${SGLJAX_PY}" ]]; then
  SGLJAX_PY="python"
fi

# Serialize TPU work across all launchers on this machine. We intentionally hold
# the lock for the whole script so baseline and DFLASH runs cannot interleave
# with other TPU jobs (which would trigger recompiles + OOM + confusing metrics).
exec 9>"${TPU_LOCK_PATH}"
flock -x 9

COMMON_ARGS=(
  --model-path "${MODEL_PATH}"
  --tokenizer-path "${MODEL_PATH}"
  --grammar-backend none
  --dtype bfloat16
  --page-size 1
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --context-length "${CONTEXT_LENGTH}"
  ${MAX_SEQ_LEN:+--max-seq-len} ${MAX_SEQ_LEN:+"${MAX_SEQ_LEN}"}
  ${CHUNKED_PREFILL_SIZE:+--chunked-prefill-size} ${CHUNKED_PREFILL_SIZE:+"${CHUNKED_PREFILL_SIZE}"}
  --disable-overlap-schedule
  --skip-server-warmup
  --dataset-name random
  --random-input "${PROMPT_LEN}"
  --random-output "${MAX_NEW_TOKENS}"
  --num-prompts "${NUM_PROMPTS}"
  --seed "${SEED}"
)
if [[ -n "${MAX_TOTAL_TOKENS}" ]]; then
  COMMON_ARGS+=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
fi
if [[ -n "${MAX_RUNNING_REQUESTS}" ]]; then
  COMMON_ARGS+=(--max-running-requests "${MAX_RUNNING_REQUESTS}")
fi

_split_list() {
  local s="${1}"
  s="${s//,/ }"
  # shellcheck disable=SC2206
  echo ${s}
}
if [[ -n "${PRECOMPILE_BS_PADDINGS}" ]]; then
  # shellcheck disable=SC2207
  BS_LIST=($(_split_list "${PRECOMPILE_BS_PADDINGS}"))
  COMMON_ARGS+=(--precompile-bs-paddings "${BS_LIST[@]}")
fi
if [[ -n "${PRECOMPILE_TOKEN_PADDINGS}" ]]; then
  # shellcheck disable=SC2207
  TOK_LIST=($(_split_list "${PRECOMPILE_TOKEN_PADDINGS}"))
  COMMON_ARGS+=(--precompile-token-paddings "${TOK_LIST[@]}")
fi

if [[ "${SKIP_BASELINE}" != "1" ]]; then
  echo "[+] Baseline log: ${BASE_LOG}"
  (
    cd -- "${SGLJAX_ROOT}"
    "${SGLJAX_PY}" -m sgl_jax.bench_offline_throughput "${COMMON_ARGS[@]}"
  ) 2>&1 | tee "${BASE_LOG}"
fi

if [[ "${SKIP_DFLASH}" != "1" ]]; then
  echo "[+] DFLASH log: ${DFLASH_LOG}"
  (
    cd -- "${SGLJAX_ROOT}"
    "${SGLJAX_PY}" -m sgl_jax.bench_offline_throughput \
      "${COMMON_ARGS[@]}" \
      --speculative-algorithm DFLASH \
      --speculative-draft-model-path "${DRAFT_PATH}" \
      --speculative-num-draft-tokens "${BLOCK_SIZE}" \
      --speculative-dflash-block-size "${BLOCK_SIZE}" \
      --disable-radix-cache
  ) 2>&1 | tee "${DFLASH_LOG}"
fi

echo "[+] Done: ${RUN_NAME}"
