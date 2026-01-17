#!/usr/bin/env bash
set -euo pipefail

# End-to-end Kaggle/VERSA pipeline:
# 1) Wait for the running prune build to finish (by watching its remote log).
# 2) Read pruned model out_dir from remote manifest.
# 3) Run EAFT single-model collection for base vs pruned (seq=1024,2048).
# 4) Download EAFT artifacts to local disk and write a local parity summary MD.
#
# This is designed to be run from the repo root (or anywhere) and can be nohup'd.
#
# Required env:
#   REMOTE_JUPYTER_URL="https://.../proxy"
#
# Optional env:
#   REMOTE_JUPYTER_TOKEN=""
#
# Usage:
#   bash scripts/kaggle_pipeline_eaftreap75.sh \
#     --prune-kernel-id <kernel> \
#     --prune-remote-log logs/<log>.log \
#     --manifest-path artifacts/20b_pruned_models_eaftreap_budgeted/manifest.json
#
# Outputs:
#   - Local: harmony/cuda-norm/kaggle_fetch/eaft_models_*/...
#   - Local: harmony/cuda-norm/reports/eaftreap_budgeted_keepfrac75_parity_summary.md

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"

if [[ -z "${REMOTE_JUPYTER_URL:-}" ]]; then
  echo "[err] REMOTE_JUPYTER_URL is not set" >&2
  exit 2
fi

PRUNE_KERNEL_ID=""
PRUNE_REMOTE_LOG=""
MANIFEST_PATH="artifacts/20b_pruned_models_eaftreap_budgeted/manifest.json"
INTERVAL_S="180"

SEQ_LENS_CSV="1024,2048"
NUM_BLOCKS="256"
BATCH_SIZE="1"
SAMPLE_POINTS="200000"
TOP_K="4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prune-kernel-id) PRUNE_KERNEL_ID="$2"; shift 2;;
    --prune-remote-log) PRUNE_REMOTE_LOG="$2"; shift 2;;
    --manifest-path) MANIFEST_PATH="$2"; shift 2;;
    --interval-s) INTERVAL_S="$2"; shift 2;;
    --seq-lens-csv) SEQ_LENS_CSV="$2"; shift 2;;
    --num-blocks) NUM_BLOCKS="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --sample-points) SAMPLE_POINTS="$2"; shift 2;;
    --top-k) TOP_K="$2"; shift 2;;
    -h|--help)
      sed -n '1,220p' "$0"
      exit 0
      ;;
    *)
      echo "[err] Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${PRUNE_KERNEL_ID}" || -z "${PRUNE_REMOTE_LOG}" ]]; then
  echo "[err] --prune-kernel-id and --prune-remote-log are required" >&2
  exit 2
fi

mkdir -p "${ROOT_DIR}/kaggle_fetch" "${ROOT_DIR}/reports"

echo "[*] Waiting for prune to finish..."
echo "    prune_kernel_id=${PRUNE_KERNEL_ID}"
echo "    prune_remote_log=${PRUNE_REMOTE_LOG}"

while :; do
  # Check completion marker.
  if PYTHONPATH="${REPO_ROOT}/third_party/Versa" python -m versa run \
      --backend jupyter \
      --url "${REMOTE_JUPYTER_URL}" \
      ${REMOTE_JUPYTER_TOKEN:+--token "${REMOTE_JUPYTER_TOKEN}"} \
      --kernel-id "${PRUNE_KERNEL_ID}" \
      --cwd "/kaggle/working" \
      "bash -lc \"rg -n '\\[\\+\\] structural_prune done|\\[\\+\\] Wrote reports/20b_structural_prune_build_eaftreap_budgeted\\.md' -S ${PRUNE_REMOTE_LOG} && echo DONE || echo PENDING\"" \
      | rg -n "^DONE$" >/dev/null 2>&1; then
    break
  fi
  echo "[*] prune still running... $(date +%Y-%m-%dT%H:%M:%S%z)"
  sleep "${INTERVAL_S}"
done

echo "[+] prune finished; reading manifest..."

OUT_DIR="$(
  PYTHONPATH="${REPO_ROOT}/third_party/Versa" python -m versa run \
    --backend jupyter \
    --url "${REMOTE_JUPYTER_URL}" \
    ${REMOTE_JUPYTER_TOKEN:+--token "${REMOTE_JUPYTER_TOKEN}"} \
    --kernel-id "${PRUNE_KERNEL_ID}" \
    --cwd "/kaggle/working" \
    "python - <<'PY'\nimport json\np='${MANIFEST_PATH}'\nwith open(p,'r',encoding='utf-8') as f:\n  m=json.load(f)\nprint(m.get('out_dir',''))\nPY" \
    | tail -n 1
)"

if [[ -z "${OUT_DIR}" ]]; then
  echo "[err] Could not read out_dir from manifest at ${MANIFEST_PATH}" >&2
  exit 3
fi

echo "[+] pruned out_dir=${OUT_DIR}"

run_one() {
  local name="$1"
  local model_id="$2"
  local model_path="$3"
  local local_dir="$4"

  echo "[*] EAFT collect: ${name} model_id=${model_id} model_path=${model_path}"
  PYTHONPATH="${REPO_ROOT}/third_party/Versa" python -m versa run \
    --backend jupyter \
    --url "${REMOTE_JUPYTER_URL}" \
    ${REMOTE_JUPYTER_TOKEN:+--token "${REMOTE_JUPYTER_TOKEN}"} \
    --cwd "/kaggle/working" \
    --log-path "logs/eaft_single_${name}_$(date +%Y%m%d_%H%M%S).log" \
    --bootstrap-cmd "mkdir -p logs" \
    --bootstrap-cmd "mkdir -p /kaggle/working/eaft_cache" \
    --env-file "${ROOT_DIR}/.env" \
    --env "GPU_TYPE=H100:1" \
    --env "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1" \
    --download-remote-dir "/kaggle/working/artifacts/eaft_models" \
    --download-local-dir "${local_dir}" \
    "${ROOT_DIR}/modal/collect_calib_packs_eaft_single.py::main" -- \
      --model-id "${model_id}" \
      --model-path "${model_path}" \
      --seq-lens-csv "${SEQ_LENS_CSV}" \
      --num-blocks "${NUM_BLOCKS}" \
      --batch-size "${BATCH_SIZE}" \
      --sample-points "${SAMPLE_POINTS}" \
      --top-k "${TOP_K}" \
      --entropy-topk 20 \
      --cc-quantile 0.15
}

BASE_LOCAL="${ROOT_DIR}/kaggle_fetch/eaft_models_base_$(date +%Y%m%d_%H%M%S)"
PRUNED_LOCAL="${ROOT_DIR}/kaggle_fetch/eaft_models_pruned_$(date +%Y%m%d_%H%M%S)"

run_one "base20b" "openai/gpt-oss-20b" "/kaggle/input/gpt-oss-20b/transformers/default/1" "${BASE_LOCAL}"
run_one "pruned075" "eaftreap_budgeted_keepfrac075" "${OUT_DIR}" "${PRUNED_LOCAL}"

echo "[+] Downloaded EAFT artifacts:"
echo "    base:   ${BASE_LOCAL}"
echo "    pruned: ${PRUNED_LOCAL}"

# Find the newest JSON per run.
BASE_JSON="$(ls -t ${BASE_LOCAL}/eaft_models/*/openai__gpt-oss-20b.json 2>/dev/null | head -n 1 || true)"
PRUNED_JSON="$(ls -t ${PRUNED_LOCAL}/eaft_models/*/eaftreap_budgeted_keepfrac075.json 2>/dev/null | head -n 1 || true)"

if [[ -z "${BASE_JSON}" || -z "${PRUNED_JSON}" ]]; then
  echo "[err] Could not find expected EAFT JSONs after download." >&2
  echo "      base_json=${BASE_JSON:-<missing>}" >&2
  echo "      pruned_json=${PRUNED_JSON:-<missing>}" >&2
  exit 4
fi

OUT_MD="${ROOT_DIR}/reports/eaftreap_budgeted_keepfrac75_parity_summary.md"
python "${ROOT_DIR}/scripts/summarize_eaft_pair.py" \
  --left-json "${BASE_JSON}" \
  --right-json "${PRUNED_JSON}" \
  --out-md "${OUT_MD}" \
  --gates-json "${ROOT_DIR}/pruning/near_lossless_gates.json"

echo "[+] Wrote parity summary: ${OUT_MD}"

