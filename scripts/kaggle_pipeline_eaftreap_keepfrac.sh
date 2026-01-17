#!/usr/bin/env bash
set -euo pipefail

# General Kaggle/VERSA pipeline:
# - Wait for a pruning-track build to finish (remote log).
# - Read pruned variant + out_dir from the remote manifest.
# - Run EAFT single collectors (base vs pruned) on the same Kaggle kernel.
# - Fetch EAFT JSONs back locally and write a local parity summary MD.
#
# Required env:
#   REMOTE_JUPYTER_URL="https://.../proxy"
# Optional:
#   REMOTE_JUPYTER_TOKEN=""
#
# Usage (example):
#   bash harmony/cuda-norm/scripts/kaggle_pipeline_eaftreap_keepfrac.sh \
#     --kernel-id <kernel_id> \
#     --prune-remote-log logs/build_pruned_20b_eaftreap_keepfrac_YYYYmmdd_HHMMSS.log \
#     --manifest-path artifacts/20b_pruned_models_eaftreap_keepfrac/manifest_eaftreap_keepfrac.json \
#     --seq-lens-csv 1024,2048 --num-blocks 128 --sample-points 100000

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"

if [[ -z "${REMOTE_JUPYTER_URL:-}" ]]; then
  echo "[err] REMOTE_JUPYTER_URL is not set" >&2
  exit 2
fi

KERNEL_ID=""
PRUNE_REMOTE_LOG=""
MANIFEST_PATH="artifacts/20b_pruned_models_eaftreap_keepfrac/manifest_eaftreap_keepfrac.json"
INTERVAL_S="120"

SEQ_LENS_CSV="1024,2048"
NUM_BLOCKS="128"
BATCH_SIZE="1"
SAMPLE_POINTS="100000"
TOP_K="4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel-id) KERNEL_ID="$2"; shift 2;;
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

if [[ -z "${KERNEL_ID}" || -z "${PRUNE_REMOTE_LOG}" ]]; then
  echo "[err] --kernel-id and --prune-remote-log are required" >&2
  exit 2
fi

mkdir -p "${ROOT_DIR}/kaggle_fetch" "${ROOT_DIR}/reports"

echo "[*] Waiting for prune to finish..."
while :; do
  if PYTHONPATH="${REPO_ROOT}/third_party/Versa" python -m versa run \
      --backend jupyter \
      --url "${REMOTE_JUPYTER_URL}" \
      ${REMOTE_JUPYTER_TOKEN:+--token "${REMOTE_JUPYTER_TOKEN}"} \
      --kernel-id "${KERNEL_ID}" \
      --cwd "/kaggle/working" \
      "bash" "-lc" "rg -n '\\[\\+\\] Wrote ${MANIFEST_PATH//\//\\/}' -S ${PRUNE_REMOTE_LOG} && echo DONE || echo PENDING" \
      | rg -n "^DONE$" >/dev/null 2>&1; then
    break
  fi
  echo "[*] still running... $(date +%Y-%m-%dT%H:%M:%S%z)"
  sleep "${INTERVAL_S}"
done

echo "[+] prune finished; reading manifest..."

MODEL_LABEL="$(
  PYTHONPATH="${REPO_ROOT}/third_party/Versa" python -m versa run \
    --backend jupyter --url "${REMOTE_JUPYTER_URL}" ${REMOTE_JUPYTER_TOKEN:+--token "${REMOTE_JUPYTER_TOKEN}"} \
    --kernel-id "${KERNEL_ID}" --cwd "/kaggle/working" \
    "python" "- <<'PY'\nimport json\nm=json.load(open('${MANIFEST_PATH}','r',encoding='utf-8'))\nprint(sorted((m.get('variants') or {}).keys())[0])\nPY" | tail -n 1
)"

OUT_DIR="$(
  PYTHONPATH="${REPO_ROOT}/third_party/Versa" python -m versa run \
    --backend jupyter --url "${REMOTE_JUPYTER_URL}" ${REMOTE_JUPYTER_TOKEN:+--token "${REMOTE_JUPYTER_TOKEN}"} \
    --kernel-id "${KERNEL_ID}" --cwd "/kaggle/working" \
    "python" "- <<'PY'\nimport json\nm=json.load(open('${MANIFEST_PATH}','r',encoding='utf-8'))\nv=m.get('variants') or {}\nprint(v.get(sorted(v.keys())[0],''))\nPY" | tail -n 1
)"

if [[ -z "${MODEL_LABEL}" || -z "${OUT_DIR}" ]]; then
  echo "[err] Could not read pruned variant from manifest ${MANIFEST_PATH}" >&2
  exit 3
fi

echo "[+] pruned label=${MODEL_LABEL}"
echo "[+] pruned out_dir=${OUT_DIR}"

echo "[*] Starting EAFT collectors on Kaggle kernel..."
export REMOTE_JUPYTER_KERNEL_ID="${KERNEL_ID}"

bash "${ROOT_DIR}/scripts/versa_run_eaft_single_kaggle.sh" \
  --kernel-id "${KERNEL_ID}" \
  --model-id "openai/gpt-oss-20b" \
  --model-path "/kaggle/input/gpt-oss-20b/transformers/default/1" \
  --seq-lens-csv "${SEQ_LENS_CSV}" \
  --num-blocks "${NUM_BLOCKS}" --batch-size "${BATCH_SIZE}" \
  --sample-points "${SAMPLE_POINTS}" --top-k "${TOP_K}" \
  --skip-predownload \
  --no-detach

bash "${ROOT_DIR}/scripts/versa_run_eaft_single_kaggle.sh" \
  --kernel-id "${KERNEL_ID}" \
  --model-id "${MODEL_LABEL}" \
  --model-path "${OUT_DIR}" \
  --seq-lens-csv "${SEQ_LENS_CSV}" \
  --num-blocks "${NUM_BLOCKS}" --batch-size "${BATCH_SIZE}" \
  --sample-points "${SAMPLE_POINTS}" --top-k "${TOP_K}" \
  --skip-predownload \
  --no-detach

echo "[*] Fetching EAFT artifacts from Kaggle..."
FETCH_DIR="${ROOT_DIR}/kaggle_fetch/eaft_models_${MODEL_LABEL}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${FETCH_DIR}"

PYTHONPATH="${REPO_ROOT}/third_party/Versa" python -m versa fetch \
  --url "${REMOTE_JUPYTER_URL}" \
  ${REMOTE_JUPYTER_TOKEN:+--token "${REMOTE_JUPYTER_TOKEN}"} \
  --kernel-id "${KERNEL_ID}" \
  --remote-dir "/kaggle/working/cuda-norm/artifacts/eaft_models" \
  --local-dir "${FETCH_DIR}/eaft_models"

BASE_JSON="$(ls -t ${FETCH_DIR}/eaft_models/*/openai_gpt-oss-20b.json 2>/dev/null | head -n 1 || true)"
PRUNED_JSON="$(ls -t ${FETCH_DIR}/eaft_models/*/${MODEL_LABEL}.json 2>/dev/null | head -n 1 || true)"
if [[ -z "${BASE_JSON}" || -z "${PRUNED_JSON}" ]]; then
  echo "[err] Could not find fetched EAFT JSONs." >&2
  echo "      base=${BASE_JSON:-<missing>}" >&2
  echo "      pruned=${PRUNED_JSON:-<missing>}" >&2
  exit 4
fi

OUT_MD="${ROOT_DIR}/reports/eaftreap_keepfrac_parity_${MODEL_LABEL}.md"
python "${ROOT_DIR}/scripts/summarize_eaft_pair.py" \
  --left-json "${BASE_JSON}" \
  --right-json "${PRUNED_JSON}" \
  --out-md "${OUT_MD}" \
  --gates-json "${ROOT_DIR}/pruning/near_lossless_gates.json"

echo "[+] Wrote ${OUT_MD}"
