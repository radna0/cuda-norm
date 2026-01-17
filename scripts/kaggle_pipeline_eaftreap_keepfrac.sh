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
BATCH_SIZE="8"
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
      "bash" "-lc" "rg -n '\\[\\+\\] Wrote .*manifest_eaftreap_keepfrac\\.json' -S ${PRUNE_REMOTE_LOG} && echo DONE || echo PENDING" \
      | rg -n "^DONE$" >/dev/null 2>&1; then
    break
  fi
  echo "[*] still running... $(date +%Y-%m-%dT%H:%M:%S%z)"
  sleep "${INTERVAL_S}"
done

echo "[+] prune finished; reading manifest..."

echo "[*] Downloading pruning manifest via stdout (versa fetch is unreliable)..."
FETCH_DIR="${ROOT_DIR}/kaggle_fetch/prune_manifest_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${FETCH_DIR}"
LOCAL_MANIFEST="${FETCH_DIR}/manifest_eaftreap_keepfrac.json"

PYTHONPATH="${REPO_ROOT}/third_party/Versa" python -m versa run \
  --backend jupyter \
  --url "${REMOTE_JUPYTER_URL}" \
  ${REMOTE_JUPYTER_TOKEN:+--token "${REMOTE_JUPYTER_TOKEN}"} \
  --kernel-id "${KERNEL_ID}" \
  --cwd "/kaggle/working" \
  "python" "-c" "import base64, pathlib, sys; p=pathlib.Path('${MANIFEST_PATH}'); print('___BEGIN_B64___'); print(base64.b64encode(p.read_bytes()).decode('utf-8')); print('___END_B64___')" \
  | awk '/^___BEGIN_B64___$/{f=1;next} /^___END_B64___$/{f=0} f{print}' \
  | tr -d '\n' \
  | python -c "import base64,sys; sys.stdout.buffer.write(base64.b64decode(sys.stdin.buffer.read().strip() or b''))" \
  > "${LOCAL_MANIFEST}"

if [[ ! -s "${LOCAL_MANIFEST}" ]]; then
  echo "[err] failed to download manifest to ${LOCAL_MANIFEST}" >&2
  exit 3
fi

MODEL_LABEL="$(python - <<PY
import json
m=json.load(open('${LOCAL_MANIFEST}','r',encoding='utf-8'))
v=m.get('variants') or {}
print(sorted(v.keys())[0] if v else '')
PY
)"

OUT_DIR="$(python - <<PY
import json
m=json.load(open('${LOCAL_MANIFEST}','r',encoding='utf-8'))
v=m.get('variants') or {}
k=sorted(v.keys())[0] if v else ''
print(v.get(k,''))
PY
)"

if [[ -z "${MODEL_LABEL}" || -z "${OUT_DIR}" ]]; then
  echo "[err] Could not parse downloaded manifest ${LOCAL_MANIFEST}" >&2
  exit 4
fi

echo "[+] pruned label=${MODEL_LABEL}"
echo "[+] pruned out_dir=${OUT_DIR}"

echo "[*] Ensuring SGLang is installed + overlay applied on Kaggle kernel..."
export REMOTE_JUPYTER_KERNEL_ID="${KERNEL_ID}"
bash "${ROOT_DIR}/scripts/versa_install_sglang_dflash.sh" --kernel-id "${KERNEL_ID}" --no-detach

echo "[*] Starting EAFT collectors on Kaggle kernel..."
export REMOTE_JUPYTER_KERNEL_ID="${KERNEL_ID}"

_wait_for_wrote() {
  local remote_log="$1"
  echo "[*] waiting for ${remote_log} ..."
  while :; do
    if PYTHONPATH="${REPO_ROOT}/third_party/Versa" python -m versa run \
        --backend jupyter \
        --url "${REMOTE_JUPYTER_URL}" \
        ${REMOTE_JUPYTER_TOKEN:+--token "${REMOTE_JUPYTER_TOKEN}"} \
        --kernel-id "${KERNEL_ID}" \
        --cwd "/kaggle/working" \
        bash -lc "rg -n '\\[\\+\\] Wrote ' -S '${remote_log}' | tail -n 1 || true" \
        | rg -v '^\\[versa\\]' \
        | rg -q "\\[\\+\\] Wrote "; then
      break
    fi
    echo "[*] still running... $(date -Is)"
    sleep "${INTERVAL_S}"
  done
}

_extract_wrote_path() {
  local remote_log="$1"
  PYTHONPATH="${REPO_ROOT}/third_party/Versa" python -m versa run \
    --backend jupyter \
    --url "${REMOTE_JUPYTER_URL}" \
    ${REMOTE_JUPYTER_TOKEN:+--token "${REMOTE_JUPYTER_TOKEN}"} \
    --kernel-id "${KERNEL_ID}" \
    --cwd "/kaggle/working" \
    bash -lc "rg -n '\\[\\+\\] Wrote ' -S '${remote_log}' | tail -n 1 | sed -E 's/^.*\\[\\+\\] Wrote //' || true" \
    | rg -v '^\\[versa\\]' \
    | tail -n 1
}

BASE_START_OUT="$(
  bash "${ROOT_DIR}/scripts/versa_run_eaft_single_kaggle.sh" \
    --kernel-id "${KERNEL_ID}" \
    --model-id "openai/gpt-oss-20b" \
    --model-path "/kaggle/input/gpt-oss-20b/transformers/default/1" \
    --seq-lens-csv "${SEQ_LENS_CSV}" \
    --num-blocks "${NUM_BLOCKS}" --batch-size "${BATCH_SIZE}" \
    --sample-points "${SAMPLE_POINTS}" --top-k "${TOP_K}" \
    --progress-every-s 30 --max-new-tokens 1 \
    --skip-predownload
)"
echo "${BASE_START_OUT}"
BASE_REMOTE_LOG="$(printf '%s\n' "${BASE_START_OUT}" | rg '^\\s*remote_log=' | sed -E 's/^\\s*remote_log=//' | tail -n 1)"
if [[ -z "${BASE_REMOTE_LOG}" ]]; then
  echo "[err] Could not parse base remote_log" >&2
  exit 5
fi

_wait_for_wrote "${BASE_REMOTE_LOG}"

PRUNED_START_OUT="$(
  bash "${ROOT_DIR}/scripts/versa_run_eaft_single_kaggle.sh" \
    --kernel-id "${KERNEL_ID}" \
    --model-id "${MODEL_LABEL}" \
    --model-path "${OUT_DIR}" \
    --seq-lens-csv "${SEQ_LENS_CSV}" \
    --num-blocks "${NUM_BLOCKS}" --batch-size "${BATCH_SIZE}" \
    --sample-points "${SAMPLE_POINTS}" --top-k "${TOP_K}" \
    --progress-every-s 30 --max-new-tokens 1 \
    --skip-predownload
)"
echo "${PRUNED_START_OUT}"
PRUNED_REMOTE_LOG="$(printf '%s\n' "${PRUNED_START_OUT}" | rg '^\\s*remote_log=' | sed -E 's/^\\s*remote_log=//' | tail -n 1)"
if [[ -z "${PRUNED_REMOTE_LOG}" ]]; then
  echo "[err] Could not parse pruned remote_log" >&2
  exit 5
fi

_wait_for_wrote "${PRUNED_REMOTE_LOG}"

echo "[*] Downloading EAFT JSONs via stdout..."
FETCH_DIR="${ROOT_DIR}/kaggle_fetch/eaft_models_${MODEL_LABEL}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${FETCH_DIR}"

BASE_REMOTE="$(_extract_wrote_path "${BASE_REMOTE_LOG}")"
PRUNED_REMOTE="$(_extract_wrote_path "${PRUNED_REMOTE_LOG}")"

if [[ -z "${BASE_REMOTE}" || -z "${PRUNED_REMOTE}" ]]; then
  echo "[err] Could not locate EAFT JSONs on Kaggle." >&2
  echo "      base_remote=${BASE_REMOTE:-<missing>}" >&2
  echo "      pruned_remote=${PRUNED_REMOTE:-<missing>}" >&2
  exit 5
fi

BASE_REMOTE_DIR="$(dirname "${BASE_REMOTE}")"
PRUNED_REMOTE_DIR="$(dirname "${PRUNED_REMOTE}")"

echo "[*] Downloading EAFT outputs from Kaggle (download_dir)..."
mkdir -p "${FETCH_DIR}/base" "${FETCH_DIR}/pruned"
PYTHONPATH="${REPO_ROOT}/third_party/Versa" python -m versa run \
  --backend jupyter --url "${REMOTE_JUPYTER_URL}" ${REMOTE_JUPYTER_TOKEN:+--token "${REMOTE_JUPYTER_TOKEN}"} \
  --kernel-id "${KERNEL_ID}" --cwd "/kaggle/working" \
  --download-remote-dir "${BASE_REMOTE_DIR}" --download-local-dir "${FETCH_DIR}/base" \
  python -c "print('download base eaft dir')"
PYTHONPATH="${REPO_ROOT}/third_party/Versa" python -m versa run \
  --backend jupyter --url "${REMOTE_JUPYTER_URL}" ${REMOTE_JUPYTER_TOKEN:+--token "${REMOTE_JUPYTER_TOKEN}"} \
  --kernel-id "${KERNEL_ID}" --cwd "/kaggle/working" \
  --download-remote-dir "${PRUNED_REMOTE_DIR}" --download-local-dir "${FETCH_DIR}/pruned" \
  python -c "print('download pruned eaft dir')"

BASE_JSON="${FETCH_DIR}/base/$(basename "${BASE_REMOTE}")"
PRUNED_JSON="${FETCH_DIR}/pruned/$(basename "${PRUNED_REMOTE}")"

if [[ ! -s "${BASE_JSON}" || ! -s "${PRUNED_JSON}" ]]; then
  echo "[err] Failed to download EAFT JSONs via download_dir." >&2
  echo "      base_json=${BASE_JSON}" >&2
  echo "      pruned_json=${PRUNED_JSON}" >&2
  exit 6
fi

OUT_MD="${ROOT_DIR}/reports/eaftreap_keepfrac_parity_${MODEL_LABEL}.md"
python "${ROOT_DIR}/scripts/summarize_eaft_pair.py" \
  --left-json "${BASE_JSON}" \
  --right-json "${PRUNED_JSON}" \
  --out-md "${OUT_MD}" \
  --gates-json "${ROOT_DIR}/pruning/near_lossless_gates.json"

echo "[+] Wrote ${OUT_MD}"
