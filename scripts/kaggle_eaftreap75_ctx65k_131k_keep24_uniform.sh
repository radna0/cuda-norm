#!/usr/bin/env bash
set -euo pipefail

# Kaggle/VERSA: "production-style" long-context validation for keep_n=24/32
# (keep_frac=0.75) EAFT-REAP, at fixed top_k=4 (unchanged).
#
# Runs EAFT parity on:
# - seq=65536, blocks=16, bs=1   (~1,048,576 tokens/pack)
# - seq=131072, blocks=8, bs=1  (~1,048,576 tokens/pack)
#
# Always run under nohup and monitor the local log.
#
# Usage:
#   bash harmony/cuda-norm/scripts/kaggle_eaftreap75_ctx65k_131k_keep24_uniform.sh --kernel-id <id>

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"

KERNEL_ID=""
PRUNED_MODEL_PATH_OVERRIDE=""
PRUNED_KAGGLEHUB_DATASET=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel-id) KERNEL_ID="$2"; shift 2;;
    --pruned-model-path) PRUNED_MODEL_PATH_OVERRIDE="$2"; shift 2;;
    --pruned-kagglehub-dataset) PRUNED_KAGGLEHUB_DATASET="$2"; shift 2;;
    -h|--help)
      sed -n '1,200p' "$0"
      exit 0
      ;;
    *)
      echo "[err] Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

set -a
source "${ROOT_DIR}/.env"
set +a
if [[ -z "${KAGGLE_URL:-}" ]]; then
  echo "[err] KAGGLE_URL is not set (check ${ROOT_DIR}/.env)" >&2
  exit 2
fi

export REMOTE_JUPYTER_URL="${KAGGLE_URL}"
if [[ -z "${KERNEL_ID}" ]]; then
  KERNEL_ID="$(bash "${ROOT_DIR}/scripts/kaggle_get_active_kernel_id.sh")"
fi
export REMOTE_JUPYTER_KERNEL_ID="${KERNEL_ID}"

echo "[*] kernel_id=${KERNEL_ID}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
TMP_DIR="${REPO_ROOT}/harmony/cuda-norm/reports/_tmp_runs/eaftreap75_ctx65k_131k_keep24_uniform_${RUN_TAG}"
mkdir -p "${TMP_DIR}"

MANIFEST_LOCAL="${REPO_ROOT}/harmony/cuda-norm/artifacts/20b_pruned_models_eaftreap_keepfrac/manifest_eaftreap_keepfrac.json"
if [[ ! -f "${MANIFEST_LOCAL}" ]]; then
  echo "[err] missing manifest: ${MANIFEST_LOCAL}" >&2
  exit 2
fi

PRUNED_VARIANT_NAME="calib_union_keep24of32_k75_eaftreap"
PRUNED_MODEL_PATH="$(
  python - <<PY
import json
from pathlib import Path
m=json.loads(Path("${MANIFEST_LOCAL}").read_text())
print(m["variants"]["calib_union_keep24of32_k75_eaftreap"])
PY
)"

if [[ -n "${PRUNED_MODEL_PATH_OVERRIDE}" ]]; then
  PRUNED_MODEL_PATH="${PRUNED_MODEL_PATH_OVERRIDE}"
fi

echo "[+] pruned_variant_name=${PRUNED_VARIANT_NAME}"
echo "[+] pruned_model_path=${PRUNED_MODEL_PATH}"

remote_path_exists() {
  local path="$1"
  local rel="${path#/}"
  rel="${rel#/kaggle/working/}"
  rel="${rel#kaggle/working/}"
  local base="${KAGGLE_URL%/}"
  local url="${base}/api/contents/${rel}"
  local attempt code
  for attempt in 1 2 3 4 5; do
    code="$(
      curl -sS -o /dev/null -w '%{http_code}' \
        --retry 4 --retry-all-errors --retry-delay 1 \
        --connect-timeout 5 --max-time 20 \
        "${url}" || true
    )"
    if [[ "${code}" == "200" ]]; then
      return 0
    fi
    if [[ "${code}" == "404" ]]; then
      return 1
    fi
    sleep 1
  done
  return 1
}

if [[ -z "${PRUNED_MODEL_PATH_OVERRIDE}" ]] && ! remote_path_exists "${PRUNED_MODEL_PATH}"; then
  echo "[warn] pruned checkpoint missing on this kernel; rebuilding keep24 first..." >&2
  bash "${ROOT_DIR}/scripts/kaggle_build_keep24_only.sh"
  if ! remote_path_exists "${PRUNED_MODEL_PATH}"; then
    echo "[err] keep24 build did not produce expected path: ${PRUNED_MODEL_PATH}" >&2
    exit 2
  fi
fi

if [[ -n "${PRUNED_MODEL_PATH_OVERRIDE}" ]] && ! remote_path_exists "${PRUNED_MODEL_PATH}"; then
  if [[ -n "${PRUNED_KAGGLEHUB_DATASET}" ]]; then
    echo "[warn] pruned model path missing on kernel; will download via kagglehub dataset=${PRUNED_KAGGLEHUB_DATASET}" >&2
    PRUNED_MODEL_PATH=""
  else
    echo "[err] pruned model path not found on this Kaggle kernel: ${PRUNED_MODEL_PATH}" >&2
    echo "[err] Attach the dataset to this notebook and restart the kernel, then rerun." >&2
    echo "[err] Dataset (upload target): nyannyanmira/gptoss20b-keep24of32-k75-eaftreap" >&2
    echo "[err] Or pass: --pruned-kagglehub-dataset nyannyanmira/gptoss20b-keep24of32-k75-eaftreap" >&2
    exit 2
  fi
fi

run_eaft() {
  local model_id="$1"
  local model_path="$2"
  local kagglehub_dataset="$3"
  local seq_len="$4"
  local num_blocks="$5"
  local batch_size="$6"
  local sample_points="$7"
  bash "${ROOT_DIR}/scripts/versa_run_eaft_single_kaggle.sh" \
    --kernel-id "${KERNEL_ID}" \
    --model-id "${model_id}" \
    ${model_path:+--model-path "${model_path}"} \
    ${kagglehub_dataset:+--kagglehub-dataset "${kagglehub_dataset}"} \
    --seq-lens-csv "${seq_len}" \
    --num-blocks "${num_blocks}" \
    --batch-size "${batch_size}" \
    --sample-points "${sample_points}" \
    --top-k 4 --entropy-topk 20 --cc-quantile 0.15 \
    --progress-every-s 120 \
    --skip-predownload
}

extract_remote_log() {
  local text="$1"
  echo "${text}" | rg -n "remote_log=" | tail -n 1 | sed -E 's/^.*remote_log=//'
}

extract_written_json_relpath() {
  local remote_log="$1"
  local tmp="${REPO_ROOT}/harmony/cuda-norm/unsloth_logs/remote_${remote_log//\\//_}"
  mkdir -p "$(dirname "${tmp}")"
  bash "${ROOT_DIR}/scripts/kaggle_download_file.sh" --remote-path "${remote_log}" --out "${tmp}"
  python - <<PY
import re
from pathlib import Path
txt=Path("${tmp}").read_text(encoding="utf-8", errors="ignore")
for line in txt.splitlines()[::-1]:
    if "Wrote " in line and line.strip().endswith(".json"):
        m=re.search(r"Wrote\\s+(/kaggle/working/[^\\s]+\\.json)\\s*$", line)
        if m:
            p=m.group(1)
            print(p.replace('/kaggle/working/','',1))
            raise SystemExit(0)
raise SystemExit("could not find JSON write path in remote log")
PY
}

download_remote_file_rel() {
  local remote_rel="$1"
  local out="${REPO_ROOT}/harmony/cuda-norm/${remote_rel}"
  mkdir -p "$(dirname "${out}")"
  bash "${ROOT_DIR}/scripts/kaggle_download_file.sh" --remote-path "${remote_rel}" --out "${out}"
  echo "${out}"
}

run_pair_one() {
  local name="$1"
  local seq_len="$2"
  local num_blocks="$3"
  local batch_size="$4"

  echo
  echo "============================================================"
  echo "[*] matrix=${name} seq=${seq_len} blocks=${num_blocks} bs=${batch_size}"
  echo "============================================================"

  echo "[*] EAFT: base"
  OUT_L="$(run_eaft 'openai/gpt-oss-20b' '/kaggle/input/gpt-oss-20b/transformers/default/1' '' "${seq_len}" "${num_blocks}" "${batch_size}" 200000)"
  echo "${OUT_L}"
  LOG_L="$(extract_remote_log "${OUT_L}")"
  echo "[*] base_remote_log=${LOG_L}"

  bash "${ROOT_DIR}/scripts/kaggle_watch_file.sh" \
    --remote-path "${LOG_L}" \
    --pattern "\\[\\+\\] Wrote /kaggle/working/artifacts/eaft_models/" \
    --fail-pattern "Fatal Python error: Segmentation fault|Segmentation fault \\(core dumped\\)|torch\\.OutOfMemoryError|CUDA out of memory|OutOfMemoryError|Traceback \\(most recent call last\\)|Failed to import modal/collect_calib_packs_eaft_single\\.py|\\[versa\\] exitcode=[1-9]" \
    --interval-s 120 \
    --tail-n 120

  JSON_L_REL="$(extract_written_json_relpath "${LOG_L}")"
  JSON_L_LOCAL="$(download_remote_file_rel "${JSON_L_REL}")"
  echo "[+] base_json_local=${JSON_L_LOCAL}"

  echo "[*] EAFT: pruned (keep24)"
  OUT_R="$(run_eaft "${PRUNED_VARIANT_NAME}" "${PRUNED_MODEL_PATH}" "${PRUNED_KAGGLEHUB_DATASET}" "${seq_len}" "${num_blocks}" "${batch_size}" 200000)"
  echo "${OUT_R}"
  LOG_R="$(extract_remote_log "${OUT_R}")"
  echo "[*] pruned_remote_log=${LOG_R}"

  bash "${ROOT_DIR}/scripts/kaggle_watch_file.sh" \
    --remote-path "${LOG_R}" \
    --pattern "\\[\\+\\] Wrote /kaggle/working/artifacts/eaft_models/" \
    --fail-pattern "Fatal Python error: Segmentation fault|Segmentation fault \\(core dumped\\)|torch\\.OutOfMemoryError|CUDA out of memory|OutOfMemoryError|Traceback \\(most recent call last\\)|Failed to import modal/collect_calib_packs_eaft_single\\.py|\\[versa\\] exitcode=[1-9]" \
    --interval-s 120 \
    --tail-n 120

  JSON_R_REL="$(extract_written_json_relpath "${LOG_R}")"
  JSON_R_LOCAL="$(download_remote_file_rel "${JSON_R_REL}")"
  echo "[+] pruned_json_local=${JSON_R_LOCAL}"

  local out_md="${TMP_DIR}/${name}.md"
  python "${REPO_ROOT}/harmony/cuda-norm/scripts/summarize_eaft_pair.py" \
    --left-json "${JSON_L_LOCAL}" \
    --right-json "${JSON_R_LOCAL}" \
    --out-md "${out_md}"
  echo "[+] wrote ${out_md}"
}

mkdir -p "${REPO_ROOT}/harmony/cuda-norm/reports"

run_pair_one "ctx65536_blocks16_bs1" 65536 16 1
run_pair_one "ctx131072_blocks8_bs1" 131072 8 1

FINAL_MD="${REPO_ROOT}/harmony/cuda-norm/reports/eaftreap75_ctx65k_131k.md"
{
  echo "# EAFT parity (ctx=65K/131K tokenbudget ~1M) — keep24/32 EAFT-REAP"
  echo
  echo "- Base: \`openai/gpt-oss-20b\`"
  echo "- Pruned: \`${PRUNED_VARIANT_NAME}\`"
  echo "- Pruned path: \`${PRUNED_MODEL_PATH}\`"
  echo "- top_k: 4 (unchanged)"
  echo
  echo "## Matrix results"
  echo
  for f in "${TMP_DIR}"/*.md; do
    echo "---"
    echo
    cat "${f}"
    echo
  done
} > "${FINAL_MD}"

echo "[+] wrote ${FINAL_MD}"
