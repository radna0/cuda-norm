#!/usr/bin/env bash
set -euo pipefail

# Kaggle/VERSA: Noise-floor measurement for the extreme long-context regime.
#
# Purpose:
# - quantify the natural wobble (ΔPPL/ΔCC/JS2D) when comparing the *same* model
#   to itself under the 65K/131K evaluation matrix.
#
# This lets us interpret 0.75 EAFT-REAP deltas at long context as “real” vs noise.
#
# Usage (always nohup):
#   bash harmony/cuda-norm/scripts/kaggle_base_vs_base_noise_ctx65k_131k.sh --kernel-id <id>

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"

KERNEL_ID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel-id) KERNEL_ID="$2"; shift 2;;
    -h|--help)
      sed -n '1,160p' "$0"
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
TMP_DIR="${REPO_ROOT}/harmony/cuda-norm/reports/_tmp_runs/base_vs_base_noise_ctx65k_131k_${RUN_TAG}"
mkdir -p "${TMP_DIR}"

run_eaft() {
  local model_id="$1"
  local model_path="$2"
  local seq_len="$3"
  local num_blocks="$4"
  local batch_size="$5"
  local sample_points="$6"
  bash "${ROOT_DIR}/scripts/versa_run_eaft_single_kaggle.sh" \
    --kernel-id "${KERNEL_ID}" \
    --model-id "${model_id}" \
    ${model_path:+--model-path "${model_path}"} \
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

  echo "[*] EAFT: base(A)"
  OUT_L="$(run_eaft 'openai/gpt-oss-20b' '/kaggle/input/gpt-oss-20b/transformers/default/1' "${seq_len}" "${num_blocks}" "${batch_size}" 200000)"
  echo "${OUT_L}"
  LOG_L="$(extract_remote_log "${OUT_L}")"
  echo "[*] left_remote_log=${LOG_L}"

  bash "${ROOT_DIR}/scripts/kaggle_watch_file.sh" \
    --remote-path "${LOG_L}" \
    --pattern "\\[\\+\\] Wrote /kaggle/working/artifacts/eaft_models/" \
    --fail-pattern "Fatal Python error: Segmentation fault|Segmentation fault \\(core dumped\\)|torch\\.OutOfMemoryError|CUDA out of memory|OutOfMemoryError|Traceback \\(most recent call last\\)|Failed to import modal/collect_calib_packs_eaft_single\\.py|\\[versa\\] exitcode=[1-9]" \
    --interval-s 120 \
    --tail-n 120

  JSON_L_REL="$(extract_written_json_relpath "${LOG_L}")"
  JSON_L_LOCAL="$(download_remote_file_rel "${JSON_L_REL}")"
  echo "[+] left_json_local=${JSON_L_LOCAL}"

  echo "[*] EAFT: base(B)"
  OUT_R="$(run_eaft 'openai/gpt-oss-20b' '/kaggle/input/gpt-oss-20b/transformers/default/1' "${seq_len}" "${num_blocks}" "${batch_size}" 200000)"
  echo "${OUT_R}"
  LOG_R="$(extract_remote_log "${OUT_R}")"
  echo "[*] right_remote_log=${LOG_R}"

  bash "${ROOT_DIR}/scripts/kaggle_watch_file.sh" \
    --remote-path "${LOG_R}" \
    --pattern "\\[\\+\\] Wrote /kaggle/working/artifacts/eaft_models/" \
    --fail-pattern "Fatal Python error: Segmentation fault|Segmentation fault \\(core dumped\\)|torch\\.OutOfMemoryError|CUDA out of memory|OutOfMemoryError|Traceback \\(most recent call last\\)|Failed to import modal/collect_calib_packs_eaft_single\\.py|\\[versa\\] exitcode=[1-9]" \
    --interval-s 120 \
    --tail-n 120

  JSON_R_REL="$(extract_written_json_relpath "${LOG_R}")"
  JSON_R_LOCAL="$(download_remote_file_rel "${JSON_R_REL}")"
  echo "[+] right_json_local=${JSON_R_LOCAL}"

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

FINAL_MD="${REPO_ROOT}/harmony/cuda-norm/reports/eaft_base_vs_base_noise_ctx65k_131k.md"
{
  echo "# EAFT noise floor (ctx=65K/131K tokenbudget ~1M) — base vs base"
  echo
  echo "- Model: `openai/gpt-oss-20b` (same checkpoint both sides)"
  echo "- top_k: 4"
  echo
  echo "## Matrix results"
  echo
  for f in \"${TMP_DIR}\"/*.md; do
    echo \"---\"
    echo
    cat \"${f}\"
    echo
  done
} > \"${FINAL_MD}\"

echo \"[+] wrote ${FINAL_MD}\"

