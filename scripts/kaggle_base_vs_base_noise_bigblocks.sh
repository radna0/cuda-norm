#!/usr/bin/env bash
set -euo pipefail

# Kaggle/VERSA: run EAFT bigblocks twice on the same base model and summarize
# the deltas as a "noise floor" (what statistical wobble looks like).
#
# Always run under nohup and monitor the local log.
#
# Usage:
#   bash harmony/cuda-norm/scripts/kaggle_base_vs_base_noise_bigblocks.sh --kernel-id <id>

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"

KERNEL_ID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel-id) KERNEL_ID="$2"; shift 2;;
    -h|--help)
      sed -n '1,120p' "$0"
      exit 0
      ;;
    *)
      echo "[err] Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done
if [[ -z "${KERNEL_ID}" ]]; then
  echo "[err] --kernel-id is required" >&2
  exit 2
fi

set -a
source "${ROOT_DIR}/.env"
set +a
if [[ -z "${KAGGLE_URL:-}" ]]; then
  echo "[err] KAGGLE_URL is not set (check ${ROOT_DIR}/.env)" >&2
  exit 2
fi
export REMOTE_JUPYTER_URL="${KAGGLE_URL}"

echo "[*] kernel_id=${KERNEL_ID}"

run_eaft_base() {
  bash "${ROOT_DIR}/scripts/versa_run_eaft_single_kaggle.sh" \
    --kernel-id "${KERNEL_ID}" \
    --model-id "openai/gpt-oss-20b" \
    --model-path "/kaggle/input/gpt-oss-20b/transformers/default/1" \
    --seq-lens-csv 1024,2048 \
    --num-blocks 512 --batch-size 1 --sample-points 200000 \
    --top-k 4 --entropy-topk 20 --cc-quantile 0.15 \
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

echo "[*] EAFT bigblocks: base run A"
OUT_A="$(run_eaft_base)"
echo "${OUT_A}"
LOG_A="$(extract_remote_log "${OUT_A}")"
echo "[*] baseA_remote_log=${LOG_A}"

bash "${ROOT_DIR}/scripts/kaggle_watch_file.sh" \
  --remote-path "${LOG_A}" \
  --pattern "\\[\\+\\] Wrote /kaggle/working/artifacts/eaft_models/" \
  --fail-pattern "Fatal Python error: Segmentation fault|Segmentation fault \\(core dumped\\)|torch\\.OutOfMemoryError|CUDA out of memory|OutOfMemoryError|Traceback \\(most recent call last\\)" \
  --interval-s 60 \
  --tail-n 120

JSON_A_REL="$(extract_written_json_relpath "${LOG_A}")"
JSON_A_LOCAL="$(download_remote_file_rel "${JSON_A_REL}")"
echo "[+] baseA_json_local=${JSON_A_LOCAL}"

echo "[*] EAFT bigblocks: base run B"
OUT_B="$(run_eaft_base)"
echo "${OUT_B}"
LOG_B="$(extract_remote_log "${OUT_B}")"
echo "[*] baseB_remote_log=${LOG_B}"

bash "${ROOT_DIR}/scripts/kaggle_watch_file.sh" \
  --remote-path "${LOG_B}" \
  --pattern "\\[\\+\\] Wrote /kaggle/working/artifacts/eaft_models/" \
  --fail-pattern "Fatal Python error: Segmentation fault|Segmentation fault \\(core dumped\\)|torch\\.OutOfMemoryError|CUDA out of memory|OutOfMemoryError|Traceback \\(most recent call last\\)" \
  --interval-s 60 \
  --tail-n 120

JSON_B_REL="$(extract_written_json_relpath "${LOG_B}")"
JSON_B_LOCAL="$(download_remote_file_rel "${JSON_B_REL}")"
echo "[+] baseB_json_local=${JSON_B_LOCAL}"

OUT_MD="${REPO_ROOT}/harmony/cuda-norm/reports/eaft_base_vs_base_noise_20b_bigblocks.md"
python "${REPO_ROOT}/harmony/cuda-norm/scripts/summarize_eaft_pair.py" \
  --left-json "${JSON_A_LOCAL}" \
  --right-json "${JSON_B_LOCAL}" \
  --out-md "${OUT_MD}"

echo "[+] wrote ${OUT_MD}"

