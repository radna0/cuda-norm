#!/usr/bin/env bash
set -euo pipefail

# Orchestrate the “0.75 keep24 uniform” ablation end-to-end on Kaggle/VERSA:
#   1) Wait for a pruning build to finish
#   2) Download manifest + build report (stable remote paths)
#   3) Run EAFT single-model collectors (base + pruned) in bigblocks regime
#   4) Download the resulting JSONs
#   5) Produce a pair parity markdown via summarize_eaft_pair.py
#
# Intended usage: run this script under nohup and monitor its log.
#
# Required env:
#   KAGGLE_URL=... (from harmony/cuda-norm/.env)
# Optional env:
#   REMOTE_JUPYTER_KERNEL_ID=... (defaults to arg)
#
# Usage:
#   bash harmony/cuda-norm/scripts/kaggle_orchestrate_keep24_bigblocks.sh \
#     --kernel-id <kernel_id> \
#     --build-remote-log logs/build_pruned_20b_eaftreap_keepfrac_<ts>.log

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"

KERNEL_ID=""
BUILD_REMOTE_LOG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel-id) KERNEL_ID="$2"; shift 2;;
    --build-remote-log) BUILD_REMOTE_LOG="$2"; shift 2;;
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
  KERNEL_ID="${REMOTE_JUPYTER_KERNEL_ID:-}"
fi
if [[ -z "${KERNEL_ID}" ]]; then
  echo "[err] --kernel-id (or REMOTE_JUPYTER_KERNEL_ID) is required" >&2
  exit 2
fi
if [[ -z "${BUILD_REMOTE_LOG}" ]]; then
  echo "[err] --build-remote-log is required" >&2
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
echo "[*] build_remote_log=${BUILD_REMOTE_LOG}"
echo "[*] remote_url=${REMOTE_JUPYTER_URL}"

echo "[*] waiting for prune build completion..."
bash "${ROOT_DIR}/scripts/versa_monitor_kaggle_log.sh" \
  --kernel-id "${KERNEL_ID}" \
  --remote-log "${BUILD_REMOTE_LOG}" \
  --pattern "\\[\\+\\] Wrote reports/20b_structural_prune_build_eaftreap_keepfrac\\.md" \
  --interval-s 60 \
  --tail-n 120

echo "[+] prune build complete; downloading manifest + build report"

MANIFEST_REMOTE="artifacts/harmony_cuda_norm/20b_pruned_models_eaftreap_keepfrac/manifest_eaftreap_keepfrac.json"
BUILD_REPORT_REMOTE="reports/20b_structural_prune_build_eaftreap_keepfrac.md"

MANIFEST_LOCAL="harmony/cuda-norm/artifacts/20b_pruned_models_eaftreap_keepfrac/manifest_eaftreap_keepfrac.json"
BUILD_REPORT_LOCAL="harmony/cuda-norm/reports/20b_structural_prune_build_eaftreap_keepfrac.md"

bash "${ROOT_DIR}/scripts/kaggle_download_file.sh" --remote-path "${MANIFEST_REMOTE}" --out "${REPO_ROOT}/${MANIFEST_LOCAL}"
bash "${ROOT_DIR}/scripts/kaggle_download_file.sh" --remote-path "${BUILD_REPORT_REMOTE}" --out "${REPO_ROOT}/${BUILD_REPORT_LOCAL}"

python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("harmony/cuda-norm/artifacts/20b_pruned_models_eaftreap_keepfrac/manifest_eaftreap_keepfrac.json").read_text())
variants = manifest.get("variants") or {}
if not isinstance(variants, dict) or not variants:
    raise SystemExit("[err] manifest missing variants")
if len(variants) != 1:
    raise SystemExit(f"[err] expected 1 variant, got: {list(variants)}")
name, path = next(iter(variants.items()))
print(f"[+] pruned_variant_name={name}")
print(f"[+] pruned_model_path={path}")
PY

PRUNED_VARIANT_NAME="$(python - <<'PY'
import json
from pathlib import Path
manifest=json.loads(Path("harmony/cuda-norm/artifacts/20b_pruned_models_eaftreap_keepfrac/manifest_eaftreap_keepfrac.json").read_text())
name,_=next(iter((manifest.get("variants") or {}).items()))
print(name)
PY
)"

PRUNED_MODEL_PATH="$(python - <<'PY'
import json
from pathlib import Path
manifest=json.loads(Path("harmony/cuda-norm/artifacts/20b_pruned_models_eaftreap_keepfrac/manifest_eaftreap_keepfrac.json").read_text())
_,path=next(iter((manifest.get("variants") or {}).items()))
print(path)
PY
)"

run_eaft() {
  local model_id="$1"
  local model_path="$2"
  local slug
  slug="$(echo "${model_id}" | tr '/:' '__')"
  local out
  if [[ -n "${model_path}" ]]; then
    out="$(bash "${ROOT_DIR}/scripts/versa_run_eaft_single_kaggle.sh" \
      --kernel-id "${KERNEL_ID}" \
      --model-id "${model_id}" \
      --model-path "${model_path}" \
      --seq-lens-csv 1024,2048 \
      --num-blocks 512 --batch-size 1 --sample-points 200000 \
      --top-k 4 --entropy-topk 20 --cc-quantile 0.15 \
      --skip-predownload)"
  else
    out="$(bash "${ROOT_DIR}/scripts/versa_run_eaft_single_kaggle.sh" \
      --kernel-id "${KERNEL_ID}" \
      --model-id "${model_id}" \
      --seq-lens-csv 1024,2048 \
      --num-blocks 512 --batch-size 1 --sample-points 200000 \
      --top-k 4 --entropy-topk 20 --cc-quantile 0.15 \
      --skip-predownload)"
  fi
  echo "${out}"
}

extract_remote_log() {
  local text="$1"
  echo "${text}" | rg -n "remote_log=" | tail -n 1 | sed -E 's/^.*remote_log=//'
}

extract_json_remote_path_from_remote_log_file() {
  local remote_log="$1"
  local tmp="${REPO_ROOT}/harmony/cuda-norm/unsloth_logs/remote_${remote_log//\\//_}"
  mkdir -p "$(dirname "${tmp}")"
  bash "${ROOT_DIR}/scripts/kaggle_download_file.sh" --remote-path "${remote_log}" --out "${tmp}"
  python - <<PY
import re
from pathlib import Path
txt=Path("${tmp}").read_text(encoding="utf-8", errors="ignore")
m=None
for line in txt.splitlines()[::-1]:
    if "Wrote " in line and line.strip().endswith(".json"):
        m=re.search(r"Wrote\\s+(/kaggle/working/[^\\s]+\\.json)\\s*$", line)
        if m:
            break
if not m:
    raise SystemExit("[err] could not find JSON path in remote log")
path=m.group(1)
print(path.replace("/kaggle/working/","",1))
PY
}

download_eaft_json() {
  local remote_rel="$1"
  local local_path="${REPO_ROOT}/harmony/cuda-norm/${remote_rel}"
  mkdir -p "$(dirname "${local_path}")"
  bash "${ROOT_DIR}/scripts/kaggle_download_file.sh" --remote-path "${remote_rel}" --out "${local_path}"
  echo "${local_path}"
}

echo "[*] running EAFT bigblocks: base"
BASE_OUT="$(run_eaft "openai/gpt-oss-20b" "/kaggle/input/gpt-oss-20b/transformers/default/1")"
echo "${BASE_OUT}"
BASE_REMOTE_LOG="$(extract_remote_log "${BASE_OUT}")"
echo "[*] base_remote_log=${BASE_REMOTE_LOG}"

echo "[*] waiting for base JSON..."
bash "${ROOT_DIR}/scripts/versa_monitor_kaggle_log.sh" \
  --kernel-id "${KERNEL_ID}" \
  --remote-log "${BASE_REMOTE_LOG}" \
  --pattern "\\[\\+\\] Wrote /kaggle/working/artifacts/eaft_models/" \
  --interval-s 60 \
  --tail-n 120

BASE_JSON_REMOTE_REL="$(extract_json_remote_path_from_remote_log_file "${BASE_REMOTE_LOG}")"
echo "[+] base_json_remote_rel=${BASE_JSON_REMOTE_REL}"
BASE_JSON_LOCAL="$(download_eaft_json "${BASE_JSON_REMOTE_REL}")"
echo "[+] base_json_local=${BASE_JSON_LOCAL}"

echo "[*] running EAFT bigblocks: pruned"
PRUNED_OUT="$(run_eaft "${PRUNED_VARIANT_NAME}" "${PRUNED_MODEL_PATH}")"
echo "${PRUNED_OUT}"
PRUNED_REMOTE_LOG="$(extract_remote_log "${PRUNED_OUT}")"
echo "[*] pruned_remote_log=${PRUNED_REMOTE_LOG}"

echo "[*] waiting for pruned JSON..."
bash "${ROOT_DIR}/scripts/versa_monitor_kaggle_log.sh" \
  --kernel-id "${KERNEL_ID}" \
  --remote-log "${PRUNED_REMOTE_LOG}" \
  --pattern "\\[\\+\\] Wrote /kaggle/working/artifacts/eaft_models/" \
  --interval-s 60 \
  --tail-n 120

PRUNED_JSON_REMOTE_REL="$(extract_json_remote_path_from_remote_log_file "${PRUNED_REMOTE_LOG}")"
echo "[+] pruned_json_remote_rel=${PRUNED_JSON_REMOTE_REL}"
PRUNED_JSON_LOCAL="$(download_eaft_json "${PRUNED_JSON_REMOTE_REL}")"
echo "[+] pruned_json_local=${PRUNED_JSON_LOCAL}"

OUT_MD="${REPO_ROOT}/harmony/cuda-norm/reports/eaftreap75_bigblocks_1024_2048.md"
python "${REPO_ROOT}/harmony/cuda-norm/scripts/summarize_eaft_pair.py" \
  --left-json "${BASE_JSON_LOCAL}" \
  --right-json "${PRUNED_JSON_LOCAL}" \
  --out-md "${OUT_MD}"

echo "[+] wrote ${OUT_MD}"
