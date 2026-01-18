#!/usr/bin/env bash
set -euo pipefail

# End-to-end keep24 bigblocks parity, using curl polling (no Versa tail loops):
# 1) Wait for pruning build completion (remote build log)
# 2) Download manifest + build report from stable artifact roots
# 3) Run EAFT bigblocks collectors (base + pruned) via Versa
# 4) Poll EAFT remote logs for JSON write, download JSONs
# 5) Write parity summary markdown locally
#
# Always run this script via nohup and monitor its local log.
#
# Required:
#   harmony/cuda-norm/.env with KAGGLE_URL set
#
# Usage:
#   bash harmony/cuda-norm/scripts/kaggle_orchestrate_keep24_bigblocks_curl.sh \
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
      sed -n '1,140p' "$0"
      exit 0
      ;;
    *)
      echo "[err] Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${KERNEL_ID}" || -z "${BUILD_REMOTE_LOG}" ]]; then
  echo "[err] --kernel-id and --build-remote-log are required" >&2
  exit 2
fi

set -a
source "${ROOT_DIR}/.env"
set +a

if [[ -z "${KAGGLE_URL:-}" ]]; then
  echo "[err] KAGGLE_URL is not set (check ${ROOT_DIR}/.env)" >&2
  exit 2
fi

# kaggle_download_file.sh expects REMOTE_JUPYTER_URL.
export REMOTE_JUPYTER_URL="${KAGGLE_URL}"

echo "[*] kernel_id=${KERNEL_ID}"
echo "[*] build_remote_log=${BUILD_REMOTE_LOG}"

echo "[*] waiting for prune build completion (curl polling)..."
bash "${ROOT_DIR}/scripts/kaggle_watch_file.sh" \
  --remote-path "${BUILD_REMOTE_LOG}" \
  --pattern "\\[\\+\\] Wrote reports/20b_structural_prune_build_eaftreap_keepfrac\\.md" \
  --fail-pattern "Fatal Python error: Segmentation fault|Segmentation fault \\(core dumped\\)|torch\\.OutOfMemoryError|CUDA out of memory|OutOfMemoryError|Traceback \\(most recent call last\\)" \
  --interval-s 60 \
  --tail-n 120

echo "[+] prune build complete; downloading manifest + build report"

MANIFEST_REMOTE_PRIMARY="artifacts/harmony_cuda_norm/20b_pruned_models_eaftreap_keepfrac/manifest_eaftreap_keepfrac.json"
# Back-compat: older pruning runs wrote directly under /kaggle/working/artifacts/...
# (not under artifacts/harmony_cuda_norm). Try this if the primary path 404s.
MANIFEST_REMOTE_FALLBACK="artifacts/20b_pruned_models_eaftreap_keepfrac/manifest_eaftreap_keepfrac.json"
BUILD_REPORT_REMOTE="reports/20b_structural_prune_build_eaftreap_keepfrac.md"

MANIFEST_LOCAL="${REPO_ROOT}/harmony/cuda-norm/artifacts/20b_pruned_models_eaftreap_keepfrac/manifest_eaftreap_keepfrac.json"
BUILD_REPORT_LOCAL="${REPO_ROOT}/harmony/cuda-norm/reports/20b_structural_prune_build_eaftreap_keepfrac.md"

set +e
bash "${ROOT_DIR}/scripts/kaggle_download_file.sh" --remote-path "${MANIFEST_REMOTE_PRIMARY}" --out "${MANIFEST_LOCAL}"
rc="$?"
set -e
if [[ "${rc}" != "0" ]]; then
  echo "[warn] manifest not found at ${MANIFEST_REMOTE_PRIMARY}; trying fallback ${MANIFEST_REMOTE_FALLBACK}" >&2
  bash "${ROOT_DIR}/scripts/kaggle_download_file.sh" --remote-path "${MANIFEST_REMOTE_FALLBACK}" --out "${MANIFEST_LOCAL}"
fi
bash "${ROOT_DIR}/scripts/kaggle_download_file.sh" --remote-path "${BUILD_REPORT_REMOTE}" --out "${BUILD_REPORT_LOCAL}"

PRUNED_VARIANT_NAME="$(
  python - <<'PY'
import json
from pathlib import Path
manifest=json.loads(Path("harmony/cuda-norm/artifacts/20b_pruned_models_eaftreap_keepfrac/manifest_eaftreap_keepfrac.json").read_text())
variants=manifest.get("variants") or {}
if not isinstance(variants, dict) or not variants:
    raise SystemExit("missing variants in manifest")
if len(variants) != 1:
    raise SystemExit(f"expected exactly 1 variant, got {list(variants)}")
print(next(iter(variants.keys())))
PY
)"

PRUNED_MODEL_PATH="$(
  python - <<'PY'
import json
from pathlib import Path
manifest=json.loads(Path("harmony/cuda-norm/artifacts/20b_pruned_models_eaftreap_keepfrac/manifest_eaftreap_keepfrac.json").read_text())
variants=manifest.get("variants") or {}
print(next(iter(variants.values())))
PY
)"

echo "[+] pruned_variant_name=${PRUNED_VARIANT_NAME}"
echo "[+] pruned_model_path=${PRUNED_MODEL_PATH}"

run_eaft() {
  local model_id="$1"
  local model_path="$2"
  local out
  if [[ -n "${model_path}" ]]; then
    out="$(
      bash "${ROOT_DIR}/scripts/versa_run_eaft_single_kaggle.sh" \
        --kernel-id "${KERNEL_ID}" \
        --model-id "${model_id}" \
        --model-path "${model_path}" \
        --seq-lens-csv 1024,2048 \
        --num-blocks 512 --batch-size 1 --sample-points 200000 \
        --top-k 4 --entropy-topk 20 --cc-quantile 0.15 \
        --skip-predownload
    )"
  else
    out="$(
      bash "${ROOT_DIR}/scripts/versa_run_eaft_single_kaggle.sh" \
        --kernel-id "${KERNEL_ID}" \
        --model-id "${model_id}" \
        --seq-lens-csv 1024,2048 \
        --num-blocks 512 --batch-size 1 --sample-points 200000 \
        --top-k 4 --entropy-topk 20 --cc-quantile 0.15 \
        --skip-predownload
    )"
  fi
  echo "${out}"
}

extract_remote_log() {
  local text="$1"
  echo "${text}" | rg -n "remote_log=" | tail -n 1 | sed -E 's/^.*remote_log=//'
}

extract_written_json_relpath() {
  local remote_log="$1"
  # Download the remote log, then parse the "Wrote /kaggle/working/<...>.json" line.
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

echo "[*] EAFT bigblocks: base"
BASE_OUT="$(run_eaft "openai/gpt-oss-20b" "/kaggle/input/gpt-oss-20b/transformers/default/1")"
echo "${BASE_OUT}"
BASE_REMOTE_LOG="$(extract_remote_log "${BASE_OUT}")"
echo "[*] base_remote_log=${BASE_REMOTE_LOG}"

bash "${ROOT_DIR}/scripts/kaggle_watch_file.sh" \
  --remote-path "${BASE_REMOTE_LOG}" \
  --pattern "\\[\\+\\] Wrote /kaggle/working/artifacts/eaft_models/" \
  --fail-pattern "Fatal Python error: Segmentation fault|Segmentation fault \\(core dumped\\)|torch\\.OutOfMemoryError|CUDA out of memory|OutOfMemoryError|Traceback \\(most recent call last\\)" \
  --interval-s 60 \
  --tail-n 120

BASE_JSON_REL="$(extract_written_json_relpath "${BASE_REMOTE_LOG}")"
BASE_JSON_LOCAL="$(download_remote_file_rel "${BASE_JSON_REL}")"
echo "[+] base_json_local=${BASE_JSON_LOCAL}"

echo "[*] EAFT bigblocks: pruned"
PRUNED_OUT="$(run_eaft "${PRUNED_VARIANT_NAME}" "${PRUNED_MODEL_PATH}")"
echo "${PRUNED_OUT}"
PRUNED_REMOTE_LOG="$(extract_remote_log "${PRUNED_OUT}")"
echo "[*] pruned_remote_log=${PRUNED_REMOTE_LOG}"

bash "${ROOT_DIR}/scripts/kaggle_watch_file.sh" \
  --remote-path "${PRUNED_REMOTE_LOG}" \
  --pattern "\\[\\+\\] Wrote /kaggle/working/artifacts/eaft_models/" \
  --fail-pattern "Fatal Python error: Segmentation fault|Segmentation fault \\(core dumped\\)|torch\\.OutOfMemoryError|CUDA out of memory|OutOfMemoryError|Traceback \\(most recent call last\\)" \
  --interval-s 60 \
  --tail-n 120

PRUNED_JSON_REL="$(extract_written_json_relpath "${PRUNED_REMOTE_LOG}")"
PRUNED_JSON_LOCAL="$(download_remote_file_rel "${PRUNED_JSON_REL}")"
echo "[+] pruned_json_local=${PRUNED_JSON_LOCAL}"

OUT_MD="${REPO_ROOT}/harmony/cuda-norm/reports/eaftreap75_bigblocks_1024_2048.md"
python "${REPO_ROOT}/harmony/cuda-norm/scripts/summarize_eaft_pair.py" \
  --left-json "${BASE_JSON_LOCAL}" \
  --right-json "${PRUNED_JSON_LOCAL}" \
  --out-md "${OUT_MD}"

echo "[+] wrote ${OUT_MD}"
