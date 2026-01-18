#!/usr/bin/env bash
set -euo pipefail

# Kaggle/VERSA: build a "noop rewrite" (keep_n=32) and validate losslessness via
# EAFT bigblocks parity vs base.
#
# Always run under nohup and monitor the local log.
#
# Usage:
#   bash harmony/cuda-norm/scripts/kaggle_noop_rewrite_lossless_bigblocks.sh --kernel-id <id>

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

echo "[*] launching noop rewrite build..."
tmp="$(mktemp -t noop_build_launcher_XXXXXX.log)"
echo "[*] build launcher output tee=${tmp}"

set +e
bash "${ROOT_DIR}/scripts/versa_run_pruning_track_kaggle.sh" \
  --kernel-id "${KERNEL_ID}" \
  --task build_pruned_20b_noop_rewrite \
  --model-id-20b openai/gpt-oss-20b \
  2>&1 | tee "${tmp}"
rc="${PIPESTATUS[0]}"
set -e
echo "[*] build launcher exitcode=${rc}"

BUILD_REMOTE_LOG="$(
  python - <<PY
import json
from pathlib import Path
txt = Path("${tmp}").read_text(encoding="utf-8", errors="ignore")
start = txt.find("{")
end = txt.rfind("}")
if start == -1 or end == -1 or end <= start:
    raise SystemExit("")
obj = json.loads(txt[start : end + 1])
print(obj.get("log_path") or "")
PY
)"
if [[ -z "${BUILD_REMOTE_LOG}" ]]; then
  echo "[err] could not parse build remote log from launcher output" >&2
  exit 2
fi
echo "[+] build_remote_log=${BUILD_REMOTE_LOG}"

echo "[*] waiting for noop build completion (curl polling)..."
bash "${ROOT_DIR}/scripts/kaggle_watch_file.sh" \
  --remote-path "${BUILD_REMOTE_LOG}" \
  --pattern "\\[\\+\\] Wrote reports/20b_noop_rewrite_build\\.md" \
  --fail-pattern "Fatal Python error: Segmentation fault|Segmentation fault \\(core dumped\\)|torch\\.OutOfMemoryError|CUDA out of memory|OutOfMemoryError|Traceback \\(most recent call last\\)" \
  --interval-s 60 \
  --tail-n 120

echo "[+] noop build complete; downloading manifest + build report"

MANIFEST_REMOTE="artifacts/harmony_cuda_norm/20b_pruned_models_noop/manifest.json"
REPORT_REMOTE="reports/20b_noop_rewrite_build.md"

MANIFEST_LOCAL="${REPO_ROOT}/harmony/cuda-norm/artifacts/20b_pruned_models_noop/manifest.json"
REPORT_LOCAL="${REPO_ROOT}/harmony/cuda-norm/reports/20b_noop_rewrite_build.md"

bash "${ROOT_DIR}/scripts/kaggle_download_file.sh" --remote-path "${MANIFEST_REMOTE}" --out "${MANIFEST_LOCAL}"
bash "${ROOT_DIR}/scripts/kaggle_download_file.sh" --remote-path "${REPORT_REMOTE}" --out "${REPORT_LOCAL}"

NOOP_MODEL_PATH="$(
  python - <<'PY'
import json
from pathlib import Path
manifest=json.loads(Path("harmony/cuda-norm/artifacts/20b_pruned_models_noop/manifest.json").read_text())
print(manifest.get("out_dir") or "")
PY
)"
if [[ -z "${NOOP_MODEL_PATH}" ]]; then
  echo "[err] missing out_dir in noop manifest" >&2
  exit 2
fi
echo "[+] noop_model_path=${NOOP_MODEL_PATH}"

run_eaft() {
  local model_id="$1"
  local model_path="$2"
  if [[ -n "${model_path}" ]]; then
    bash "${ROOT_DIR}/scripts/versa_run_eaft_single_kaggle.sh" \
      --kernel-id "${KERNEL_ID}" \
      --model-id "${model_id}" \
      --model-path "${model_path}" \
      --seq-lens-csv 1024,2048 \
      --num-blocks 512 --batch-size 1 --sample-points 200000 \
      --top-k 4 --entropy-topk 20 --cc-quantile 0.15 \
      --skip-predownload
  else
    bash "${ROOT_DIR}/scripts/versa_run_eaft_single_kaggle.sh" \
      --kernel-id "${KERNEL_ID}" \
      --model-id "${model_id}" \
      --seq-lens-csv 1024,2048 \
      --num-blocks 512 --batch-size 1 --sample-points 200000 \
      --top-k 4 --entropy-topk 20 --cc-quantile 0.15 \
      --skip-predownload
  fi
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

echo "[*] EAFT bigblocks: noop rewrite"
NOOP_OUT="$(run_eaft "noop_rewrite_keepall_experts" "${NOOP_MODEL_PATH}")"
echo "${NOOP_OUT}"
NOOP_REMOTE_LOG="$(extract_remote_log "${NOOP_OUT}")"
echo "[*] noop_remote_log=${NOOP_REMOTE_LOG}"

bash "${ROOT_DIR}/scripts/kaggle_watch_file.sh" \
  --remote-path "${NOOP_REMOTE_LOG}" \
  --pattern "\\[\\+\\] Wrote /kaggle/working/artifacts/eaft_models/" \
  --fail-pattern "Fatal Python error: Segmentation fault|Segmentation fault \\(core dumped\\)|torch\\.OutOfMemoryError|CUDA out of memory|OutOfMemoryError|Traceback \\(most recent call last\\)" \
  --interval-s 60 \
  --tail-n 120

NOOP_JSON_REL="$(extract_written_json_relpath "${NOOP_REMOTE_LOG}")"
NOOP_JSON_LOCAL="$(download_remote_file_rel "${NOOP_JSON_REL}")"
echo "[+] noop_json_local=${NOOP_JSON_LOCAL}"

OUT_MD="${REPO_ROOT}/harmony/cuda-norm/reports/eaft_noop_rewrite_lossless_20b_bigblocks.md"
python "${REPO_ROOT}/harmony/cuda-norm/scripts/summarize_eaft_pair.py" \
  --left-json "${BASE_JSON_LOCAL}" \
  --right-json "${NOOP_JSON_LOCAL}" \
  --out-md "${OUT_MD}"

echo "[+] wrote ${OUT_MD}"

