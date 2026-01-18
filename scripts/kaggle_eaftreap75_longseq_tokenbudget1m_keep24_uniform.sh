#!/usr/bin/env bash
set -euo pipefail

# Kaggle/VERSA: validate EAFT parity at longer sequence lengths (quality-first)
# for the keep_n=24/32 (keep_frac=0.75) EAFT-REAP prune, while holding token
# budget ~constant (~1,048,576 tokens per pack).
#
# This is the next step after bigblocks PASS + noise-floor + noop-rewrite PASS:
# it tests scalability (longer seq + higher batch) without changing top_k.
#
# Always run under nohup and monitor the local log.
#
# Usage:
#   bash harmony/cuda-norm/scripts/kaggle_eaftreap75_longseq_tokenbudget1m_keep24_uniform.sh --kernel-id <id>

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

MANIFEST_LOCAL="${REPO_ROOT}/harmony/cuda-norm/artifacts/20b_pruned_models_eaftreap_keepfrac/manifest_eaftreap_keepfrac.json"
if [[ ! -f "${MANIFEST_LOCAL}" ]]; then
  echo "[err] missing manifest: ${MANIFEST_LOCAL}" >&2
  exit 2
fi

PRUNED_VARIANT_NAME="$(
  python - <<'PY'
import json
from pathlib import Path
m=json.loads(Path("harmony/cuda-norm/artifacts/20b_pruned_models_eaftreap_keepfrac/manifest_eaftreap_keepfrac.json").read_text())
vs=m.get("variants") or {}
if not isinstance(vs, dict) or not vs:
    raise SystemExit("missing variants")
if "calib_union_keep24of32_k75_eaftreap" not in vs:
    raise SystemExit(f"expected calib_union_keep24of32_k75_eaftreap in {list(vs)}")
print("calib_union_keep24of32_k75_eaftreap")
PY
)"
PRUNED_MODEL_PATH="$(
  python - <<'PY'
import json
from pathlib import Path
m=json.loads(Path("harmony/cuda-norm/artifacts/20b_pruned_models_eaftreap_keepfrac/manifest_eaftreap_keepfrac.json").read_text())
print(m["variants"]["calib_union_keep24of32_k75_eaftreap"])
PY
)"

echo "[+] pruned_variant_name=${PRUNED_VARIANT_NAME}"
echo "[+] pruned_model_path=${PRUNED_MODEL_PATH}"

remote_path_exists() {
  local path="$1"
  # Fast preflight to avoid wasting GPU time: the pruned checkpoint lives under
  # /kaggle/working (ephemeral per kernel). If it's missing, we must rebuild it
  # (or mount it as a Kaggle dataset under /kaggle/input) before running EAFT.
  #
  # Use Jupyter Contents API: 200 means the path exists.
  local rel="${path#/}"
  rel="${rel#/kaggle/working/}"
  local base="${KAGGLE_URL%/}"
  local code
  code="$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 5 --max-time 10 \"${base}/api/contents/${rel}\" || true)"
  [[ "${code}" == "200" ]]
}

if ! remote_path_exists "${PRUNED_MODEL_PATH}"; then
  echo "[err] pruned model path does not exist on this Kaggle kernel:" >&2
  echo "      ${PRUNED_MODEL_PATH}" >&2
  echo "[err] This path is ephemeral (/kaggle/working). Rebuild keep24 first:" >&2
  echo "      nohup bash harmony/cuda-norm/scripts/kaggle_keep24_pipeline.sh > harmony/cuda-norm/unsloth_logs/keep24_build.log 2>&1 &" >&2
  exit 2
fi

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
    --progress-every-s 60 \
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
  OUT_L="$(run_eaft 'openai/gpt-oss-20b' '/kaggle/input/gpt-oss-20b/transformers/default/1' "${seq_len}" "${num_blocks}" "${batch_size}" 200000)"
  echo "${OUT_L}"
  LOG_L="$(extract_remote_log "${OUT_L}")"
  echo "[*] base_remote_log=${LOG_L}"

  bash "${ROOT_DIR}/scripts/kaggle_watch_file.sh" \
    --remote-path "${LOG_L}" \
    --pattern "\\[\\+\\] Wrote /kaggle/working/artifacts/eaft_models/" \
    --fail-pattern "Fatal Python error: Segmentation fault|Segmentation fault \\(core dumped\\)|torch\\.OutOfMemoryError|CUDA out of memory|OutOfMemoryError|Traceback \\(most recent call last\\)" \
    --interval-s 120 \
    --tail-n 120

  JSON_L_REL="$(extract_written_json_relpath "${LOG_L}")"
  JSON_L_LOCAL="$(download_remote_file_rel "${JSON_L_REL}")"
  echo "[+] base_json_local=${JSON_L_LOCAL}"

  echo "[*] EAFT: pruned (keep24)"
  OUT_R="$(run_eaft "${PRUNED_VARIANT_NAME}" "${PRUNED_MODEL_PATH}" "${seq_len}" "${num_blocks}" "${batch_size}" 200000)"
  echo "${OUT_R}"
  LOG_R="$(extract_remote_log "${OUT_R}")"
  echo "[*] pruned_remote_log=${LOG_R}"

  bash "${ROOT_DIR}/scripts/kaggle_watch_file.sh" \
    --remote-path "${LOG_R}" \
    --pattern "\\[\\+\\] Wrote /kaggle/working/artifacts/eaft_models/" \
    --fail-pattern "Fatal Python error: Segmentation fault|Segmentation fault \\(core dumped\\)|torch\\.OutOfMemoryError|CUDA out of memory|OutOfMemoryError|Traceback \\(most recent call last\\)" \
    --interval-s 120 \
    --tail-n 120

  JSON_R_REL="$(extract_written_json_relpath "${LOG_R}")"
  JSON_R_LOCAL="$(download_remote_file_rel "${JSON_R_REL}")"
  echo "[+] pruned_json_local=${JSON_R_LOCAL}"

  local out_md="${REPO_ROOT}/harmony/cuda-norm/reports/_tmp_eaftreap75_${name}.md"
  python "${REPO_ROOT}/harmony/cuda-norm/scripts/summarize_eaft_pair.py" \
    --left-json "${JSON_L_LOCAL}" \
    --right-json "${JSON_R_LOCAL}" \
    --out-md "${out_md}"
  echo "[+] wrote ${out_md}"
}

mkdir -p "${REPO_ROOT}/harmony/cuda-norm/reports"

# ~1,048,576 tokens per pack:
#   blocks = 1_048_576 / seq_len
run_pair_one "seq4096_blocks256_bs4" 4096 256 4
run_pair_one "seq8192_blocks128_bs2" 8192 128 2
run_pair_one "seq16384_blocks64_bs1" 16384 64 1

FINAL_MD="${REPO_ROOT}/harmony/cuda-norm/reports/eaftreap75_longseq_tokenbudget1m_keep24_uniform.md"
{
  echo "# EAFT parity (long-seq tokenbudget ~1M) — keep24/32 EAFT-REAP"
  echo
  echo "- Base: \`openai/gpt-oss-20b\`"
  echo "- Pruned: \`${PRUNED_VARIANT_NAME}\`"
  echo "- Pruned path: \`${PRUNED_MODEL_PATH}\`"
  echo "- top_k: 4 (unchanged)"
  echo
  echo "## Matrix results"
  echo
  for f in "${REPO_ROOT}/harmony/cuda-norm/reports"/_tmp_eaftreap75_seq*.md; do
    echo "---"
    echo
    cat "${f}"
    echo
  done
} > "${FINAL_MD}"

echo "[+] wrote ${FINAL_MD}"
