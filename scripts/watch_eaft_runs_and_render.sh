#!/usr/bin/env bash
set -euo pipefail

# Watch the most recent EAFT single-model Modal runs (started via nohup) and
# re-render the dynamic compare HTML once all JSON artifacts are written.
#
# Usage:
#   ./scripts/watch_eaft_runs_and_render.sh
#
# Output:
#   reports/eaft_dynamic_compare.html

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT}/unsloth_logs"

pick_latest() {
  local pat="$1"
  ls -t "${LOG_DIR}/${pat}" 2>/dev/null | head -n 1 || true
}

BASE_20B_LOG="$(pick_latest 'eaft_single_20b_base_h100_*.log')"
REAP_20B_LOG="$(pick_latest 'eaft_single_20b_reap_h100_*.log')"
BASE_120B_LOG="$(pick_latest 'eaft_single_120b_base_b200_trtllm_*.log')"
MATH_120B_LOG="$(pick_latest 'eaft_single_120b_math_b200_trtllm_*.log')"

if [[ -z "${BASE_20B_LOG}" || -z "${REAP_20B_LOG}" || -z "${BASE_120B_LOG}" || -z "${MATH_120B_LOG}" ]]; then
  echo "[err] Missing one or more expected log files in ${LOG_DIR}"
  echo "      base20b=${BASE_20B_LOG:-<missing>}"
  echo "      reap20b=${REAP_20B_LOG:-<missing>}"
  echo "      base120b=${BASE_120B_LOG:-<missing>}"
  echo "      math120b=${MATH_120B_LOG:-<missing>}"
  exit 2
fi

echo "[*] Watching logs:"
echo "    20B base : ${BASE_20B_LOG}"
echo "    20B reap : ${REAP_20B_LOG}"
echo "    120B base: ${BASE_120B_LOG}"
echo "    120B math: ${MATH_120B_LOG}"

extract_json() {
  # Extract the JSON path from the end-of-run line:
  #   [+] Wrote artifacts/eaft_models/<run_id>/<model>.json
  local log="$1"
  rg -n "\\[\\+\\] Wrote artifacts/eaft_models/.+\\.json" "${log}" 2>/dev/null \
    | tail -n 1 \
    | sed -E 's/^.*\\[\\+\\] Wrote (artifacts\\/eaft_models\\/[^ ]+\\.json).*$/\\1/' \
    | head -n 1 \
    || true
}

json_20b_base=""
json_20b_reap=""
json_120b_base=""
json_120b_math=""

while :; do
  [[ -z "${json_20b_base}" ]] && json_20b_base="$(extract_json "${BASE_20B_LOG}")"
  [[ -z "${json_20b_reap}" ]] && json_20b_reap="$(extract_json "${REAP_20B_LOG}")"
  [[ -z "${json_120b_base}" ]] && json_120b_base="$(extract_json "${BASE_120B_LOG}")"
  [[ -z "${json_120b_math}" ]] && json_120b_math="$(extract_json "${MATH_120B_LOG}")"

  if [[ -n "${json_20b_base}" && -n "${json_20b_reap}" && -n "${json_120b_base}" && -n "${json_120b_math}" ]]; then
    break
  fi

  echo "[*] Waiting... found:"
  echo "    20B base : ${json_20b_base:-<pending>}"
  echo "    20B reap : ${json_20b_reap:-<pending>}"
  echo "    120B base: ${json_120b_base:-<pending>}"
  echo "    120B math: ${json_120b_math:-<pending>}"
  sleep 20
done

echo "[+] All JSON artifacts ready."

python "${ROOT}/scripts/render_eaft_dynamic_compare.py" \
  --input-jsons "${ROOT}/${json_20b_base},${ROOT}/${json_20b_reap},${ROOT}/${json_120b_base},${ROOT}/${json_120b_math}" \
  --html-out "${ROOT}/reports/eaft_dynamic_compare.html"

echo "[+] Rendered: ${ROOT}/reports/eaft_dynamic_compare.html"

