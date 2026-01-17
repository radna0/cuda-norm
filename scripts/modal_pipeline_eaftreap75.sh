#!/usr/bin/env bash
set -euo pipefail

# Modal pipeline: build pruned keep_frac=0.75 then evaluate parity (EAFT) base vs pruned.
#
# Assumptions:
# - Modal profile already active (e.g. `modal profile activate phamcuongkien1219`)
# - HF_TOKEN available via `harmony/cuda-norm/.env` or shell env
#
# Usage:
#   bash scripts/modal_pipeline_eaftreap75.sh --build-log <path>
#
# Outputs (local):
# - artifacts/20b_pruned_models_eaftreap_budgeted/manifest.json (written by build task)
# - artifacts/eaft_models/<run_id>/*.json (written by EAFT single collectors)
# - reports/eaftreap_budgeted_keepfrac75_parity_summary.md

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUILD_LOG=""
POLL_S="120"

SEQ_LENS_CSV="1024,2048"
NUM_BLOCKS="256"
BATCH_SIZE="1"
SAMPLE_POINTS="200000"
TOP_K="4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-log) BUILD_LOG="$2"; shift 2;;
    --poll-s) POLL_S="$2"; shift 2;;
    --seq-lens-csv) SEQ_LENS_CSV="$2"; shift 2;;
    --num-blocks) NUM_BLOCKS="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --sample-points) SAMPLE_POINTS="$2"; shift 2;;
    --top-k) TOP_K="$2"; shift 2;;
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

if [[ -z "${BUILD_LOG}" ]]; then
  echo "[err] --build-log is required" >&2
  exit 2
fi
if [[ ! -f "${BUILD_LOG}" ]]; then
  echo "[err] build log not found: ${BUILD_LOG}" >&2
  exit 2
fi

echo "[*] Waiting for build to finish: ${BUILD_LOG}"
while :; do
  if rg -n "\\[\\+\\] Wrote reports/20b_structural_prune_build_eaftreap_budgeted\\.md" -S "${BUILD_LOG}" >/dev/null 2>&1; then
    break
  fi
  if rg -n "Stopping app - local entrypoint completed\\." -S "${BUILD_LOG}" >/dev/null 2>&1; then
    # If the entrypoint completed, the report should exist; proceed.
    break
  fi
  echo "[*] still running... $(date +%Y-%m-%dT%H:%M:%S%z)"
  sleep "${POLL_S}"
done

MANIFEST="${ROOT_DIR}/artifacts/20b_pruned_models_eaftreap_budgeted/manifest.json"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "[err] missing manifest: ${MANIFEST}" >&2
  exit 3
fi

OUT_DIR="$(python - <<PY
import json
m=json.load(open('${MANIFEST}','r',encoding='utf-8'))
print(m.get('out_dir',''))
PY
)"
if [[ -z "${OUT_DIR}" ]]; then
  echo "[err] out_dir missing in manifest: ${MANIFEST}" >&2
  exit 4
fi

echo "[+] pruned model dir: ${OUT_DIR}"

mkdir -p "${ROOT_DIR}/unsloth_logs"
TS="$(date +%Y%m%d_%H%M%S)"

echo "[*] EAFT base..."
set -a
source "${ROOT_DIR}/.env" || true
set +a
modal run "${ROOT_DIR}/modal/collect_calib_packs_eaft_single.py" \
  --model-id openai/gpt-oss-20b \
  --seq-lens-csv "${SEQ_LENS_CSV}" \
  --num-blocks "${NUM_BLOCKS}" --batch-size "${BATCH_SIZE}" \
  --sample-points "${SAMPLE_POINTS}" \
  --top-k "${TOP_K}" \
  > "${ROOT_DIR}/unsloth_logs/modal_eaft_single_base20b_${TS}.log" 2>&1

echo "[*] EAFT pruned..."
modal run "${ROOT_DIR}/modal/collect_calib_packs_eaft_single.py" \
  --model-id eaftreap_budgeted_keepfrac075 \
  --model-path "${OUT_DIR}" \
  --seq-lens-csv "${SEQ_LENS_CSV}" \
  --num-blocks "${NUM_BLOCKS}" --batch-size "${BATCH_SIZE}" \
  --sample-points "${SAMPLE_POINTS}" \
  --top-k "${TOP_K}" \
  > "${ROOT_DIR}/unsloth_logs/modal_eaft_single_pruned075_${TS}.log" 2>&1

BASE_JSON="$(ls -t ${ROOT_DIR}/artifacts/eaft_models/*/openai_gpt-oss-20b.json 2>/dev/null | head -n 1 || true)"
PRUNED_JSON="$(ls -t ${ROOT_DIR}/artifacts/eaft_models/*/eaftreap_budgeted_keepfrac075.json 2>/dev/null | head -n 1 || true)"

if [[ -z "${BASE_JSON}" || -z "${PRUNED_JSON}" ]]; then
  echo "[err] Could not find EAFT JSONs after runs." >&2
  echo "      base=${BASE_JSON:-<missing>}" >&2
  echo "      pruned=${PRUNED_JSON:-<missing>}" >&2
  exit 5
fi

OUT_MD="${ROOT_DIR}/reports/eaftreap_budgeted_keepfrac75_parity_summary.md"
python "${ROOT_DIR}/scripts/summarize_eaft_pair.py" \
  --left-json "${BASE_JSON}" \
  --right-json "${PRUNED_JSON}" \
  --out-md "${OUT_MD}" \
  --gates-json "${ROOT_DIR}/pruning/near_lossless_gates.json"

echo "[+] Wrote ${OUT_MD}"

