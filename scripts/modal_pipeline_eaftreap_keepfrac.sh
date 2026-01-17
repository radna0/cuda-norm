#!/usr/bin/env bash
set -euo pipefail

# Modal pipeline: EAFT-REAP structural prune at a single keep_frac (top_k unchanged),
# then run the EAFT parity collector (SGLang) and summarize against gates.
#
# This is the "no finetune" path: profiling → structural rewrite → evaluation only.
#
# Usage (example):
#   bash harmony/cuda-norm/scripts/modal_pipeline_eaftreap_keepfrac.sh \
#     --keep-frac 0.75 --keep-n-round ceil \
#     --num-rows 2000 --profile-max-seq 2048 \
#     --gpu-type B200:1 --eval-seq-lens 1024,2048 --num-blocks 512 --sample-points 200000

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

KEEP_FRAC="0.75"
KEEP_N_ROUND="ceil"
NUM_ROWS="2000"
PROFILE_MAX_SEQ="2048"
PROFILE_BATCH_SIZE="1"

GPU_TYPE="B200:1"
EVAL_SEQ_LENS="1024,2048"
NUM_BLOCKS="512"
EVAL_BATCH_SIZE="1"
SAMPLE_POINTS="200000"
TOP_K="4"

TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-frac) KEEP_FRAC="$2"; shift 2;;
    --keep-n-round) KEEP_N_ROUND="$2"; shift 2;;
    --num-rows) NUM_ROWS="$2"; shift 2;;
    --profile-max-seq) PROFILE_MAX_SEQ="$2"; shift 2;;
    --profile-batch-size) PROFILE_BATCH_SIZE="$2"; shift 2;;
    --gpu-type) GPU_TYPE="$2"; shift 2;;
    --eval-seq-lens) EVAL_SEQ_LENS="$2"; shift 2;;
    --num-blocks) NUM_BLOCKS="$2"; shift 2;;
    --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2;;
    --sample-points) SAMPLE_POINTS="$2"; shift 2;;
    --top-k) TOP_K="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
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

mkdir -p "${ROOT_DIR}/unsloth_logs"
TS="$(date +%Y%m%d_%H%M%S)"
TAG_SAFE="$(python - <<PY
import re
s='${TAG}'.strip() or ''
s=re.sub(r'[^a-zA-Z0-9._-]+','_',s)
print(s)
PY
)"
SUF=""
if [[ -n "${TAG_SAFE}" ]]; then
  SUF="_${TAG_SAFE}"
fi

set -a
source "${ROOT_DIR}/.env" 2>/dev/null || true
set +a

echo "[*] CPU predownload (model + calib packs)..."
modal run "${ROOT_DIR}/modal/gpt_oss_pruning_track.py" --task predownload_20b \
  > "${ROOT_DIR}/unsloth_logs/modal_predownload_20b_${TS}${SUF}.log" 2>&1
modal run "${ROOT_DIR}/modal/gpt_oss_pruning_track.py" --task predownload_calib_packs \
  > "${ROOT_DIR}/unsloth_logs/modal_predownload_calib_packs_${TS}${SUF}.log" 2>&1

echo "[*] Build pruned model (EAFT-REAP keep_frac=${KEEP_FRAC})..."
modal run "${ROOT_DIR}/modal/gpt_oss_pruning_track.py" --task build_pruned_20b_eaftreap_keepfrac \
  --keep-fracs-csv "${KEEP_FRAC}" --keep-n-round "${KEEP_N_ROUND}" \
  --num-rows "${NUM_ROWS}" --max-seq-length "${PROFILE_MAX_SEQ}" --batch-size "${PROFILE_BATCH_SIZE}" \
  > "${ROOT_DIR}/unsloth_logs/modal_build_eaftreap_keepfrac_${KEEP_FRAC}_${TS}${SUF}.log" 2>&1

MANIFEST="${ROOT_DIR}/artifacts/20b_pruned_models_eaftreap_keepfrac/manifest_eaftreap_keepfrac.json"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "[err] missing manifest: ${MANIFEST}" >&2
  exit 3
fi

OUT_DIR="$(python - <<PY
import json
m=json.load(open('${MANIFEST}','r',encoding='utf-8'))
v=m.get('variants') or {}
if not v:
  raise SystemExit('no variants in manifest')
# only one variant for single keep_frac runs
name=sorted(v.keys())[0]
print(v[name])
PY
)"
MODEL_LABEL="$(python - <<PY
import json
m=json.load(open('${MANIFEST}','r',encoding='utf-8'))
v=m.get('variants') or {}
print(sorted(v.keys())[0])
PY
)"

echo "[+] pruned model: ${MODEL_LABEL}"
echo "[+] pruned model dir: ${OUT_DIR}"

echo "[*] EAFT collect base..."
env GPU_TYPE="${GPU_TYPE}" SGLANG_DISABLE_FLASHINFER_AUTOTUNE=1 modal run "${ROOT_DIR}/modal/collect_calib_packs_eaft_single.py" \
  --model-id openai/gpt-oss-20b \
  --seq-lens-csv "${EVAL_SEQ_LENS}" \
  --num-blocks "${NUM_BLOCKS}" --batch-size "${EVAL_BATCH_SIZE}" \
  --sample-points "${SAMPLE_POINTS}" \
  --top-k "${TOP_K}" \
  > "${ROOT_DIR}/unsloth_logs/modal_eaft_single_base_${KEEP_FRAC}_${TS}${SUF}.log" 2>&1

echo "[*] EAFT collect pruned..."
env GPU_TYPE="${GPU_TYPE}" SGLANG_DISABLE_FLASHINFER_AUTOTUNE=1 modal run "${ROOT_DIR}/modal/collect_calib_packs_eaft_single.py" \
  --model-id "${MODEL_LABEL}" \
  --model-path "${OUT_DIR}" \
  --seq-lens-csv "${EVAL_SEQ_LENS}" \
  --num-blocks "${NUM_BLOCKS}" --batch-size "${EVAL_BATCH_SIZE}" \
  --sample-points "${SAMPLE_POINTS}" \
  --top-k "${TOP_K}" \
  > "${ROOT_DIR}/unsloth_logs/modal_eaft_single_pruned_${KEEP_FRAC}_${TS}${SUF}.log" 2>&1

BASE_JSON="$(ls -t ${ROOT_DIR}/artifacts/eaft_models/*/openai_gpt-oss-20b.json 2>/dev/null | head -n 1 || true)"
PRUNED_JSON="$(ls -t ${ROOT_DIR}/artifacts/eaft_models/*/${MODEL_LABEL}.json 2>/dev/null | head -n 1 || true)"
if [[ -z "${BASE_JSON}" || -z "${PRUNED_JSON}" ]]; then
  echo "[err] Could not find EAFT JSONs after runs." >&2
  echo "      base=${BASE_JSON:-<missing>}" >&2
  echo "      pruned=${PRUNED_JSON:-<missing>}" >&2
  exit 4
fi

OUT_MD="${ROOT_DIR}/reports/eaftreap_keepfrac${KEEP_FRAC//./p}_parity_summary${SUF}.md"
python "${ROOT_DIR}/scripts/summarize_eaft_pair.py" \
  --left-json "${BASE_JSON}" \
  --right-json "${PRUNED_JSON}" \
  --out-md "${OUT_MD}" \
  --gates-json "${ROOT_DIR}/pruning/near_lossless_gates.json"

echo "[+] Wrote ${OUT_MD}"

