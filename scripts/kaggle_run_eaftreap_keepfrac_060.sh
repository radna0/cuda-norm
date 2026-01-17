#!/usr/bin/env bash
set -euo pipefail

# Kaggle/VERSA: build EAFT-REAP prune at keep_frac=0.60 (top_k unchanged) and
# enforce keep_n multiple-of (FlashInfer MXFP4 MoE routing requires %4==0).
#
# Notes:
# - With num_experts=32, requesting keep_frac=0.60 yields keep_n=20 (0.625) when
#   enforcing multiple-of-4. This is required for SGLang MXFP4 routing kernels.
# - Use batch_size=8 for H100 (safer VRAM); raise only after probing.
#
# Required env:
#   export REMOTE_JUPYTER_URL="https://.../proxy"
# Optional:
#   export REMOTE_JUPYTER_TOKEN=""
#   export PRUNING_ENV_FILE="harmony/cuda-norm/.env"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${REMOTE_JUPYTER_URL:-}" ]]; then
  echo "[err] REMOTE_JUPYTER_URL is not set" >&2
  exit 2
fi

TASK_LOG_DIR="${PRUNING_REMOTE_LOG_DIR:-logs}"
TS="$(date +%Y%m%d_%H%M%S)"

bash "${ROOT_DIR}/scripts/versa_run_pruning_track_kaggle.sh" \
  --task build_pruned_20b_eaftreap_keepfrac \
  --model-id-20b openai/gpt-oss-20b \
  --calib-packs-repo radna0/harmony-qwen3-calib-packs-v2-20260113 \
  --calib-pack-files "packs/reasoning_style_10k_v2/reasoning_style_10k_v2.parquet,tool_agentic_10k_v6.parquet,packs/calib_prompt_10000_v2/calib_prompt_10000_v2.parquet" \
  --calib-pack-sample-strategy "per_file" \
  --keep-fracs-csv "0.60" \
  --keep-n-round "ceil" \
  --keep-n-multiple-of "4" \
  --num-rows "5000" \
  --max-seq-length "2048" \
  --batch-size "8"

echo "[+] submitted keepfrac=0.60 build on Kaggle (check remote logs under /kaggle/working/${TASK_LOG_DIR}/)"

