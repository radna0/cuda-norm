#!/usr/bin/env bash
set -euo pipefail

# Upload a local folder to a Hugging Face *dataset* repo.
#
# Authentication:
# - Preferred: HF_TOKEN env var (do not hardcode tokens in scripts).
# - Fallback: `huggingface-cli login` stored token (no HF_TOKEN required).
#
# Usage:
#   HF_TOKEN=... ./upload_folder_to_hf_dataset_repo.sh <repo_id> <local_dir> [--public]
#
# Example:
#   HF_TOKEN=... ./upload_folder_to_hf_dataset_repo.sh \
#     radna0/nemotron-math-v2-harmony-tools-meta \
#     ./cpu_out/nvidia_math_v2

REPO_ID="${1:?usage: $0 <repo_id> <local_dir> [--public]}"
LOCAL_DIR="${2:?usage: $0 <repo_id> <local_dir> [--public]}"
VISIBILITY_FLAG="${3:-}"

token_args=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  token_args=(--token "$HF_TOKEN")
else
  if huggingface-cli auth whoami >/dev/null 2>&1; then
    echo "[hf] using stored huggingface-cli auth (HF_TOKEN not set)"
  else
    echo "[err] HF_TOKEN is not set and huggingface-cli is not logged in." >&2
    echo "[err] Run: huggingface-cli login  OR  export HF_TOKEN=..." >&2
    exit 2
  fi
fi

if [[ ! -d "$LOCAL_DIR" ]]; then
  echo "[err] not a directory: $LOCAL_DIR" >&2
  exit 2
fi

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

repo_create_args=(repo create "$REPO_ID" --repo-type dataset --exist-ok "${token_args[@]}")
if [[ "$VISIBILITY_FLAG" == "--public" ]]; then
  :
else
  repo_create_args+=(--private)
fi

echo "[hf] ensuring repo exists: ${REPO_ID}"
huggingface-cli "${repo_create_args[@]}"

echo "[hf] uploading folder: ${LOCAL_DIR} -> ${REPO_ID}"
huggingface-cli upload-large-folder \
  "$REPO_ID" \
  "$LOCAL_DIR" \
  --repo-type dataset \
  "${token_args[@]}" \
  --num-workers "${HF_UPLOAD_WORKERS:-8}"

echo "[done] upload complete: https://huggingface.co/datasets/${REPO_ID}"
