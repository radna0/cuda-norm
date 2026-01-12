#!/usr/bin/env bash
set -euo pipefail

# Upload a local folder to a Hugging Face *model* repo.
#
# Requires a token in HF_TOKEN (do not hardcode tokens in scripts).
#
# Usage:
#   HF_TOKEN=... ./upload_folder_to_hf_model_repo.sh <repo_id> <local_dir> [--public]
#
# Example:
#   HF_TOKEN=... ./upload_folder_to_hf_model_repo.sh \
#     radna0/gpt-oss-120b-math \
#     /dev/shm/gpt-oss-120b-math-kaggle-v2/model

REPO_ID="${1:?usage: $0 <repo_id> <local_dir> [--public]}"
LOCAL_DIR="${2:?usage: $0 <repo_id> <local_dir> [--public]}"
VISIBILITY_FLAG="${3:-}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[err] HF_TOKEN is not set. Export HF_TOKEN in your shell and re-run." >&2
  exit 2
fi

if [[ ! -d "$LOCAL_DIR" ]]; then
  echo "[err] not a directory: $LOCAL_DIR" >&2
  exit 2
fi

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

repo_create_args=(repo create "$REPO_ID" --repo-type model --exist-ok --token "$HF_TOKEN")
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
  --repo-type model \
  --token "$HF_TOKEN" \
  --num-workers "${HF_UPLOAD_WORKERS:-8}"

echo "[done] upload complete: https://huggingface.co/${REPO_ID}"
