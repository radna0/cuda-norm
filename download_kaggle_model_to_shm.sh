#!/usr/bin/env bash
set -euo pipefail

# Download a Kaggle Model Instance Version bundle to /dev/shm.
#
# Prefers Kaggle CLI (requires credentials), and falls back to the public
# download URL if Kaggle credentials aren't configured.
#
# Usage:
#   ./download_kaggle_model_to_shm.sh \
#     [model_instance_version] \
#     [out_dir] \
#     [out_filename]
#
# Example:
#   ./download_kaggle_model_to_shm.sh \
#     reyvan14/gpt-oss-120b-math/transformers/default/2 \
#     /dev/shm/gpt-oss-120b-math-kaggle-v2 \
#     model.tar.gz

MODEL_INSTANCE_VERSION="${1:-reyvan14/gpt-oss-120b-math/transformers/default/2}"
OUT_DIR="${2:-/dev/shm/gpt-oss-120b-math-kaggle-v2}"
OUT_FILENAME="${3:-model.tar.gz}"

mkdir -p "$OUT_DIR"

KAGGLE_CONFIG_A="${KAGGLE_CONFIG_DIR:-$HOME/.config/kaggle}/kaggle.json"
KAGGLE_CONFIG_B="$HOME/.kaggle/kaggle.json"

have_kaggle_auth=0
if command -v kaggle >/dev/null 2>&1; then
  if [[ -f "$KAGGLE_CONFIG_A" || -f "$KAGGLE_CONFIG_B" ]]; then
    have_kaggle_auth=1
  elif [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
    have_kaggle_auth=1
  fi
fi

if [[ "$have_kaggle_auth" -eq 1 ]]; then
  echo "[kaggle] downloading ${MODEL_INSTANCE_VERSION} -> ${OUT_DIR}"
  kaggle models instances versions download -q -p "$OUT_DIR" "$MODEL_INSTANCE_VERSION"
  echo "[done] download complete: ${OUT_DIR}"
  ls -lh "$OUT_DIR"
  exit 0
fi

echo "[warn] Kaggle CLI credentials not found; falling back to direct download URL." >&2
echo "[hint] Put your API token at ${KAGGLE_CONFIG_A} (chmod 600) or set KAGGLE_USERNAME/KAGGLE_KEY." >&2

URL="https://www.kaggle.com/api/v1/models/${MODEL_INSTANCE_VERSION}/download"
OUT_TAR="${OUT_DIR}/${OUT_FILENAME}"

echo "[curl] downloading -> ${OUT_TAR}"
curl -sS -L --fail --retry 20 --retry-delay 5 --retry-connrefused -C - -o "$OUT_TAR" "$URL"
echo "[done] download complete: ${OUT_TAR}"
ls -lh "$OUT_DIR"

