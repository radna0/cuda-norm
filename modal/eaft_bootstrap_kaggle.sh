#!/usr/bin/env bash
set -euo pipefail

# One-time bootstrap for EAFT collectors running inside a Kaggle kernel via Versa.
#
# We keep this as a separate script to avoid nested-quote breakage in
# `versa run --bootstrap-cmd "..."`
#
# Idempotent via sentinel in /kaggle/working/eaft_cache (persists for the life of the kernel).

SENTINEL="${EAFT_BOOTSTRAP_SENTINEL:-eaft_cache/.eaft_bootstrap_v4}"

mkdir -p "$(dirname "${SENTINEL}")"

if [[ -f "${SENTINEL}" ]]; then
  echo "[bootstrap] reuse ${SENTINEL}"
  exit 0
fi

echo "[bootstrap] installing deps (first run only)"

python -m pip install -U pip >/dev/null
python -m pip install -q \
  modal datasets \
  "transformers==4.56.2" tokenizers safetensors \
  pyarrow pandas accelerate \
  huggingface-hub hf_transfer \
  kagglehub

# Avoid torchvision ABI mismatches in Kaggle images.
python -m pip uninstall -y torchvision >/dev/null 2>&1 || true

TRITON_VERSION="${TRITON_VERSION:-3.4.0}"
KERNELS_VERSION="${KERNELS_VERSION:-0.11.7}"
python -m pip install -q "triton==${TRITON_VERSION}" "kernels==${KERNELS_VERSION}"

python -m pip install -q "sglang[all]"

if ! python -c "import torch; print('[bootstrap] torch', torch.__version__)" >/dev/null 2>&1; then
  TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
  python -m pip install -q torch --index-url "${TORCH_INDEX_URL}"
fi

touch "${SENTINEL}"
echo "[bootstrap] done"

