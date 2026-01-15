#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export HF_HOME="${HF_HOME:-/dev/shm/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/dev/shm/hf/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/dev/shm/hf/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/dev/shm/xdg}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/dev/shm/jax_compilation_cache_dflash}"
export TMPDIR="${TMPDIR:-/dev/shm/tmp}"
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$JAX_COMPILATION_CACHE_DIR" "$TMPDIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "[setup] creating venv (.venv-easydel) with Python 3.11..."
uv venv .venv-easydel --python 3.11
# shellcheck disable=SC1091
source .venv-easydel/bin/activate

echo "[setup] installing EasyDeL (editable) + TPU deps..."
uv pip install -U pip setuptools wheel
uv pip install -e "external/EasyDeL[tpu]"

echo "[setup] torch (CPU) for HF torch->JAX bridge..."
mkdir -p /dev/shm/uv_cache
UV_CACHE_DIR=/dev/shm/uv_cache uv pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0

echo "[setup] extra utilities..."
uv pip install safetensors ml_dtypes

echo "[setup] done. Activate with: source $ROOT/.venv-easydel/bin/activate"
