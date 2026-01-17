#!/usr/bin/env bash
set -euo pipefail

# Sync `harmony/cuda-norm` to a remote Jupyter server (Kaggle) and overlay our
# patched SGLang sources (including DFLASH support) into the remote site-packages.
#
# Required:
#   export REMOTE_JUPYTER_URL="https://.../proxy"
# Optional:
#   export REMOTE_JUPYTER_TOKEN=""
#
# Notes:
# - This does NOT start any long-running GPU job; it's a bootstrap/install step.
# - We avoid rebuilding sgl-kernel from source by installing `sglang[all]` from pip
#   and then copying our patched python files over it.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"
SYNC_DIR="${ROOT_DIR}/.versa_sync_min"

if [[ -z "${REMOTE_JUPYTER_URL:-}" ]]; then
  echo "REMOTE_JUPYTER_URL is not set"
  exit 2
fi

KERNEL_ID="${REMOTE_JUPYTER_KERNEL_ID:-}"
DETACH="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel-id) KERNEL_ID="$2"; shift 2;;
    --no-detach) DETACH=""; shift 1;;
    -h|--help)
      sed -n '1,160p' "$0"
      exit 0
      ;;
    *)
      echo "[err] Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

PYTHONPATH="${REPO_ROOT}/third_party/Versa" \
python -m versa run \
  --backend jupyter \
  --url "${REMOTE_JUPYTER_URL}" \
  ${REMOTE_JUPYTER_TOKEN:+--token "${REMOTE_JUPYTER_TOKEN}"} \
  ${KERNEL_ID:+--kernel-id "${KERNEL_ID}"} \
  --sync-local-dir "${SYNC_DIR}" \
  --sync-remote-dir "cuda-norm" \
  ${DETACH:+--detach} \
  --bootstrap-cmd "python -m pip install -U pip" \
  --bootstrap-cmd "python -m pip install 'sglang[all]'" \
  --bootstrap-cmd "python -m pip install kernels==0.11.7 || true" \
  --bootstrap-cmd "python cuda-norm/scripts/sglang_overlay_install.py" \
  --bootstrap-cmd "python -c \"import sglang; from sglang.srt.speculative.spec_info import SpeculativeAlgorithm as A; print('DFLASH' in [x.name for x in A])\"" \
  python -c "import sglang; print('sglang dflash overlay installed:', sglang.__version__)"
