#!/usr/bin/env bash
set -euo pipefail

# Remote Kaggle smoke test:
#  - installs/overlays SGLang (DFLASH-enabled)
#  - converts an HF DFlash draft checkpoint into an SGLang-loadable checkpoint
#  - runs a minimal "load config + resolve DFLASH" check (no big GPU decode here)
#
# Required:
#   export REMOTE_JUPYTER_URL="https://.../proxy"
# Optional:
#   export REMOTE_JUPYTER_TOKEN=""
#
# Inputs:
#   SRC_CKPT_DIR: remote path to HF draft ckpt (must be present on remote)
#   DST_CKPT_DIR: remote output path for shimmed ckpt
#
# Example:
#   SRC_CKPT_DIR=/kaggle/working/dflash_ckpt/step_001000 \
#   DST_CKPT_DIR=/kaggle/working/dflash_ckpt_sglang/step_001000 \
#   bash harmony/cuda-norm/scripts/versa_test_sglang_dflash_load.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"

if [[ -z "${REMOTE_JUPYTER_URL:-}" ]]; then
  echo "REMOTE_JUPYTER_URL is not set"
  exit 2
fi

if [[ -z "${SRC_CKPT_DIR:-}" ]]; then
  echo "SRC_CKPT_DIR is not set (remote path)"
  exit 2
fi

if [[ -z "${DST_CKPT_DIR:-}" ]]; then
  echo "DST_CKPT_DIR is not set (remote path)"
  exit 2
fi

PYTHONPATH="${REPO_ROOT}/third_party/Versa" \
python -m versa run \
  --backend jupyter \
  --url "${REMOTE_JUPYTER_URL}" \
  ${REMOTE_JUPYTER_TOKEN:+--token "${REMOTE_JUPYTER_TOKEN}"} \
  --sync-local-dir "${ROOT_DIR}" \
  --sync-remote-dir "cuda-norm" \
  --bootstrap-cmd "python -m pip install -U pip" \
  --bootstrap-cmd "python -m pip install 'sglang[all]' transformers==4.56.2" \
  --bootstrap-cmd "python cuda-norm/scripts/sglang_overlay_install.py" \
  --bootstrap-cmd "python -c \"from sglang.srt.speculative.spec_info import SpeculativeAlgorithm as A; print('DFLASH' in [x.name for x in A])\"" \
  --bootstrap-cmd "python cuda-norm/scripts/convert_hf_dflash_ckpt_to_sglang.py --src \"${SRC_CKPT_DIR}\" --dst \"${DST_CKPT_DIR}\" --force" \
  python -c "import json; import pathlib; p=pathlib.Path('${DST_CKPT_DIR}')/'config.json'; print('shim cfg ok', p.exists()); print(json.loads(p.read_text())['architectures'])"

