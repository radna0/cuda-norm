#!/usr/bin/env bash
set -euo pipefail

# CPU-only unit tests for DFlash logic (no GPU/TPU required).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${SCRIPT_DIR}/..:${SCRIPT_DIR}/../external/EasyDeL:${PYTHONPATH:-}"
export EASYDEL_SKIP_VERSION_CHECK=1
export JAX_PLATFORMS=cpu

python "${SCRIPT_DIR}/test_dflash_perfect_draft_parity_cpu.py"
python "${SCRIPT_DIR}/test_dflash_chunked_ce_parity_cpu.py"

echo "[+] cpu unit tests ok" >&2
