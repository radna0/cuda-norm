#!/usr/bin/env bash
set -euo pipefail

# CPU-only unit tests for DFlash logic (no GPU/TPU required).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH:-}"

python "${SCRIPT_DIR}/test_dflash_perfect_draft_parity_cpu.py"

echo "[+] cpu unit tests ok" >&2
