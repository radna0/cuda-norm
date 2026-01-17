#!/usr/bin/env bash
set -euo pipefail

# Run TPU-side unit tests for DFlash / EasyDeL integration.
#
# Expected to run *after* EasyDeL is installed (see setup_tpu_easydel_env.sh).
#
# Usage (TPU VM / Kaggle TPU):
#   source harmony/cuda-norm/.venv-easydel/bin/activate
#   harmony/cuda-norm/scripts/run_tpu_dflash_tests.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="${ROOT}/harmony/cuda-norm:${PYTHONPATH:-}"

python "${ROOT}/harmony/cuda-norm/scripts/test_dflash_tpu_suite.py"
