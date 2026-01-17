#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for a small TPU smoke run of the cached DFlash decode harness.
#
# Required env:
#   TEACHER_SNAPSHOT_DIR
#   DRAFT_RUN_DIR
#
# Optional:
#   RUN_NAME
#   ALSO_RUN_BASELINE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_NAME="${RUN_NAME:-tpu_smoke_cached_ctx512_b8}"
export ALSO_RUN_BASELINE="${ALSO_RUN_BASELINE:-1}"

set +e
"${SCRIPT_DIR}/run_tpu_dflash_decode_cached_logged.sh" \
  --platform tpu \
  --max-prompt-len 512 \
  --max-new-tokens 128 \
  --block-size 8
exit $?
