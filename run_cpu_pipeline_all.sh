#!/usr/bin/env bash
set -euo pipefail

# Convenience runner for the EPYC CPU box.
# Normalizes the requested NVIDIA Nemotron datasets to Parquet (Harmony text-first),
# then builds simple candidate pools for Modal NLL scoring.
#
# Usage:
#   ./run_cpu_pipeline_all.sh [--out_root cpu_out] [--pools_root cpu_pools] [--max_records N]
#
# Notes:
# - By default this processes *all* records (can be very large). Use --max_records for a quick smoke test.
# - Requires network access to HF unless you adapt cpu_normalize_dataset.py to read local shards.

OUT_ROOT="/dev/shm/harmony_cpu_out"
POOLS_ROOT="/dev/shm/harmony_cpu_pools"
MAX_RECORDS="0"
NUM_WORKERS="$(nproc)"
MIN_SEGMENT_MB="32"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out_root) OUT_ROOT="$2"; shift 2 ;;
    --pools_root) POOLS_ROOT="$2"; shift 2 ;;
    --max_records) MAX_RECORDS="$2"; shift 2 ;;
    --num_workers) NUM_WORKERS="$2"; shift 2 ;;
    --min_segment_mb) MIN_SEGMENT_MB="$2"; shift 2 ;;
    *) echo "[err] unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT_ROOT" "$POOLS_ROOT"

# Avoid oversubscribing native libs (we're using multiprocessing).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# Put HF cache in RAM to avoid filling the root disk during huge downloads.
export HF_HOME="${HF_HOME:-/dev/shm/hf}"
mkdir -p "$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

datasets=(
  "nvidia/Nemotron-Math-v2"
  "nvidia/Nemotron-Math-Proofs-v1"
  "nvidia/Nemotron-Science-v1"
  "nvidia/Nemotron-Agentic-v1"
  "nvidia/Nemotron-Instruction-Following-Chat-v1"
)

cache_dir=""

for ds in "${datasets[@]}"; do
  tag="${ds//\//__}"
  out_dir="${OUT_ROOT}/${tag}"
  pools_dir="${POOLS_ROOT}/${tag}"

  echo "[*] normalize: ${ds} -> ${out_dir}"
  python cpu_normalize_dataset.py \
    --dataset "$ds" \
    --out_dir "$out_dir" \
    --hf_layout --write_readme \
    --drop_invalid_harmony --drop_empty_completion \
    --num_workers "$NUM_WORKERS" --min_segment_mb "$MIN_SEGMENT_MB" \
    --max_records "$MAX_RECORDS"

  # Free RAM-disk space: the hub cache is re-downloadable.
  cache_dir="${HF_HOME}/hub/datasets--${ds//\//--}"
  if [[ -d "$cache_dir" ]]; then
    echo "[*] cleanup: rm -rf ${cache_dir}"
    rm -rf "$cache_dir"
  fi

  echo "[*] pools: ${out_dir} -> ${pools_dir}"
  python cpu_build_candidate_pools.py \
    --in_dir "$out_dir" \
    --out_dir "$pools_dir"
done

echo "[done] cpu normalization + pools complete."
