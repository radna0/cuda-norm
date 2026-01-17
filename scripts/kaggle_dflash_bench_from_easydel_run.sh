#!/usr/bin/env bash
set -euo pipefail

# Benchmark SGLang DFLASH using an EasyDeL (TPU/JAX) draft run directory.
#
# Inputs:
#   EASYDEL_RUN_DIR: path to a run-* directory containing:
#     - config.json (draft config)
#     - tensorstore_index.json
#     - model/ (zarr arrays)
#
# Output:
#   Converts to a SGLang-loadable draft checkpoint and runs the SGLang benchmark.

cd /kaggle/working

export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export TRANSFORMERS_NO_JAX=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"

EASYDEL_RUN_DIR="${EASYDEL_RUN_DIR:-}"
if [[ -z "$EASYDEL_RUN_DIR" ]]; then
  echo "EASYDEL_RUN_DIR is required" >&2
  exit 2
fi

_load_dotenv_kv() {
  local env_file="$1"
  [[ -f "$env_file" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    [[ -z "$line" ]] && continue
    [[ "${line:0:1}" == "#" ]] && continue
    [[ "$line" == export\ * ]] && line="${line#export }"
    if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
      local key="${line%%=*}"
      local val="${line#*=}"
      if [[ "${val:0:1}" == "\"" && "${val: -1}" == "\"" ]]; then
        val="${val:1:${#val}-2}"
      elif [[ "${val:0:1}" == "'" && "${val: -1}" == "'" ]]; then
        val="${val:1:${#val}-2}"
      fi
      if [[ -z "${!key:-}" ]]; then
        export "${key}=${val}"
      fi
    fi
  done < "$env_file"
}

_load_dotenv_kv "/kaggle/working/cuda-norm-sync/.env"

PYDEPS=/kaggle/working/.pydeps
mkdir -p "$PYDEPS"
export PYTHONPATH="/kaggle/working/cuda-norm-sync:$PYDEPS:${PYTHONPATH:-}"

echo "[stage] deps" >&2
python -m pip -q install --no-cache-dir -U -t "$PYDEPS" \
  huggingface_hub==0.36.0 hf-transfer \
  safetensors datasets pyarrow requests tokenizers
python -m pip -q install --no-cache-dir -U -t "$PYDEPS" --no-deps transformers==4.57.1

mkdir -p "$PYDEPS/sklearn" "$PYDEPS/sklearn/metrics" "$PYDEPS/tensorflow"
cat >"$PYDEPS/sklearn/__init__.py" <<'PY'
__all__ = []
__version__ = "0.0.0-stub"
PY
cat >"$PYDEPS/sklearn/metrics/__init__.py" <<'PY'
def roc_curve(*args, **kwargs):  # pragma: no cover
    raise RuntimeError("sklearn is stubbed out on Kaggle for ABI safety; roc_curve unavailable")
PY
cat >"$PYDEPS/tensorflow/__init__.py" <<'PY'
__all__ = []
__version__ = "0.0.0-stub"
PY

echo "[stage] install sglang" >&2
python -m pip -q install --no-cache-dir -U -t "$PYDEPS" --no-deps sglang==0.5.7
python -m pip -q install --no-cache-dir -U -t "$PYDEPS" fastapi uvicorn pydantic

export SGLANG_OVERLAY_SRC=/kaggle/working/cuda-norm-sync/sglang-flashinfer/python/sglang
echo "[stage] overlay sglang" >&2
python /kaggle/working/cuda-norm-sync/scripts/sglang_overlay_install.py

TARGET_MODEL="${TARGET_MODEL:-openai/gpt-oss-20b}"
TEACHER_ATTN_BACKEND="${TEACHER_ATTN_BACKEND:-fa3}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"

OUT_ROOT="/kaggle/working/dflash_from_easydel"
mkdir -p "$OUT_ROOT"
CONVERTED_DIR="${OUT_ROOT}/draft_sglang"

echo "[stage] convert easydel -> sglang" >&2
python /kaggle/working/cuda-norm-sync/scripts/convert_easydel_dflash_ckpt_to_sglang.py \
  --run-dir "$EASYDEL_RUN_DIR" \
  --dst "$CONVERTED_DIR" \
  --keep-fc-bias \
  --force

echo "[stage] smoke load" >&2
python /kaggle/working/cuda-norm-sync/scripts/sglang_dflash_draft_load_smoke.py \
  --ckpt "$CONVERTED_DIR"

echo "[stage] benchmark" >&2
python /kaggle/working/cuda-norm-sync/scripts/dflash_gptoss20b_bench_sglang_kaggle.py \
  --target-model "$TARGET_MODEL" \
  --draft-model "$CONVERTED_DIR" \
  --attention-backend "$TEACHER_ATTN_BACKEND" \
  --block-size "$BLOCK_SIZE" \
  --max-new-tokens "${BENCH_MAX_NEW_TOKENS:-2048}" \
  --concurrency "${BENCH_CONCURRENCY:-4}" \
  --num-prompts "${BENCH_NUM_PROMPTS:-8}" \
  | tee "${OUT_ROOT}/bench.json"

echo "[+] done: ${OUT_ROOT}" >&2

