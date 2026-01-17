#!/usr/bin/env bash
set -euo pipefail

# Run a DFlash vs baseline decode throughput sweep on Kaggle.
#
# Requires:
#   - SGLang + overlay installed (this script installs them into .pydeps)
#   - A draft checkpoint (either HF-trained and converted, or EasyDeL converted)
#
# Use:
#   DRAFT_MODEL_DIR=/path/to/sglang_dflash_draft \
#   TARGET_MODEL=openai/gpt-oss-20b \
#   ./kaggle_dflash_bench_sweep.sh

cd /kaggle/working

export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export TRANSFORMERS_NO_JAX=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"

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
DRAFT_MODEL_DIR="${DRAFT_MODEL_DIR:-}"
if [[ -z "$DRAFT_MODEL_DIR" ]]; then
  echo "DRAFT_MODEL_DIR is required" >&2
  exit 2
fi

ATTN_BACKEND="${ATTN_BACKEND:-fa3}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
NUM_PROMPTS="${NUM_PROMPTS:-32}"
TIMEOUT_S="${TIMEOUT_S:-7200}"

CONCURRENCY_CSV="${CONCURRENCY_CSV:-1,2,4,8,16,32}"
OUT_DIR="${OUT_DIR:-/kaggle/working/dflash_bench_sweep}"
mkdir -p "$OUT_DIR"

echo "[stage] sweep" >&2
IFS=',' read -ra CONCS <<<"$CONCURRENCY_CSV"
for c in "${CONCS[@]}"; do
  c="${c//[[:space:]]/}"
  [[ -z "$c" ]] && continue
  echo "[bench] concurrency=$c" >&2
  python /kaggle/working/cuda-norm-sync/scripts/dflash_gptoss20b_bench_sglang_kaggle.py \
    --target-model "$TARGET_MODEL" \
    --draft-model "$DRAFT_MODEL_DIR" \
    --attention-backend "$ATTN_BACKEND" \
    --block-size "$BLOCK_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --concurrency "$c" \
    --num-prompts "$NUM_PROMPTS" \
    --timeout-s "$TIMEOUT_S" \
    | tee "${OUT_DIR}/bench_c${c}.json"
done

echo "[+] done: ${OUT_DIR}" >&2

