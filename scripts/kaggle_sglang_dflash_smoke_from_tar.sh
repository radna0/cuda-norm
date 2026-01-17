#!/usr/bin/env bash
set -euo pipefail

# Kaggle/H100: smoke load + benchmark an exported SGLang DFlashDraftModel tarball.
#
# Inputs:
#   DRAFT_TAR_GZ: path to tar.gz containing a directory with config.json + model shards
#   TARGET_MODEL: e.g. openai/gpt-oss-20b
#
# Example:
#   DRAFT_TAR_GZ=/kaggle/input/.../gptoss20b_run2000_sglang.tar.gz \
#   TARGET_MODEL=openai/gpt-oss-20b \
#   ./cuda-norm-sync/scripts/kaggle_sglang_dflash_smoke_from_tar.sh

cd /kaggle/working

export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export TRANSFORMERS_NO_JAX=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"

DRAFT_TAR_GZ="${DRAFT_TAR_GZ:-}"
if [[ -z "$DRAFT_TAR_GZ" ]]; then
  echo "DRAFT_TAR_GZ is required" >&2
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
python -m pip -q install --no-cache-dir -U -t "$PYDEPS" --no-deps sglang==0.5.7
python -m pip -q install --no-cache-dir -U -t "$PYDEPS" fastapi uvicorn pydantic

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

export SGLANG_OVERLAY_SRC=/kaggle/working/cuda-norm-sync/sglang-flashinfer/python/sglang
echo "[stage] overlay sglang" >&2
python /kaggle/working/cuda-norm-sync/scripts/sglang_overlay_install.py

OUT_DIR="/kaggle/working/dflash_ckpt"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
echo "[stage] untar" >&2
tar -xzf "$DRAFT_TAR_GZ" -C "$OUT_DIR"

CKPT_DIR="$(find "$OUT_DIR" -maxdepth 2 -type f -name config.json -print -quit | xargs -r dirname)"
if [[ -z "$CKPT_DIR" ]]; then
  echo "could not find config.json inside tarball" >&2
  exit 2
fi

echo "[stage] smoke load" >&2
python /kaggle/working/cuda-norm-sync/scripts/sglang_dflash_draft_load_smoke.py --ckpt "$CKPT_DIR"

TARGET_MODEL="${TARGET_MODEL:-openai/gpt-oss-20b}"
ATTN_BACKEND="${ATTN_BACKEND:-fa3}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"

echo "[stage] benchmark" >&2
python /kaggle/working/cuda-norm-sync/scripts/dflash_gptoss20b_bench_sglang_kaggle.py \
  --target-model "$TARGET_MODEL" \
  --draft-model "$CKPT_DIR" \
  --attention-backend "$ATTN_BACKEND" \
  --block-size "$BLOCK_SIZE" \
  --max-new-tokens "${MAX_NEW_TOKENS:-2048}" \
  --concurrency "${CONCURRENCY:-4}" \
  --num-prompts "${NUM_PROMPTS:-8}"

echo "[+] done" >&2

