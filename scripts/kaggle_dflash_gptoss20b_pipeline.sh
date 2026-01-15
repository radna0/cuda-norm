#!/usr/bin/env bash
set -euo pipefail

# End-to-end GPT-OSS-20B DFlash pipeline on Kaggle (H100):
#   1) install deps into /kaggle/working/.pydeps
#   2) overlay our patched SGLang sources (DFLASH + GPT-OSS fixes)
#   3) train HF draft checkpoint (SGLang teacher)
#   4) convert HF draft -> SGLang DFlashDraftModel checkpoint
#   5) run baseline vs DFLASH decode benchmark (max_new_tokens=2048)

cd /kaggle/working

export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export TRANSFORMERS_NO_JAX=1
export HF_HUB_ENABLE_HF_TRANSFER=1
# Kaggle's prebundled Triton toolchain can ship a non-executable `ptxas`.
# Force Triton to use CUDA's ptxas so FA3 kernels can compile.
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"

echo "[stage] start" >&2

_load_dotenv_kv() {
  local env_file="$1"
  [[ -f "$env_file" ]] || return 0
  # Parse only KEY=VALUE lines; ignore INI sections / other formats.
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"  # ltrim
    [[ -z "$line" ]] && continue
    [[ "${line:0:1}" == "#" ]] && continue
    [[ "$line" == export\ * ]] && line="${line#export }"
    if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
      local key="${line%%=*}"
      local val="${line#*=}"
      # strip surrounding quotes
      if [[ "${val:0:1}" == "\"" && "${val: -1}" == "\"" ]]; then
        val="${val:1:${#val}-2}"
      elif [[ "${val:0:1}" == "'" && "${val: -1}" == "'" ]]; then
        val="${val:1:${#val}-2}"
      fi
      # Only set if not already provided via environment.
      if [[ -z "${!key:-}" ]]; then
        export "${key}=${val}"
      fi
    fi
  done < "$env_file"
}

# Load secrets (HF_TOKEN) if present in synced `.env` (do not `source` it).
_load_dotenv_kv "/kaggle/working/cuda-norm-sync/.env"

PYDEPS=/kaggle/working/.pydeps
mkdir -p "$PYDEPS"
export PYTHONPATH="/kaggle/working/cuda-norm-sync:$PYDEPS:${PYTHONPATH:-}"

python -V
nvidia-smi -L || true

echo "[stage] pip deps" >&2

# Install runtime deps locally (Kaggle site-packages can be read-only).
python -m pip -q install --no-cache-dir -U -t "$PYDEPS" \
  huggingface_hub==0.36.0 hf-transfer \
  safetensors datasets pyarrow requests tokenizers

# Avoid re-downloading huge deps (torch/cu*) into $PYDEPS. Kaggle already ships torch.
python -m pip -q install --no-cache-dir -U -t "$PYDEPS" --no-deps transformers==4.57.1

# Kaggle images sometimes ship a binary-incompatible scikit-learn build
# (sklearn import crashes due to numpy ABI). Transformers may import sklearn
# in some optional paths; shadow it with a tiny pure-python stub.
mkdir -p "$PYDEPS/sklearn"
cat >"$PYDEPS/sklearn/__init__.py" <<'PY'
__all__ = []
__version__ = "0.0.0-stub"
PY

# Transformers tries to import `sklearn.metrics.roc_curve` in some paths; provide a stub.
mkdir -p "$PYDEPS/sklearn/metrics"
cat >"$PYDEPS/sklearn/metrics/__init__.py" <<'PY'
def roc_curve(*args, **kwargs):  # pragma: no cover
    raise RuntimeError("sklearn is stubbed out on Kaggle for ABI safety; roc_curve unavailable")
PY

# Transformers (sometimes) imports TensorFlow if it's installed; Kaggle's TF stack
# is frequently incompatible with the system NumPy / protobuf, so shadow it.
mkdir -p "$PYDEPS/tensorflow"
cat >"$PYDEPS/tensorflow/__init__.py" <<'PY'
__all__ = []
__version__ = "0.0.0-stub"
PY

echo "[stage] install sglang" >&2

# SGLang (install without deps; we overlay our patched sources after).
python -m pip -q install --no-cache-dir -U -t "$PYDEPS" --no-deps sglang==0.5.7
python -m pip -q install --no-cache-dir -U -t "$PYDEPS" fastapi uvicorn pydantic

python - <<'PY'
import sglang
print("sglang", getattr(sglang, "__version__", "unknown"))
PY

export SGLANG_OVERLAY_SRC=/kaggle/working/cuda-norm-sync/sglang-flashinfer/python/sglang
echo "[stage] overlay sglang" >&2
python /kaggle/working/cuda-norm-sync/scripts/sglang_overlay_install.py
echo "[stage] train draft" >&2

DATASET_REPO="${DATASET_REPO:-radna0/harmony-qwen3-calib-packs-v2-20260113}"
TRAIN_FILES_CSV="${TRAIN_FILES_CSV:-packs/reasoning_style_10k_v2/reasoning_style_10k_v2.parquet,tool_agentic_10k_v6.parquet,packs/calib_prompt_10000_v2/calib_prompt_10000_v2.parquet}"
TARGET_MODEL="${TARGET_MODEL:-openai/gpt-oss-20b}"
SEQ_LEN="${SEQ_LEN:-4096}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-4}"
MLP_RATIO="${MLP_RATIO:-4.0}"
MAX_STEPS="${MAX_STEPS:-200}"
SAVE_EVERY="${SAVE_EVERY:-200}"
LR="${LR:-2e-4}"
TEACHER_ATTN_BACKEND="${TEACHER_ATTN_BACKEND:-fa3}"
TEACHER_MEM_FRACTION="${TEACHER_MEM_FRACTION:-0.75}"

OUT_ROOT="/kaggle/working/dflash_gptoss20b"

python /kaggle/working/cuda-norm-sync/scripts/dflash_gptoss20b_train_kaggle.py \
  --target-model "$TARGET_MODEL" \
  --dataset-repo "$DATASET_REPO" \
  --train-files-csv "$TRAIN_FILES_CSV" \
  --seq-len "$SEQ_LEN" \
  --block-size "$BLOCK_SIZE" \
  --num-hidden-layers "$NUM_HIDDEN_LAYERS" \
  --mlp-ratio "$MLP_RATIO" \
  --max-steps "$MAX_STEPS" \
  --save-every "$SAVE_EVERY" \
  --lr "$LR" \
  --teacher-attn-backend "$TEACHER_ATTN_BACKEND" \
  --teacher-mem-fraction "$TEACHER_MEM_FRACTION" \
  --out-root "$OUT_ROOT"

# Find the newest run directory and benchmark one or more checkpoints.
echo "[stage] convert+benchmark" >&2
RUN_DIR="$(ls -1dt "$OUT_ROOT"/*/ | head -n 1)"
BENCHMARK_STEPS_CSV="${BENCHMARK_STEPS_CSV:-}"

bench_one() {
  local step_dir="$1"
  local step_name
  step_name="$(basename "$step_dir")"
  local converted_dir="${step_dir}_sglang"
  echo "[stage] convert ${step_name}" >&2
  python /kaggle/working/cuda-norm-sync/scripts/convert_hf_dflash_ckpt_to_sglang.py \
    --src "$step_dir" \
    --dst "$converted_dir" \
    --force
  echo "[stage] benchmark ${step_name}" >&2
  python /kaggle/working/cuda-norm-sync/scripts/dflash_gptoss20b_bench_sglang_kaggle.py \
    --target-model "$TARGET_MODEL" \
    --draft-model "$converted_dir" \
    --attention-backend "$TEACHER_ATTN_BACKEND" \
    --block-size "$BLOCK_SIZE" \
    --max-new-tokens "${BENCH_MAX_NEW_TOKENS:-2048}" \
    --concurrency "${BENCH_CONCURRENCY:-4}" \
    --num-prompts "${BENCH_NUM_PROMPTS:-8}" \
    | tee "${RUN_DIR}/bench_${step_name}.json"
}

if [[ -n "$BENCHMARK_STEPS_CSV" ]]; then
  IFS=',' read -ra _STEPS <<<"$BENCHMARK_STEPS_CSV"
  for raw in "${_STEPS[@]}"; do
    raw="${raw//[[:space:]]/}"
    [[ -z "$raw" ]] && continue
    step_dir="${RUN_DIR}/step_$(printf '%06d' "$raw")"
    if [[ ! -d "$step_dir" ]]; then
      echo "[warn] benchmark step dir missing: $step_dir" >&2
      continue
    fi
    bench_one "$step_dir"
  done
else
  LATEST_STEP_DIR="$(ls -1dt "$RUN_DIR"/step_* | head -n 1)"
  bench_one "$LATEST_STEP_DIR"
fi

echo "[+] done"
