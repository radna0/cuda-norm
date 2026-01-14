"""
Spike: verify SGLang can return GPT-OSS-20B hidden states (FA3 on H100).

Goal:
  - Load `openai/gpt-oss-20b` via SGLang Engine
  - Request `return_hidden_states=True`
  - Confirm what comes back (shapes/types), without building a training loop yet

Run (H100; logs to unsloth_logs/):
  mkdir -p harmony/cuda-norm/unsloth_logs
  ts=$(date +%Y%m%d_%H%M%S)
  nohup modal run harmony/cuda-norm/modal/dflash_sglang_teacher_spike_20b.py \
    --attention-backend fa3 --context-length 2048 --max-new-tokens 1 \
    > harmony/cuda-norm/unsloth_logs/dflash_sglang_spike_${ts}.log 2>&1 &
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import modal


APP_NAME = "dflash-sglang-teacher-spike-20b"
BASE_IMAGE = "nvidia/cuda:12.8.0-devel-ubuntu24.04"
_repo_root = Path(__file__).resolve().parents[1]


def _maybe_load_repo_dotenv() -> None:
    try:
        dotenv_path = _repo_root / ".env"
        if not dotenv_path.exists():
            return
        for raw in dotenv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                val = val[1:-1]
            os.environ.setdefault(key, val)
    except Exception:
        return


_maybe_load_repo_dotenv()

hf_cache_volume = modal.Volume.from_name("hf-cache-persistent", create_if_missing=True)

_secrets: list[modal.Secret] = []
if os.environ.get("HF_TOKEN"):
    _secrets.append(modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}))

SGLANG_PY_SRC = _repo_root / "sglang-flashinfer" / "python" / "sglang"
SGL_KERNEL_PY_SRC = _repo_root / "sglang-flashinfer" / "sgl-kernel" / "python" / "sgl_kernel"

image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.11")
    .apt_install(
        "git",
        "build-essential",
        "clang",
        "cmake",
        "python3-dev",
        "libnuma-dev",
        "numactl",
        "ninja-build",
    )
    .run_commands(
        "pip install --upgrade pip",
        # Install sglang + deps, then override python sources with our local fork.
        "pip install 'sglang[all]'",
        "pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128",
        # Misc deps used by our other scripts.
        "pip install 'transformers>=4.57.1,!=4.57.2' numpy pandas datasets pyarrow huggingface-hub hf-transfer",
    )
    .add_local_dir(
        str(SGLANG_PY_SRC),
        remote_path="/root/sglang-src",
        copy=True,
        ignore=["**/__pycache__", "**/__pycache__/**"],
    )
    .add_local_dir(
        str(SGL_KERNEL_PY_SRC),
        remote_path="/root/sgl-kernel-src",
        copy=True,
        ignore=["**/__pycache__", "**/__pycache__/**"],
    )
    .run_commands(
        "cp -rfv /root/sglang-src/* /usr/local/lib/python3.11/site-packages/sglang/",
        "find /usr/local/lib/python3.11/site-packages/sglang -name '__pycache__' -type d -exec rm -rf {} +",
        "cp -rfv /root/sgl-kernel-src/* /usr/local/lib/python3.11/site-packages/sgl_kernel/",
        "find /usr/local/lib/python3.11/site-packages/sgl_kernel -name '__pycache__' -type d -exec rm -rf {} +",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/root/hf_cache",
            "HUGGINGFACE_HUB_CACHE": "/root/hf_cache/hub",
            "TRANSFORMERS_CACHE": "/root/hf_cache/transformers",
            "HF_DATASETS_CACHE": "/root/hf_cache/datasets",
            # Allow using local sgl_kernel python sources even if wheel metadata mismatches.
            "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
        }
    )
)

app = modal.App(APP_NAME)


def _summarize_hidden_blob(hidden_states_obj):
    """
    SGLang returns `meta_info["hidden_states"]` as a Python list.
    For prefill, it appends a 2D list (seq_len x hidden_size).
    For decode, it appends 1D lists (hidden_size) per step.
    """
    if hidden_states_obj is None:
        return {"present": False}
    if not isinstance(hidden_states_obj, list):
        return {"present": True, "type": str(type(hidden_states_obj))}
    n = len(hidden_states_obj)
    if n == 0:
        return {"present": True, "n_items": 0}
    first = hidden_states_obj[0]
    out = {"present": True, "n_items": n}
    if isinstance(first, list) and first and isinstance(first[0], list):
        out["prefill_seq_len"] = len(first)
        out["hidden_size"] = len(first[0]) if first and isinstance(first[0], list) else None
    elif isinstance(first, list):
        out["hidden_size"] = len(first)
    # Summarize decode items if present
    if n >= 2 and isinstance(hidden_states_obj[-1], list) and hidden_states_obj[-1] and not isinstance(
        hidden_states_obj[-1][0], list
    ):
        out["decode_hidden_size"] = len(hidden_states_obj[-1])
    return out


@app.function(
    image=image,
    gpu="H100:1",
    timeout=60 * 60,
    cpu=8.0,
    memory=262144,
    volumes={"/root/hf_cache": hf_cache_volume},
    secrets=_secrets,
)
def spike_remote(
    *,
    model_id: str,
    attention_backend: str,
    context_length: int,
    max_new_tokens: int,
):
    import sglang as sgl

    hf_cache_volume.reload()

    t0 = time.time()
    engine = sgl.Engine(
        model_path=str(model_id),
        tp_size=1,
        attention_backend=str(attention_backend),
        dtype="bfloat16",
        context_length=int(context_length),
        max_running_requests=1,
        max_total_tokens=int(min(max(4096, context_length * 2), 65536)),
        disable_cuda_graph=True,
        allow_auto_truncate=True,
        enable_return_hidden_states=True,
    )
    t1 = time.time()

    prompt = "Write a one-sentence summary of what speculative decoding is."
    out = engine.generate(
        prompt=prompt,
        sampling_params={"temperature": 0.0, "max_new_tokens": int(max_new_tokens)},
        return_hidden_states=True,
        return_logprob=False,
    )
    t2 = time.time()
    engine.shutdown()

    meta = out.get("meta_info", {}) if isinstance(out, dict) else {}
    hidden = meta.get("hidden_states", None)
    summary = {
        "model_id": model_id,
        "attention_backend": attention_backend,
        "context_length": int(context_length),
        "max_new_tokens": int(max_new_tokens),
        "engine_init_s": float(t1 - t0),
        "first_request_s": float(t2 - t1),
        "out_keys": list(out.keys()) if isinstance(out, dict) else str(type(out)),
        "meta_keys": list(meta.keys()) if isinstance(meta, dict) else str(type(meta)),
        "hidden_summary": _summarize_hidden_blob(hidden),
        "text_preview": (out.get("text", "")[:240] if isinstance(out, dict) else "") or "",
    }
    return summary


@app.local_entrypoint()
def main(
    model_id: str = "openai/gpt-oss-20b",
    attention_backend: str = "fa3",
    context_length: int = 2048,
    max_new_tokens: int = 1,
):
    print(
        spike_remote.remote(
            model_id=model_id,
            attention_backend=attention_backend,
            context_length=int(context_length),
            max_new_tokens=int(max_new_tokens),
        )
    )

