"""
Smoke: in-process SGLang teacher-forward returns GPU hidden states (no IPC).

This is gated behind an env flag in the *training* plan, but this script is the
minimal proof that the in-process path works.

Run (H100; logs to unsloth_logs/):
  mkdir -p harmony/cuda-norm/unsloth_logs
  ts=$(date +%Y%m%d_%H%M%S)
  nohup modal run harmony/cuda-norm/modal/dflash_sglang_inproc_teacher_smoke_20b.py \
    --attention-backend fa3 --context-length 4096 \
    > harmony/cuda-norm/unsloth_logs/dflash_sglang_inproc_smoke_${ts}.log 2>&1 &
"""

from __future__ import annotations

import os
from pathlib import Path

import modal


APP_NAME = "dflash-sglang-inproc-teacher-smoke-20b"
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
        "pip install 'sglang[all]'",
        "pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128",
        "pip install transformers==4.56.2 tokenizers safetensors huggingface-hub hf-transfer",
    )
    .add_local_dir(str(SGLANG_PY_SRC), remote_path="/root/sglang-src", copy=True)
    .add_local_dir(str(SGL_KERNEL_PY_SRC), remote_path="/root/sgl-kernel-src", copy=True)
    .add_local_dir(str(_repo_root / "dflash_gptoss"), remote_path="/root/dflash_gptoss", copy=True)
    .run_commands(
        "cp -rfv /root/sglang-src/* /usr/local/lib/python3.11/site-packages/sglang/",
        "find /usr/local/lib/python3.11/site-packages/sglang -name '__pycache__' -type d -exec rm -rf {} +",
        "cp -rfv /root/sgl-kernel-src/* /usr/local/lib/python3.11/site-packages/sgl_kernel/",
        "find /usr/local/lib/python3.11/site-packages/sgl_kernel -name '__pycache__' -type d -exec rm -rf {} +",
    )
    .env(
        {
            "PYTHONPATH": "/root",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/root/hf_cache",
            "HUGGINGFACE_HUB_CACHE": "/root/hf_cache/hub",
            "TRANSFORMERS_CACHE": "/root/hf_cache/transformers",
            "HF_DATASETS_CACHE": "/root/hf_cache/datasets",
            "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
            # Avoid torch inductor async compile worker shutdown hangs in short smoke runs.
            "TORCHINDUCTOR_DISABLE": "1",
        }
    )
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="H100:1",
    timeout=60 * 60,
    cpu=8.0,
    memory=262144,
    volumes={"/root/hf_cache": hf_cache_volume},
    secrets=_secrets,
)
def smoke_remote(*, attention_backend: str, context_length: int) -> dict:
    import torch
    from transformers import AutoConfig
    from transformers import AutoTokenizer

    from dflash_gptoss.sglang_inproc_teacher import SGLangInprocTeacher
    from dflash_gptoss.utils import build_target_layer_ids

    hf_cache_volume.reload()

    model_id = "openai/gpt-oss-20b"
    cfg = AutoConfig.from_pretrained(model_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    prompt = "Explain what a draft model is for speculative decoding."
    input_ids = tok(prompt, return_tensors="pt").input_ids

    # Match DFlash conditioning: capture multiple layer features and concatenate them.
    # (Uses the same layer selection rule as the DFlash model.)
    num_target_layers = int(getattr(cfg, "num_hidden_layers"))
    num_draft_layers = 4
    layers_to_capture = build_target_layer_ids(num_target_layers, num_draft_layers)

    teacher = SGLangInprocTeacher(
        model_path=model_id,
        attention_backend=attention_backend,
        context_length=int(context_length),
        dtype="bfloat16",
        mem_fraction_static=0.80,
        layers_to_capture=layers_to_capture,
    )
    out = teacher.prefill_hidden_states(input_ids)
    h = out.hidden_states
    # Smoke the lm_head projection path too (without running full sampling).
    # `h` can be concatenated aux-layer features (k*hidden). lm_head expects
    # hidden_size, so just test on the first hidden_size slice.
    logits = teacher.lm_head_logits(h[:, : teacher.hidden_size])
    teacher.close()
    return {
        "input_len": int(out.input_ids.numel()),
        "hidden_shape": list(h.shape),
        "hidden_device": str(h.device),
        "hidden_dtype": str(h.dtype),
        "hidden_mean": float(h.float().mean().item()),
        "logits_shape": list(logits.shape),
        "layers_to_capture_len": len(layers_to_capture),
    }


@app.local_entrypoint()
def main(attention_backend: str = "fa3", context_length: int = 4096):
    print(smoke_remote.remote(attention_backend=attention_backend, context_length=int(context_length)))
