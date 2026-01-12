"""
Modal GPU job: compute embeddings using SGLang Engine (supports true FP8/FP4 backends).

Why this exists (vs HF/torch):
- HF Transformers + ModelOpt/TE in-process often does not hit the "real" FP8/FP4 tensor-core kernels
  for the full stack (attention + GEMM). SGLang + FlashInfer/TRT-LLM backends can.
- SGLang provides a first-class embedding mode (`is_embedding=True`) and returns embeddings through
  `engine.encode()`, matching its internal pooler/normalization logic.

This job reads Parquet candidate shards (`id`, `embed_text` or `text`) and writes embedding shards:
  id, embedding (FixedSizeList<float16>), prompt_tokens, plus passthrough metadata columns.

Important:
- For quality-first selection, always validate embedding geometry vs a BF16 SGLang reference using
  `cpu_eval_embedding_geometry.py` before using any quantized/FP8/FP4 embeddings for clustering.
- HF uploads default to private repos (manager requirement).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import modal

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

APP_NAME = "harmony-qwen-embedding-sglang"

# Optional: prefer reading candidates from a persistent Modal volume mounted at `/root/data`.
# If the directory exists and contains Parquet, `embed_sglang()` will read from it instead of HF.
_candidate_mount_dir = (
    (os.environ.get("CANDIDATE_MOUNT_DIR") or "").strip() or "/root/data/candidates"
)

_secrets = []
_hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if _hf_token:
    _secrets.append(modal.Secret.from_dict({"HF_TOKEN": _hf_token}))

_run_env: dict[str, str | None] = {
    "CANDIDATE_DATASET_ID": os.environ.get("CANDIDATE_DATASET_ID"),
    "CANDIDATE_SUBDIR": os.environ.get("CANDIDATE_SUBDIR"),
    # Optional "mounted candidates" mode (e.g., candidates uploaded into the data volume).
    "CANDIDATE_MOUNT_DIR": _candidate_mount_dir,
    "TEXT_COLUMN": os.environ.get("TEXT_COLUMN"),
    "MODEL_ID": os.environ.get("MODEL_ID"),
    "MODEL_PATH": os.environ.get("MODEL_PATH"),
    "MODEL_PATH_WAIT_S": os.environ.get("MODEL_PATH_WAIT_S"),
    "TRUST_REMOTE_CODE": os.environ.get("TRUST_REMOTE_CODE"),
    "OUT_DIM": os.environ.get("OUT_DIM"),
    "MAX_TOKENS": os.environ.get("MAX_TOKENS"),
    "BATCH_SIZE": os.environ.get("BATCH_SIZE"),
    "MAX_RECORDS": os.environ.get("MAX_RECORDS"),
    "ROWS_PER_SHARD": os.environ.get("ROWS_PER_SHARD"),
    "LOG_EVERY_S": os.environ.get("LOG_EVERY_S"),
    "LOCAL_FILES_ONLY": os.environ.get("LOCAL_FILES_ONLY"),
    "RUN_TAG": os.environ.get("RUN_TAG"),
    "OUT_DATASET_ID": os.environ.get("OUT_DATASET_ID"),
    "OUT_SUBDIR": os.environ.get("OUT_SUBDIR"),
    "FILE_SHARD_INDEX": os.environ.get("FILE_SHARD_INDEX"),
    "FILE_SHARD_COUNT": os.environ.get("FILE_SHARD_COUNT"),
    "BENCH_ONLY": os.environ.get("BENCH_ONLY"),
    "BENCH_STEPS": os.environ.get("BENCH_STEPS"),
    "BENCH_USE_INPUT_IDS": os.environ.get("BENCH_USE_INPUT_IDS"),
    "USE_INPUT_IDS_COLUMN": os.environ.get("USE_INPUT_IDS_COLUMN"),
    "INPUT_IDS_COLUMN": os.environ.get("INPUT_IDS_COLUMN"),
    # SGLang knobs
    "SGLANG_ATTENTION_BACKEND": os.environ.get("SGLANG_ATTENTION_BACKEND"),
    "SGLANG_PREFILL_ATTENTION_BACKEND": os.environ.get("SGLANG_PREFILL_ATTENTION_BACKEND"),
    "SGLANG_DECODE_ATTENTION_BACKEND": os.environ.get("SGLANG_DECODE_ATTENTION_BACKEND"),
    "SGLANG_QUANTIZATION": os.environ.get("SGLANG_QUANTIZATION"),
    "SGLANG_FP8_GEMM_BACKEND": os.environ.get("SGLANG_FP8_GEMM_BACKEND"),
    "SGLANG_DISABLE_CUDA_GRAPH": os.environ.get("SGLANG_DISABLE_CUDA_GRAPH"),
    "SGLANG_LOG_LEVEL": os.environ.get("SGLANG_LOG_LEVEL"),
    "SGLANG_MAX_RUNNING_REQUESTS": os.environ.get("SGLANG_MAX_RUNNING_REQUESTS"),
    "SGLANG_MAX_TOTAL_TOKENS": os.environ.get("SGLANG_MAX_TOTAL_TOKENS"),
    "SGLANG_QUANTIZATION_PARAM_PATH": os.environ.get("SGLANG_QUANTIZATION_PARAM_PATH"),
    # Parallelism
    "SGLANG_TP_SIZE": os.environ.get("SGLANG_TP_SIZE") or os.environ.get("TP_SIZE"),
    # TRTLLM/Flash* backends have important performance modes keyed off KV-cache dtype + page size.
    "SGLANG_KV_CACHE_DTYPE": os.environ.get("SGLANG_KV_CACHE_DTYPE"),
    "SGLANG_PAGE_SIZE": os.environ.get("SGLANG_PAGE_SIZE"),
    "SGLANG_TOKENIZER_WORKERS": os.environ.get("SGLANG_TOKENIZER_WORKERS"),
    "SGLANG_SKIP_TOKENIZER_INIT": os.environ.get("SGLANG_SKIP_TOKENIZER_INIT"),
    "SGLANG_ENABLE_DYNAMIC_BATCH_TOKENIZER": os.environ.get(
        "SGLANG_ENABLE_DYNAMIC_BATCH_TOKENIZER"
    ),
    "SGLANG_DYNAMIC_BATCH_TOKENIZER_BATCH_SIZE": os.environ.get(
        "SGLANG_DYNAMIC_BATCH_TOKENIZER_BATCH_SIZE"
    ),
    "SGLANG_DYNAMIC_BATCH_TOKENIZER_BATCH_TIMEOUT": os.environ.get(
        "SGLANG_DYNAMIC_BATCH_TOKENIZER_BATCH_TIMEOUT"
    ),
    # Advanced perf knobs (read by SGLang envs at import time)
    "SGLANG_ENABLE_JIT_DEEPGEMM": os.environ.get("SGLANG_ENABLE_JIT_DEEPGEMM"),
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": os.environ.get("SGLANG_JIT_DEEPGEMM_PRECOMPILE"),
    "SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS": os.environ.get(
        "SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS"
    ),
    "SGLANG_ENABLE_FLASHINFER_FP8_GEMM": os.environ.get(
        "SGLANG_ENABLE_FLASHINFER_FP8_GEMM"
    ),
    "SGLANG_FLASHINFER_WORKSPACE_SIZE": os.environ.get("SGLANG_FLASHINFER_WORKSPACE_SIZE"),
    "SGLANG_FLASHINFER_FP4_GEMM_BACKEND": os.environ.get("SGLANG_FLASHINFER_FP4_GEMM_BACKEND"),
    # Optional: install a locally built `sgl-kernel` wheel from the wheels volume at runtime.
    "SGL_KERNEL_WHEEL_PATH": os.environ.get("SGL_KERNEL_WHEEL_PATH"),
    "SGL_KERNEL_WHEEL_DIR": os.environ.get("SGL_KERNEL_WHEEL_DIR"),
}

# Modal setup
app = modal.App(APP_NAME)

model_volume = modal.Volume.from_name("qwen-embed-model-weights", create_if_missing=True)
data_volume = modal.Volume.from_name("harmony-embed-data", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("hf-cache-persistent", create_if_missing=True)
flashinfer_cache_volume = modal.Volume.from_name("flashinfer-jit-cache", create_if_missing=True)
sgl_kernel_wheels_volume = modal.Volume.from_name("sgl-kernel-wheels", create_if_missing=True)

GPU_SPEC = os.environ.get("QWEN_EMBED_GPU", "B200:1")

sglang_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu24.04", add_python="3.11")
    .apt_install(
        "git",
        "build-essential",
        "clang",
        "cmake",
        "python3-dev",
        "libnuma-dev",
        "numactl",
        "wget",
        "ninja-build",
    )
    .run_commands(
        "pip install --upgrade pip",
        # sgl-kernel builds via scikit-build-core and requires a recent CMake
        # for policies like CMP0169/CMP0177 (Ubuntu's cmake can be too old).
        "pip install -U scikit-build-core wheel cmake",
        "pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128",
        "pip install hf_transfer",
        "pip install 'transformers>=4.57.1,!=4.57.2' datasets==3.2.0 pyarrow==22.0.0 numpy==2.2.0 msgspec protobuf sentencepiece",
        "pip install 'sglang[all]'",
    )
    # NOTE: We want SM100-capable attention kernels on B200. The default `sgl-kernel`
    # wheel is often compiled for older SMs only, which leads to runtime errors like:
    #   "no kernel image is available for execution on the device"
    # So we rebuild `sgl-kernel` from source with an explicit arch list.
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
            # Build-time arch list for CUDA extensions (covers H100 SM90a + B200 SM100a).
            # This is intentionally broad; if you want a faster image build, use just "10.0a".
            "TORCH_CUDA_ARCH_LIST": "9.0a;10.0a",
        }
    )
)

# Optional: inject a patched SGLang source tree (flashinfer fork) to pick up fp8/fp4 plumbing.
# Keep this best-effort so the job still runs in BF16 even when the local fork isn't present.
_sglang_src_dir = os.path.join(PROJECT_ROOT, "sglang-flashinfer/python/sglang")
if os.path.isdir(_sglang_src_dir):
    sglang_image = sglang_image.add_local_dir(
        _sglang_src_dir,
        remote_path="/root/sglang-src",
        copy=True,
        ignore=["**/__pycache__", "**/__pycache__/**"],
    ).run_commands(
        "cp -rfv /root/sglang-src/* /usr/local/lib/python3.11/site-packages/sglang/",
        "find /usr/local/lib/python3.11/site-packages/sglang -name '__pycache__' -type d -exec rm -rf {} +",
    )
else:
    print(f"[warn] no local sglang override at {_sglang_src_dir}; using pip 'sglang[all]'", flush=True)

# Optional: rebuild `sgl-kernel` from source for SM100 attention/kernels.
# This is expensive (minutes+) and should only be enabled when doing kernel work.
if os.environ.get("REBUILD_SGL_KERNEL", "0") == "1":
    sglang_image = (
        sglang_image.add_local_dir(
            os.path.join(PROJECT_ROOT, "sglang-flashinfer/sgl-kernel"),
            remote_path="/root/sgl-kernel",
            copy=True,
            ignore=[
                "**/__pycache__",
                "**/__pycache__/**",
                ".git",
                ".git/**",
                "**/.git",
                "**/.git/**",
                "build",
                "build/**",
                "dist",
                "dist/**",
            ],
        )
        .run_commands(
            "echo '=== Rebuilding sgl-kernel from source (SM90a/SM100a) ==='",
            "python3 -c \"import torch; print('torch', torch.__version__); print('TORCH_CUDA_ARCH_LIST', __import__('os').environ.get('TORCH_CUDA_ARCH_LIST'))\"",
            "pip uninstall -y sgl-kernel || true",
            # Some transitive deps (e.g., dlpack) require setting this policy floor.
            "CMAKE_ARGS='-DCMAKE_POLICY_VERSION_MINIMUM=3.5' "
            "SKBUILD_CMAKE_ARGS='-DCMAKE_POLICY_VERSION_MINIMUM=3.5' "
            # Reduce build time: only compile the torch extension for Blackwell (SM100a) unless
            # explicitly overridden.
            "CC=gcc CXX=g++ "
            # Prevent OOM in the Modal image builder by limiting compile parallelism.
            "CMAKE_BUILD_PARALLEL_LEVEL=2 "
            "TORCH_CUDA_ARCH_LIST=${SGL_KERNEL_TORCH_CUDA_ARCH_LIST:-10.0a} "
            "pip install -v --no-build-isolation /root/sgl-kernel",
            "python3 -c \"import sgl_kernel; print('sgl_kernel', getattr(sgl_kernel, '__version__', 'unknown')); print('sgl_kernel path', sgl_kernel.__file__)\"",
            "echo '=== sgl-kernel rebuild complete ==='",
        )
    )

# Optional: override FlashInfer from local source, but only when explicitly requested.
# This is needed for fp4/mxfp4 experiments when the pip wheel doesn't include the kernels.
# Default is OFF because building FlashInfer from source is slow and fragile.
_flashinfer_src = os.path.join(PROJECT_ROOT, "flashinfer")
if os.environ.get("USE_LOCAL_FLASHINFER", "0") == "1":
    if not os.path.isdir(_flashinfer_src):
        raise RuntimeError(
            f"USE_LOCAL_FLASHINFER=1 but local dir {_flashinfer_src} does not exist"
        )
    sglang_image = sglang_image.add_local_dir(
        _flashinfer_src,
        remote_path="/root/flashinfer",
        copy=True,
        ignore=[".git", ".git/**", "**/.git", "**/.git/**"],
    ).run_commands(
        "echo '=== Building FlashInfer from local source (optional) ==='",
        "python3 -c \"import flashinfer; print('FlashInfer(pip) version:', getattr(flashinfer, '__version__', 'unknown'))\" || true",
        # Try building cubins (non-fatal).
        "cd /root/flashinfer/flashinfer-cubin && (FLASHINFER_CUDA_ARCH_LIST=9.0a,10.0 pip install . --no-build-isolation -v > /root/flashinfer_cubin_build.log 2>&1 || (tail -n 200 /root/flashinfer_cubin_build.log || true); true)",
        # Try building FlashInfer itself (non-fatal). If it fails, keep the pip wheel.
        "cd /root/flashinfer && (pip install . --no-build-isolation -v > /root/flashinfer_build.log 2>&1 || (tail -n 200 /root/flashinfer_build.log || true); true)",
        "python3 -c \"import flashinfer; print('FlashInfer(final) version:', getattr(flashinfer, '__version__', 'unknown'))\" || true",
        "echo '=== FlashInfer override complete ==='",
    )


@app.function(
    image=sglang_image,
    env=_run_env,
    secrets=_secrets,
    volumes={
        "/models": model_volume,
        "/root/.cache/huggingface": hf_cache_volume,
        "/root/.cache/flashinfer": flashinfer_cache_volume,
    },
    timeout=60 * 60 * 2,
    cpu=8.0,
    memory=65536,
)
def prefetch_model() -> str:
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    model_path_env = (os.environ.get("MODEL_PATH") or "").strip() or None
    if model_path_env:
        local = Path(model_path_env)
        if local.exists():
            print(f"[*] prefetch model: using MODEL_PATH={local}", flush=True)
            return str(local)
        # Important Modal detail: volumes are snapshotted at container start; waiting for another
        # job to commit new files into /models will NOT become visible here. So for /models paths,
        # fail fast and instruct the caller to run exports first (then rerun this job).
        if str(local).startswith("/models/"):
            raise RuntimeError(
                f"MODEL_PATH={local} does not exist at container start. "
                "If this is a ModelOpt export, run the export job first (commit the /models volume), "
                "then start this bench job in a fresh Modal run."
            )

        wait_s = int(os.environ.get("MODEL_PATH_WAIT_S") or "0")
        if wait_s <= 0:
            raise RuntimeError(f"MODEL_PATH={local} does not exist inside container")
        print(f"[*] prefetch model: waiting up to {wait_s}s for MODEL_PATH={local}", flush=True)
        t0 = time.time()
        while (time.time() - t0) < wait_s:
            if local.exists():
                print(f"[ok] prefetch model: found MODEL_PATH={local}", flush=True)
                return str(local)
            time.sleep(10)
        raise RuntimeError(f"timed out waiting for MODEL_PATH={local} (wait_s={wait_s})")

    model_id = (os.environ.get("MODEL_ID") or "").strip() or "Qwen/Qwen3-Embedding-8B"
    local_dir = Path("/models") / model_id.replace("/", "__")
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[*] prefetch model: {model_id} -> {local_dir}", flush=True)
    snapshot_download(
        repo_id=model_id,
        repo_type="model",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    model_volume.commit()
    print(f"[ok] model cached dt={time.time() - t0:.1f}s", flush=True)
    return str(local_dir)


@app.function(
    image=sglang_image,
    env=_run_env,
    secrets=_secrets,
    volumes={"/root/.cache/huggingface": hf_cache_volume},
    timeout=60 * 60,
    cpu=8.0,
    memory=65536,
)
def prefetch_candidates() -> str:
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    dataset_id = (os.environ.get("CANDIDATE_DATASET_ID") or "").strip()
    if not dataset_id:
        raise RuntimeError("Set CANDIDATE_DATASET_ID")
    subdir = (os.environ.get("CANDIDATE_SUBDIR") or "").strip() or None

    allow_patterns = ["**/*.parquet", "README.md"]
    if subdir:
        allow_patterns = [f"{subdir}/*.parquet", f"{subdir}/**/*.parquet", "README.md"]

    t0 = time.time()
    print(f"[*] prefetch dataset: {dataset_id} subdir={subdir!r}", flush=True)
    snap_path = snapshot_download(repo_id=dataset_id, repo_type="dataset", allow_patterns=allow_patterns)
    hf_cache_volume.commit()
    print(f"[ok] dataset cached at {snap_path} dt={time.time() - t0:.1f}s", flush=True)
    return str(snap_path)


@app.function(
    image=sglang_image,
    env=_run_env,
    gpu=GPU_SPEC,
    secrets=_secrets,
    volumes={
        "/models": model_volume,
        "/root/.cache/huggingface": hf_cache_volume,
        "/root/.cache/flashinfer": flashinfer_cache_volume,
	        "/root/sgl_kernel_wheels": sgl_kernel_wheels_volume,
	        "/root/data": data_volume,
	    },
    timeout=60 * 60 * 24,
    cpu=16.0,
    memory=262144,
)
def embed_sglang() -> str:
    import json
    import math
    from dataclasses import dataclass
    from hashlib import sha1
    from typing import Any

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import subprocess
    import torch
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    # Optional: install a prebuilt local `sgl-kernel` wheel (built on CPU) to avoid
    # rebuilding in the Modal image builder and to get SM100a kernels.
    wheel_path_env = (os.environ.get("SGL_KERNEL_WHEEL_PATH") or "").strip() or None
    wheel_dir_env = (os.environ.get("SGL_KERNEL_WHEEL_DIR") or "").strip() or None
    wheel_root = Path("/root/sgl_kernel_wheels")

    wheel_path: Path | None = None
    if wheel_path_env:
        wheel_path = Path(wheel_path_env)
    elif wheel_dir_env:
        wheels = sorted(Path(wheel_dir_env).glob("*.whl"))
        if wheels:
            wheel_path = wheels[-1]
    elif wheel_root.exists():
        # Pick the newest timestamped subdir that contains a wheel.
        subdirs = sorted([p for p in wheel_root.iterdir() if p.is_dir()], key=lambda p: p.name)
        for d in reversed(subdirs):
            wheels = sorted(d.glob("*.whl"))
            if wheels:
                wheel_path = wheels[-1]
                break

    if wheel_path and wheel_path.exists():
        print(f"[*] installing sgl-kernel wheel: {wheel_path}", flush=True)
        subprocess.check_call(["python", "-m", "pip", "install", "-U", str(wheel_path)])
        subprocess.check_call(["python", "-c", "import sgl_kernel; print('sgl_kernel', sgl_kernel.__file__)"])

    import sglang as sgl

    dataset_id = (os.environ.get("CANDIDATE_DATASET_ID") or "").strip()
    subdir = (os.environ.get("CANDIDATE_SUBDIR") or "").strip() or None
    text_col = (os.environ.get("TEXT_COLUMN") or "").strip() or "embed_text"
    input_ids_col = (os.environ.get("INPUT_IDS_COLUMN") or "").strip() or "input_ids"

    model_id = (os.environ.get("MODEL_ID") or "").strip() or "Qwen/Qwen3-Embedding-8B"
    model_path_env = (os.environ.get("MODEL_PATH") or "").strip() or None
    model_path_wait_s = int(os.environ.get("MODEL_PATH_WAIT_S") or "0")
    trust_remote_code = (os.environ.get("TRUST_REMOTE_CODE") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    out_dim = int(os.environ.get("OUT_DIM") or "256")
    max_tokens = int(os.environ.get("MAX_TOKENS") or "1024")
    batch_size = int(os.environ.get("BATCH_SIZE") or "128")
    max_records = int(os.environ.get("MAX_RECORDS") or "0")
    rows_per_shard = int(os.environ.get("ROWS_PER_SHARD") or "200000")
    log_every_s = float(os.environ.get("LOG_EVERY_S") or "10")
    # Quality-first: if SGLang/quantized kernels produce non-finite embeddings, we must fail fast.
    fail_on_nonfinite = (os.environ.get("FAIL_ON_NONFINITE") or "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }
    local_files_only = (os.environ.get("LOCAL_FILES_ONLY") or "1").strip() == "1"
    out_dataset_id = (os.environ.get("OUT_DATASET_ID") or "").strip() or None
    out_subdir = (os.environ.get("OUT_SUBDIR") or "").strip() or None

    file_shard_index = int(os.environ.get("FILE_SHARD_INDEX") or "0")
    file_shard_count = int(os.environ.get("FILE_SHARD_COUNT") or "0")
    bench_only = (os.environ.get("BENCH_ONLY") or "0").strip().lower() in {"1", "true", "yes"}
    bench_steps = int((os.environ.get("BENCH_STEPS") or "").strip() or "50")
    bench_use_input_ids = (os.environ.get("BENCH_USE_INPUT_IDS") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    use_input_ids_column = (os.environ.get("USE_INPUT_IDS_COLUMN") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }

    attn_backend = (os.environ.get("SGLANG_ATTENTION_BACKEND") or "flashinfer").strip()
    prefill_attn_backend = (os.environ.get("SGLANG_PREFILL_ATTENTION_BACKEND") or "").strip() or None
    decode_attn_backend = (os.environ.get("SGLANG_DECODE_ATTENTION_BACKEND") or "").strip() or None
    quantization = (os.environ.get("SGLANG_QUANTIZATION") or "").strip() or None
    fp8_gemm_backend = (os.environ.get("SGLANG_FP8_GEMM_BACKEND") or "").strip() or None
    disable_cuda_graph = (os.environ.get("SGLANG_DISABLE_CUDA_GRAPH") or "0").strip() in {"1", "true", "yes"}
    kv_cache_dtype = (os.environ.get("SGLANG_KV_CACHE_DTYPE") or "").strip() or None
    page_size_env = (os.environ.get("SGLANG_PAGE_SIZE") or "").strip() or None
    fp4_gemm_backend_env = (os.environ.get("SGLANG_FLASHINFER_FP4_GEMM_BACKEND") or "").strip() or None
    max_running_requests_env = (os.environ.get("SGLANG_MAX_RUNNING_REQUESTS") or "").strip()
    max_total_tokens_env = (os.environ.get("SGLANG_MAX_TOTAL_TOKENS") or "").strip()
    quantization_param_path = (os.environ.get("SGLANG_QUANTIZATION_PARAM_PATH") or "").strip() or None
    tp_size_env = (os.environ.get("SGLANG_TP_SIZE") or "").strip()
    tokenizer_workers_env = (os.environ.get("SGLANG_TOKENIZER_WORKERS") or "").strip()
    skip_tokenizer_init_env = (os.environ.get("SGLANG_SKIP_TOKENIZER_INIT") or "").strip()
    enable_dyn_tok_env = (os.environ.get("SGLANG_ENABLE_DYNAMIC_BATCH_TOKENIZER") or "").strip()
    dyn_tok_bs_env = (os.environ.get("SGLANG_DYNAMIC_BATCH_TOKENIZER_BATCH_SIZE") or "").strip()
    dyn_tok_to_env = (os.environ.get("SGLANG_DYNAMIC_BATCH_TOKENIZER_BATCH_TIMEOUT") or "").strip()

    out_tag = (os.environ.get("RUN_TAG") or "").strip()
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_id = out_tag or sha1(
        f"{dataset_id}|{subdir}|{model_id}|sglang|{quantization}|{attn_backend}|{ts}".encode("utf-8")
    ).hexdigest()[:12]
    out_dir = Path("/root/data/embeddings") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Backwards-compat / ergonomics: older runs used "deepgemm" and "flashinfer".
    if fp8_gemm_backend == "deepgemm":
        fp8_gemm_backend = "deep_gemm"
    if fp8_gemm_backend == "flashinfer":
        fp8_gemm_backend = "flashinfer_trtllm"

    allow_patterns = ["**/*.parquet", "README.md"]
    if subdir:
        allow_patterns = [f"{subdir}/*.parquet", f"{subdir}/**/*.parquet", "README.md"]

    parquet_files: list[Path] = []
    total_files = 0
    if bench_only:
        print("[*] BENCH_ONLY=1: skipping candidate dataset download", flush=True)
    else:
        mount_dir = (os.environ.get("CANDIDATE_MOUNT_DIR") or "").strip() or None
        scan_root: Path | None = None
        if mount_dir:
            p = Path(mount_dir)
            if p.exists():
                cand_root = (p / subdir) if subdir else p
                # Only treat this as a "mounted candidates dir" if it actually contains Parquet.
                try:
                    next(cand_root.rglob("*.parquet"))
                except StopIteration:
                    cand_root = None
                if cand_root is not None:
                    scan_root = cand_root
                    print(f"[*] using mounted candidates dir: {scan_root}", flush=True)
        if scan_root is None:
            if not dataset_id:
                raise RuntimeError(
                    "Set CANDIDATE_DATASET_ID (HF dataset) or upload candidates into the data volume "
                    "and set CANDIDATE_MOUNT_DIR."
                )
            print(
                f"[*] snapshot_download (local_files_only={local_files_only}): {dataset_id} subdir={subdir!r}",
                flush=True,
            )
            snap_path = snapshot_download(
                repo_id=dataset_id,
                repo_type="dataset",
                allow_patterns=allow_patterns,
                local_files_only=local_files_only,
            )
            snap = Path(snap_path)
            scan_root = (snap / subdir) if subdir else snap
        parquet_files = sorted(scan_root.rglob("*.parquet"))
        if not parquet_files:
            raise RuntimeError(f"no parquet files under {scan_root}")
        total_files = len(parquet_files)
        if file_shard_count:
            if file_shard_count < 1:
                raise RuntimeError("FILE_SHARD_COUNT must be >= 1")
            if file_shard_index < 0 or file_shard_index >= file_shard_count:
                raise RuntimeError("FILE_SHARD_INDEX must be in [0, FILE_SHARD_COUNT)")
            parquet_files = [
                p for i, p in enumerate(parquet_files) if (i % file_shard_count) == file_shard_index
            ]
            print(
                f"[*] found {total_files} parquet files (shard {file_shard_index}/{file_shard_count} -> {len(parquet_files)})",
                flush=True,
            )
        else:
            print(f"[*] found {total_files} parquet files", flush=True)

    if model_path_env:
        local_model_dir = Path(model_path_env)
    else:
        local_model_dir = Path("/models") / model_id.replace("/", "__")

    def _model_dir_ready(p: Path) -> bool:
        if not p.exists():
            return False
        cfg = p / "config.json"
        if not cfg.exists() or cfg.stat().st_size == 0:
            return False
        try:
            import json as _json

            obj = _json.loads(cfg.read_text(encoding="utf-8"))
        except Exception:
            return False
        arch = obj.get("architectures")
        if not isinstance(arch, list) or not arch or not isinstance(arch[0], str):
            return False
        if not isinstance(obj.get("model_type"), str) or not obj.get("model_type"):
            return False
        # Ensure tokenizer artifacts exist (SGLang needs a tokenizer even for embeddings).
        if not (
            (p / "tokenizer.json").exists()
            or (p / "tokenizer.model").exists()
            or (p / "tokenizer_config.json").exists()
        ):
            return False
        # Require at least one weight artifact to be present.
        if (p / "model.safetensors.index.json").exists():
            return True
        if list(p.glob("*.safetensors")):
            return True
        return False

    if not _model_dir_ready(local_model_dir):
        if model_path_env:
            # Same Modal-volume caveat as in prefetch_model(): waiting for another job’s commit
            # will not update a running container’s /models view. So if MODEL_PATH was provided,
            # require it to be ready at container start.
            raise RuntimeError(
                f"MODEL_PATH={local_model_dir} is not ready (missing/invalid config or weights). "
                "Run the export job first, then start this job in a fresh Modal run."
            )
        # Download into the persistent /models volume so future runs are warm.
        print(f"[*] snapshot_download model -> {local_model_dir} (may take a while)", flush=True)
        snapshot_download(
            repo_id=model_id,
            repo_type="model",
            local_dir=str(local_model_dir),
            local_dir_use_symlinks=False,
        )
        model_volume.commit()
        print("[ok] model downloaded into /models", flush=True)

    log_level = (os.environ.get("SGLANG_LOG_LEVEL") or "error").strip().lower()
    if log_level not in {"error", "warning", "info", "debug"}:
        log_level = "info"

    attn_desc = f"attn={attn_backend}"
    if prefill_attn_backend:
        attn_desc += f" prefill_attn={prefill_attn_backend}"
    if decode_attn_backend:
        attn_desc += f" decode_attn={decode_attn_backend}"
    print(
        f"[*] sglang.Engine(model_path={local_model_dir}, is_embedding=True, {attn_desc}, "
        f"quant={quantization or 'none'} fp8_gemm={fp8_gemm_backend or 'auto'} "
        f"trust_remote_code={trust_remote_code} log_level={log_level})",
        flush=True,
    )
    if fp4_gemm_backend_env:
        print(f"[*] env SGLANG_FLASHINFER_FP4_GEMM_BACKEND={fp4_gemm_backend_env}", flush=True)
    engine_kwargs: dict[str, Any] = {
        "model_path": str(local_model_dir),
        "is_embedding": True,
        "trust_remote_code": trust_remote_code,
        "context_length": max_tokens,
        "allow_auto_truncate": True,
        "attention_backend": attn_backend,
        "prefill_attention_backend": prefill_attn_backend,
        "decode_attention_backend": decode_attn_backend,
        "disable_cuda_graph": disable_cuda_graph,
        # Engine init can involve JIT compilation and/or heavy weight conversion for quantization.
        # Use a generous watchdog to avoid spurious termination during first-run builds.
        "watchdog_timeout": 3600.0,
        "log_level": log_level,
    }
    # Let SGLang choose CUDA-graph batch sizes by default (its heuristics account for GPU capacity
    # and TP/PP). For manual overrides, set SGLANG_CUDA_GRAPH_MAX_BS.
    cuda_graph_max_bs_env = (os.environ.get("SGLANG_CUDA_GRAPH_MAX_BS") or "").strip()
    if cuda_graph_max_bs_env:
        engine_kwargs["cuda_graph_max_bs"] = int(cuda_graph_max_bs_env)
    if tp_size_env:
        tp = int(tp_size_env)
        if tp < 1:
            raise RuntimeError("SGLANG_TP_SIZE must be >= 1")
        engine_kwargs["tp_size"] = tp
        # When using tp_size>1, the engine will map ranks to CUDA devices starting at base_gpu_id.
        # Keep defaults unless the caller overrides via env/model configs.
        engine_kwargs.setdefault("base_gpu_id", 0)
        engine_kwargs.setdefault("gpu_id_step", 1)
    # Qwen/Qwen3-Embedding-8B ships its "matryoshka dimensions" metadata via the
    # SentenceTransformers wrapper files, not HF `config.json`. SGLang's Matryoshka
    # validation currently keys off HF config fields, so we explicitly enable it
    # when the caller requests `dimensions` < hidden_size.
    if out_dim and out_dim > 0:
        engine_kwargs["json_model_override_args"] = '{"is_matryoshka": true}'
    if quantization:
        engine_kwargs["quantization"] = quantization
    if fp8_gemm_backend:
        engine_kwargs["fp8_gemm_runner_backend"] = fp8_gemm_backend
    if quantization_param_path:
        engine_kwargs["quantization_param_path"] = quantization_param_path
    if kv_cache_dtype:
        engine_kwargs["kv_cache_dtype"] = kv_cache_dtype
    if page_size_env:
        engine_kwargs["page_size"] = int(page_size_env)
    if max_running_requests_env:
        engine_kwargs["max_running_requests"] = int(max_running_requests_env)
    if max_total_tokens_env:
        engine_kwargs["max_total_tokens"] = int(max_total_tokens_env)
    if tokenizer_workers_env:
        tw = int(tokenizer_workers_env)
        # NOTE: `sglang.Engine.encode()` (and our embedding job) expect `engine.tokenizer_manager`
        # to provide `generate_request()`. In the current sglang-flashinfer fork, setting
        # tokenizer_worker_num > 1 swaps in a `MultiTokenizerRouter` which does not implement
        # that API, causing hard failures like:
        #   AttributeError("'MultiTokenizerRouter' object has no attribute 'generate_request'")
        #
        # Until SGLang exposes a compatible multi-tokenizer API for `Engine`, we fail-fast here.
        if tw != 1:
            raise RuntimeError(
                "SGLANG_TOKENIZER_WORKERS>1 is not supported for this embedding job (Engine.encode path). "
                "Keep tokenizer_worker_num=1 and use CPU pretokenization + USE_INPUT_IDS_COLUMN=1 for scale."
            )
        engine_kwargs["tokenizer_worker_num"] = tw
    if skip_tokenizer_init_env:
        engine_kwargs["skip_tokenizer_init"] = skip_tokenizer_init_env.lower() in {"1", "true", "yes"}
    if enable_dyn_tok_env:
        engine_kwargs["enable_dynamic_batch_tokenizer"] = enable_dyn_tok_env.lower() in {
            "1",
            "true",
            "yes",
        }
    if dyn_tok_bs_env:
        engine_kwargs["dynamic_batch_tokenizer_batch_size"] = int(dyn_tok_bs_env)
    if dyn_tok_to_env:
        engine_kwargs["dynamic_batch_tokenizer_batch_timeout"] = float(dyn_tok_to_env)

    import threading

    t_engine0 = time.time()
    stop_flag = {"stop": False}

    def _init_heartbeat() -> None:
        while not stop_flag["stop"]:
            dt = time.time() - t_engine0
            print(f"[wait] engine init dt={dt:.1f}s ...", flush=True)
            time.sleep(10.0)

    hb = threading.Thread(target=_init_heartbeat, daemon=True)
    hb.start()
    engine = sgl.Engine(**engine_kwargs)
    stop_flag["stop"] = True
    print(f"[ok] engine ready dt={time.time() - t_engine0:.1f}s", flush=True)

    def _encode_input_ids(batch_ids: list[list[int]]) -> list[dict[str, Any]]:
        from sglang.srt.managers.io_struct import EmbeddingReqInput

        obj = EmbeddingReqInput(input_ids=batch_ids, dimensions=out_dim)
        generator = engine.tokenizer_manager.generate_request(obj, None)
        ret = engine.loop.run_until_complete(generator.__anext__())
        if isinstance(ret, dict):
            return [ret]
        return list(ret)

    if bench_only:
        # Synthetic prompt: long repeated token-ish string; SGLang will truncate to `max_tokens`.
        # This removes dataset I/O and Parquet write time so we can focus on engine throughput.
        prompt = ("tool trace " * (max_tokens * 2)).strip()
        prompts = [prompt] * batch_size

        # Warmup
        t0_warm = time.time()
        batch_ids: list[list[int]] | None = None
        if bench_use_input_ids:
            # Pre-tokenize once, then send input_ids directly (bypasses tokenizer workers).
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(
                str(local_model_dir),
                use_fast=True,
                trust_remote_code=trust_remote_code,
            )
            ids = tok(prompt, add_special_tokens=False, truncation=True, max_length=max_tokens)["input_ids"]
            ids = ids[:max_tokens]
            if not ids:
                ids = [tok.eos_token_id] if tok.eos_token_id is not None else [0]
            batch_ids = [ids] * batch_size
            _ = _encode_input_ids(batch_ids)
        else:
            _ = engine.encode(prompt=prompts, dimensions=out_dim)
        torch.cuda.synchronize()
        print(f"[bench] warmup dt={time.time() - t0_warm:.3f}s", flush=True)

        tok_sum_local = 0
        t_start_local = time.time()
        for i in range(bench_steps):
            if bench_use_input_ids:
                assert batch_ids is not None
                ret_list = _encode_input_ids(batch_ids)
            else:
                ret = engine.encode(prompt=prompts, dimensions=out_dim)
                if isinstance(ret, dict):
                    ret_list = [ret]
                else:
                    ret_list = list(ret)
            for item in ret_list:
                pt = 0
                if isinstance(item, dict):
                    if "prompt_tokens" in item:
                        pt = int(item["prompt_tokens"] or 0)
                    elif "meta_info" in item and isinstance(item["meta_info"], dict):
                        pt = int(item["meta_info"].get("prompt_tokens", 0) or 0)
                tok_sum_local += pt

            if (i + 1) % max(1, bench_steps // 10) == 0:
                dt = time.time() - t_start_local
                print(
                    f"[bench] step={i+1}/{bench_steps} tok/s={tok_sum_local / max(dt, 1e-6):.0f} "
                    f"avg_tok/row={(tok_sum_local / ((i+1) * batch_size)) if batch_size else 0.0:.1f}",
                    flush=True,
                )

        torch.cuda.synchronize()
        dt = time.time() - t_start_local
        tok_s = tok_sum_local / max(dt, 1e-6)
        rows = bench_steps * batch_size
        manifest = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "bench_only": True,
            "bench_steps": int(bench_steps),
            "bench_use_input_ids": bool(bench_use_input_ids),
            "rows_processed": int(rows),
            "tokens_processed": int(tok_sum_local),
            "tok_per_sec": float(tok_s),
            "avg_tokens_per_row": float(tok_sum_local / max(1, rows)),
            "model_id": model_id,
            "out_dim": int(out_dim),
            "max_tokens": int(max_tokens),
            "batch_size": int(batch_size),
            "sglang_attention_backend": attn_backend,
            "sglang_quantization": quantization or "",
            "sglang_fp8_gemm_backend": fp8_gemm_backend or "",
            "sglang_fp4_gemm_backend": fp4_gemm_backend_env or "",
            "sglang_kv_cache_dtype": kv_cache_dtype or "",
            "sglang_quantization_param_path": quantization_param_path or "",
        }
        try:
            manifest["cuda_max_mem_allocated_gb"] = float(torch.cuda.max_memory_allocated() / (1024**3))
            manifest["cuda_max_mem_reserved_gb"] = float(torch.cuda.max_memory_reserved() / (1024**3))
        except Exception:
            pass
        (out_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"[ok] wrote {out_dir}/run_manifest.json", flush=True)
        data_volume.commit()
        hf_cache_volume.commit()
        flashinfer_cache_volume.commit()
        model_volume.commit()
        return str(out_dir)

    @dataclass
    class ShardWriter:
        out_dir: Path
        rows_per_shard: int
        compression: str = "zstd"
        shard_index: int = 0
        rows_in_shard: int = 0
        schema: pa.Schema | None = None
        _writer: pq.ParquetWriter | None = None

        def _path(self) -> Path:
            return self.out_dir / f"part-{self.shard_index:05d}.parquet"

        def _open_if_needed(self, schema: pa.Schema) -> None:
            if self._writer is not None:
                return
            self.schema = schema
            self._writer = pq.ParquetWriter(str(self._path()), schema, compression=self.compression)

        def _close(self) -> None:
            if self._writer is None:
                return
            path = self._path()
            self._writer.close()
            self._writer = None
            print(f"[write] {path} rows={self.rows_in_shard}", flush=True)
            self.shard_index += 1
            self.rows_in_shard = 0

        def write_table(self, table: pa.Table) -> None:
            if table.num_rows == 0:
                return
            if self.schema is None:
                self._open_if_needed(table.schema)
            assert self.schema is not None
            if table.schema != self.schema:
                table = table.cast(self.schema)
            remaining = table
            while remaining.num_rows:
                if self._writer is None:
                    self._open_if_needed(self.schema)
                assert self._writer is not None
                cap = self.rows_per_shard - self.rows_in_shard
                if cap <= 0:
                    self._close()
                    continue
                if remaining.num_rows <= cap:
                    self._writer.write_table(remaining)
                    self.rows_in_shard += remaining.num_rows
                    if self.rows_in_shard >= self.rows_per_shard:
                        self._close()
                    break
                head = remaining.slice(0, cap)
                self._writer.write_table(head)
                self.rows_in_shard += head.num_rows
                self._close()
                remaining = remaining.slice(cap)

        def flush(self) -> None:
            self._close()

    writer = ShardWriter(out_dir=out_dir, rows_per_shard=rows_per_shard)

    processed = 0
    tok_sum = 0
    tok_padded_sum = 0
    tok_lens_sample: list[int] = []
    tok_lens_sample_cap = 200_000
    t_start = time.time()
    last_log = time.time()
    t_read_s = 0.0
    t_encode_s = 0.0
    t_pack_s = 0.0
    t_write_s = 0.0

    for file_i, pf in enumerate(parquet_files, start=1):
        file_reported = False
        t0_read = time.time()
        parquet = pq.ParquetFile(pf)
        t_read_s += time.time() - t0_read
        # IMPORTANT: use the Arrow schema (top-level columns). Parquet "leaf" names for list
        # columns include only the element field (e.g. "element"), which would incorrectly
        # hide the presence of a list column like "input_ids".
        cols = set(parquet.schema_arrow.names)
        if "id" not in cols:
            raise RuntimeError(f"missing required column 'id' in {pf}")
        use_text_col = text_col if text_col in cols else "text"
        if use_input_ids_column:
            if input_ids_col not in cols:
                raise RuntimeError(
                    f"USE_INPUT_IDS_COLUMN=1 but missing required column {input_ids_col!r} in {pf}"
                )
        else:
            if use_text_col not in cols:
                raise RuntimeError(f"missing required text column {text_col!r} (or 'text') in {pf}")

        passthrough_cols = [
            c
            for c in [
                "dataset",
                "split",
                "loss_mode",
                "meta_domain",
                "meta_difficulty_bin",
                "meta_correctness",
                "quality_has_tool",
                "quality_valid_tool_schema",
                "stats_embed_word_count",
                "mix_group",
            ]
            if c in cols
        ]
        if use_input_ids_column:
            read_cols = ["id", input_ids_col] + passthrough_cols
        else:
            read_cols = ["id", use_text_col] + passthrough_cols

        print(f"[file] idx={file_i}/{len(parquet_files)} path={pf}", flush=True)
        for batch in parquet.iter_batches(columns=read_cols, batch_size=max(1024, batch_size * 4)):
            t0_read = time.time()
            table = pa.Table.from_batches([batch])
            t_read_s += time.time() - t0_read
            ids = table["id"].to_pylist()
            texts = table[use_text_col].to_pylist() if not use_input_ids_column else None
            input_ids = table[input_ids_col].to_pylist() if use_input_ids_column else None

            n_rows = len(ids)
            for start in range(0, n_rows, batch_size):
                sub_ids = ids[start : start + batch_size]
                if use_input_ids_column:
                    sub_input_ids = input_ids[start : start + batch_size]
                    sub_texts = None
                else:
                    sub_input_ids = None
                    sub_texts = [
                        (t if isinstance(t, str) else "").strip()
                        for t in texts[start : start + batch_size]
                    ]

                # SGLang returns either a dict (single) or list[dict] (batch) depending on input.
                t0_encode = time.time()
                if use_input_ids_column:
                    ret_list = _encode_input_ids(sub_input_ids)  # type: ignore[arg-type]
                    ret = ret_list
                else:
                    ret = engine.encode(prompt=sub_texts, dimensions=out_dim)
                t_encode_s += time.time() - t0_encode
                if isinstance(ret, dict):
                    ret_list = [ret]
                else:
                    ret_list = list(ret)
                if len(ret_list) != len(sub_ids):
                    raise RuntimeError(f"sglang returned {len(ret_list)} embeddings for batch size {len(sub_ids)}")

                embs: list[list[float]] = []
                prompt_tokens: list[int] = []
                for item in ret_list:
                    emb = item.get("embedding") if isinstance(item, dict) else None
                    if emb is None:
                        raise RuntimeError(f"sglang encode output missing 'embedding' field: keys={list(item.keys()) if isinstance(item, dict) else type(item)}")
                    embs.append(emb)
                    pt = 0
                    if isinstance(item, dict):
                        # Both formats appear in the wild (depending on code path).
                        if "prompt_tokens" in item:
                            pt = int(item["prompt_tokens"] or 0)
                        elif "meta_info" in item and isinstance(item["meta_info"], dict):
                            pt = int(item["meta_info"].get("prompt_tokens", 0) or 0)
                    prompt_tokens.append(pt)

                # Metrics
                tok_sum += int(sum(prompt_tokens))
                if prompt_tokens and not file_reported:
                    lens_arr = np.array(prompt_tokens, dtype=np.int32)
                    est_p50 = float(np.quantile(lens_arr, 0.50))
                    est_p90 = float(np.quantile(lens_arr, 0.90))
                    est_p99 = float(np.quantile(lens_arr, 0.99))
                    print(f"[file_est] idx={file_i}/{len(parquet_files)} p50={est_p50:.0f} p90={est_p90:.0f} p99={est_p99:.0f} (first batch)", flush=True)
                    file_reported = True
                if len(tok_lens_sample) < tok_lens_sample_cap:
                    tok_lens_sample.extend(int(x) for x in prompt_tokens[: (tok_lens_sample_cap - len(tok_lens_sample))])

                # Padding multiplier is not directly observable (SGLang handles tokenization internally).
                # We leave tok_padded_sum as 0 and report only token-based throughput.

                t0_pack = time.time()
                emb_np = np.asarray(embs, dtype=np.float16)
                if emb_np.ndim != 2 or emb_np.shape[1] != out_dim:
                    raise RuntimeError(f"unexpected embedding shape {emb_np.shape}, expected (*,{out_dim})")
                if fail_on_nonfinite:
                    finite = np.isfinite(emb_np).all(axis=1)
                    if not finite.all():
                        bad = int((~finite).sum())
                        raise RuntimeError(
                            f"non-finite embeddings detected (bad_rows={bad}/{len(finite)}) "
                            f"quantization={quantization!r} model={model_id!r} file={pf}"
                        )
                flat = emb_np.reshape(-1)
                vec_values = pa.array(flat, type=pa.float16())
                vec = pa.FixedSizeListArray.from_arrays(vec_values, out_dim)

                out_cols: dict[str, Any] = {
                    "id": pa.array([str(x) for x in sub_ids], type=pa.string()),
                    "model_id": pa.array([model_id] * len(sub_ids), type=pa.string()),
                    "out_dim": pa.array([out_dim] * len(sub_ids), type=pa.int32()),
                    "max_tokens": pa.array([max_tokens] * len(sub_ids), type=pa.int32()),
                    "embedding": vec,
                    "prompt_tokens": pa.array(prompt_tokens, type=pa.int32()),
                    "sglang_attention_backend": pa.array([attn_backend] * len(sub_ids), type=pa.string()),
                    "sglang_quantization": pa.array([quantization or ""] * len(sub_ids), type=pa.string()),
                }
                for c in passthrough_cols:
                    out_cols[c] = table[c].slice(start, len(sub_ids))
                t_pack_s += time.time() - t0_pack
                t0_write = time.time()
                writer.write_table(pa.table(out_cols))
                t_write_s += time.time() - t0_write

                processed += len(sub_ids)
                if max_records and processed >= max_records:
                    break

                now = time.time()
                if now - last_log >= log_every_s:
                    dt = now - t_start
                    tok_s = tok_sum / max(dt, 1e-6)
                    rows_s = processed / max(dt, 1e-6)
                    avg_tok = (tok_sum / processed) if processed else 0.0
                    stage = (
                        f"read={t_read_s:.1f}s encode={t_encode_s:.1f}s "
                        f"pack={t_pack_s:.1f}s write={t_write_s:.1f}s"
                    )
                    print(
                        f"[prog] rows={processed} rows/s={rows_s:.2f} tok/s={tok_s:.0f} "
                        f"avg_tok/row={avg_tok:.1f} {stage}",
                        flush=True,
                    )
                    last_log = now

            if max_records and processed >= max_records:
                break
        if max_records and processed >= max_records:
            break

    writer.flush()

    tok_arr = np.array(tok_lens_sample, dtype=np.int32) if tok_lens_sample else np.array([], dtype=np.int32)
    p50 = float(np.quantile(tok_arr, 0.50)) if tok_arr.size else math.nan
    p90 = float(np.quantile(tok_arr, 0.90)) if tok_arr.size else math.nan
    p99 = float(np.quantile(tok_arr, 0.99)) if tok_arr.size else math.nan

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "dataset_id": dataset_id,
        "subdir": subdir,
        "text_column": text_col,
        "use_input_ids_column": bool(use_input_ids_column),
        "input_ids_column": input_ids_col if use_input_ids_column else "",
        "model_id": model_id,
        "out_dim": out_dim,
        "max_tokens": max_tokens,
        "batch_size": batch_size,
        "processed": processed,
        "tokens_processed": tok_sum,
        "tokens_padded_processed": tok_padded_sum,
        "padding_multiplier": (tok_padded_sum / tok_sum) if tok_sum else 0.0,
        "avg_tokens_per_row": (tok_sum / processed) if processed else 0.0,
        "tok_len_sample_n": int(tok_arr.size),
        "tok_len_p50": p50,
        "tok_len_p90": p90,
        "tok_len_p99": p99,
        "sglang_attention_backend": attn_backend,
        "sglang_quantization": quantization or "",
        "sglang_fp8_gemm_backend": fp8_gemm_backend or "",
        "sglang_fp4_gemm_backend": fp4_gemm_backend_env or "",
        "disable_cuda_graph": bool(disable_cuda_graph),
        "elapsed_s": time.time() - t_start,
    }
    try:
        manifest["cuda_max_mem_allocated_gb"] = float(torch.cuda.max_memory_allocated() / (1024**3))
        manifest["cuda_max_mem_reserved_gb"] = float(torch.cuda.max_memory_reserved() / (1024**3))
    except Exception:
        pass
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_dir}/run_manifest.json", flush=True)

    if out_dataset_id:
        from huggingface_hub import HfApi

        api = HfApi()
        # Manager requirement: always keep HF outputs private.
        # Note: create_repo(exist_ok=True) will not flip an existing public repo,
        # so we also force visibility when possible.
        api.create_repo(repo_id=out_dataset_id, repo_type="dataset", private=True, exist_ok=True)
        try:
            if hasattr(api, "update_repo_settings"):
                api.update_repo_settings(repo_id=out_dataset_id, repo_type="dataset", private=True)
            else:
                api.update_repo_visibility(repo_id=out_dataset_id, repo_type="dataset", private=True)
        except Exception as e:
            print(f"[warn] could not force HF repo private for {out_dataset_id}: {e}", flush=True)
        path_in_repo = out_subdir.rstrip("/") if out_subdir else f"embeddings/{run_id}"
        print(f"[*] upload_folder -> {out_dataset_id}:{path_in_repo}", flush=True)
        api.upload_folder(
            repo_id=out_dataset_id,
            repo_type="dataset",
            folder_path=str(out_dir),
            path_in_repo=path_in_repo,
            commit_message=f"Add SGLang embeddings run {run_id}",
        )
        print("[ok] upload complete", flush=True)

    data_volume.commit()
    hf_cache_volume.commit()
    flashinfer_cache_volume.commit()
    model_volume.commit()
    return str(out_dir)


@app.local_entrypoint()
def main() -> None:
    # IMPORTANT: avoid burning GPU time on downloads.
    #
    # This local entrypoint runs on the caller machine (CPU) and can prefetch:
    # - model weights into the persistent `/models` volume
    # - candidate parquet shards into the persistent HF cache volume
    #
    # Then the GPU function (`embed_sglang`) can run with `LOCAL_FILES_ONLY=1`.
    bench_only = (os.environ.get("BENCH_ONLY") or "0").strip().lower() in {"1", "true", "yes"}
    dataset_id = (os.environ.get("CANDIDATE_DATASET_ID") or "").strip()
    local_dir = (os.environ.get("CANDIDATE_LOCAL_DIR") or "").strip()
    subdir = (os.environ.get("CANDIDATE_SUBDIR") or "").strip()

    prefetch_only = (os.environ.get("PREFETCH_ONLY") or "0").strip().lower() in {"1", "true", "yes"}
    skip_prefetch = (os.environ.get("SKIP_PREFETCH") or "0").strip().lower() in {"1", "true", "yes"}

    # Local candidates mode (developer convenience):
    # Modal 1.3.x `App.function` does not support runtime mounts, so we can't directly mount a local
    # directory into the container. Instead, upload the local directory into the persistent data
    # volume using `modal volume put`, and point CANDIDATE_MOUNT_DIR/CANDIDATE_SUBDIR at it.
    if local_dir and not dataset_id:
        if not subdir:
            raise RuntimeError(
                "CANDIDATE_LOCAL_DIR is set but CANDIDATE_SUBDIR is empty. "
                "Set CANDIDATE_SUBDIR to the destination folder name inside the data volume "
                "(e.g. CANDIDATE_SUBDIR=local_smoke_v1)."
            )
        mount_dir = (os.environ.get("CANDIDATE_MOUNT_DIR") or "").strip() or _candidate_mount_dir
        if not mount_dir.startswith("/root/data/"):
            raise RuntimeError(
                "CANDIDATE_LOCAL_DIR mode requires CANDIDATE_MOUNT_DIR under /root/data "
                f"(got {mount_dir!r})."
            )
        rel = mount_dir[len("/root/data/") :].strip("/")
        remote_path = f"/{rel}/{subdir}".rstrip("/") + "/"
        vol_name = "harmony-embed-data"
        print(f"[*] uploading local candidates -> volume {vol_name}:{remote_path}", flush=True)
        import subprocess

        subprocess.check_call(["modal", "volume", "put", "--force", vol_name, local_dir, remote_path])
        print("[ok] local candidates uploaded", flush=True)

    if not skip_prefetch:
        prefetch_model.remote()
        # Skip HF snapshot prefetch when using local mount mode.
        if (not local_dir) and dataset_id and (prefetch_only or not bench_only):
            prefetch_candidates.remote()

    if prefetch_only:
        print("[ok] PREFETCH_ONLY=1: completed prefetch; skipping GPU embed", flush=True)
        return

    # Default behavior: run synchronously so stdout streams into the local nohup log file.
    # If you explicitly want fire-and-forget, set MODAL_SPAWN=1.
    spawn = (os.environ.get("MODAL_SPAWN") or "0").strip().lower() in {"1", "true", "yes"}
    if spawn:
        call = embed_sglang.spawn()
        print(f"spawned: {call}")
        return
    print("[*] launching embed_sglang (GPU)…", flush=True)
    out_dir = embed_sglang.remote()
    print(out_dir)
