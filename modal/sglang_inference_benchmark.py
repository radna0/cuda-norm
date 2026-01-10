"""
SGLang Performance Benchmark on Modal H100
Tests BF16, FP8, and FP4 (E2M1) KV cache
Builds FlashInfer from source with FP4 KV cache support patches
Outputs results and errors to PROJECT_ROOT/logs/ folder
"""

import modal
import time
import os
import sys
import math
import inspect
from pathlib import Path

# Project root path (absolute)
# NOTE: This file lives in `modal/`, so repo root is one directory up.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# GPT-OSS is trained on the Harmony response format and will not behave correctly
# if you feed raw plain-text prompts. By default, format prompts via the model's
# chat template (which applies Harmony).
USE_HARMONY = os.getenv("SGLANG_BENCH_USE_HARMONY", "1") == "1"
SYSTEM_PROMPT = os.getenv(
    "SGLANG_BENCH_SYSTEM_PROMPT",
    "You are a helpful assistant.\nReasoning: medium",
)
USER_PROMPT = os.getenv(
    "SGLANG_BENCH_USER_PROMPT",
    "Explain Einstein's theory of special relativity in detail.",
)

# Optional correctness checks (runs on Modal; writes details to the remote log).
RUN_PPL = os.getenv("SGLANG_BENCH_RUN_PPL", "0") == "1"
PPL_DATASET_REPO_ID = os.getenv(
    "SGLANG_BENCH_PPL_DATASET_REPO_ID", "radna0/nemotron-math-v2-harmony-tools"
)
PPL_DATASET_SPLIT = os.getenv("SGLANG_BENCH_PPL_DATASET_SPLIT", "high_part00")
PPL_DATASET_MAX_ROWS = int(os.getenv("SGLANG_BENCH_PPL_DATASET_MAX_ROWS", "20000"))
PPL_DATASET_FILE = os.getenv(
    "SGLANG_BENCH_PPL_DATASET_FILE",
    f"/data/{PPL_DATASET_REPO_ID.replace('/', '__')}__{PPL_DATASET_SPLIT}.jsonl",
)
PPL_SEQ_LEN = int(os.getenv("SGLANG_BENCH_PPL_SEQ_LEN", "512"))
PPL_NUM_SAMPLES = int(os.getenv("SGLANG_BENCH_PPL_NUM_SAMPLES", "16"))
PPL_BATCH_SIZE = int(os.getenv("SGLANG_BENCH_PPL_BATCH_SIZE", "1"))
DISABLE_CUDA_GRAPH = os.getenv("SGLANG_BENCH_DISABLE_CUDA_GRAPH", "0") == "1"

# Extra configs (opt-in to keep iteration time reasonable)
RUN_FP8 = os.getenv("SGLANG_BENCH_RUN_FP8", "0") == "1"
RUN_FP4 = os.getenv("SGLANG_BENCH_RUN_FP4", "0") == "1"
RUN_BF16 = os.getenv("SGLANG_BENCH_RUN_BF16", "1") == "1"
RUN_FP8_FA3 = os.getenv("SGLANG_BENCH_RUN_FP8_FA3", "0") == "1"

BATCH_SIZE = int(os.getenv("SGLANG_BENCH_BATCH_SIZE", "1"))
MAX_NEW_TOKENS = int(os.getenv("SGLANG_BENCH_MAX_NEW_TOKENS", "256"))

# Optional speculative decoding (EAGLE3) benchmarks.
RUN_EAGLE3 = os.getenv("SGLANG_BENCH_RUN_EAGLE3", "0") == "1"
EAGLE3_STEPS = int(os.getenv("SGLANG_BENCH_EAGLE3_STEPS", "4"))
EAGLE3_TOPK = int(os.getenv("SGLANG_BENCH_EAGLE3_TOPK", "1"))
# For topk>1, explicitly capping total draft tokens is usually required.
EAGLE3_NUM_DRAFT_TOKENS = os.getenv("SGLANG_BENCH_EAGLE3_NUM_DRAFT_TOKENS", "")
EAGLE3_NUM_DRAFT_TOKENS = (
    int(EAGLE3_NUM_DRAFT_TOKENS) if EAGLE3_NUM_DRAFT_TOKENS else None
)

# Modal setup
app = modal.App("sglang-flashinfer-fp4-benchmark")
log_volume = modal.Volume.from_name("sglang-benchmark-logs", create_if_missing=True)
model_volume = modal.Volume.from_name("gpt-oss-model-weights", create_if_missing=True)
data_volume = modal.Volume.from_name("gpt-oss-harmony-tools", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
# Persist FlashInfer JIT artifacts across runs (e.g. /root/.cache/flashinfer/0.6.0/90a/*).
# This avoids re-compiling heavy SM90 kernels on every Modal invocation.
flashinfer_cache_volume = modal.Volume.from_name(
    "flashinfer-jit-cache", create_if_missing=True
)

# Allow multi-GPU runs by setting `SGLANG_BENCH_GPU=H100:4` (or similar).
GPU_SPEC = os.getenv("SGLANG_BENCH_GPU", "H100")

# Create image with SGLang and FlashInfer built from source
sglang_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04",
        add_python="3.11",
    )
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
        "pip install 'sglang[all]'",
        # Install build dependencies for FlashInfer
        "pip install ninja cmake wheel setuptools packaging",
        "pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128",
        "pip install hf_transfer",
    )
    .run_commands(
        "pip install 'transformers>=4.57.1,!=4.57.2' sentence-transformers numpy==2.2.0 pandas polars datasets==3.2.0 scipy 'openai-harmony>=0.0.8' sentencepiece protobuf msgspec",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Inject patched FlashInfer source
    .add_local_dir(
        os.path.join(PROJECT_ROOT, "flashinfer"),
        remote_path="/root/flashinfer",
        copy=True,
        ignore=[".git", ".git/**", "**/.git", "**/.git/**"],
    )
    # Build FlashInfer from source with verbose output
    .run_commands(
        "echo '=== Building FlashInfer from source with FP4 KV cache support ===' ",
        # Ensure no stale installs shadow the custom build.
        "pip uninstall -y flashinfer flashinfer-python flashinfer-cubin || true",
        # Download/install only the SM90 subset of cubins to keep builds reasonable.
        # NOTE: FlashInfer expects the "major.minor[a]" form (e.g. "9.0a"), not "90".
        "cd /root/flashinfer/flashinfer-cubin && FLASHINFER_CUDA_ARCH_LIST=9.0a pip install . --no-build-isolation -v > /root/flashinfer_cubin_build.log 2>&1",
        "cd /root/flashinfer && pip install . --no-build-isolation -v > /root/flashinfer_build.log 2>&1",
        "echo '=== FlashInfer build complete ===' ",
        # Verify the installation
        "python3 -c \"import flashinfer; print('FlashInfer version:', getattr(flashinfer, '__version__', 'unknown')); "
        "print('FlashInfer exports:', [x for x in dir(flashinfer) if 'fp4' in x or 'mxfp8' in x])\"",
    )
    # Inject patched SGLang source directly into site-packages
    .add_local_dir(
        os.path.join(PROJECT_ROOT, "sglang-flashinfer/python/sglang"),
        remote_path="/root/sglang-src",
        copy=True,
        ignore=[
            "**/__pycache__",
            "**/__pycache__/**",
        ],
    )
    # Override sgl_kernel Python stubs to avoid importing FA4/Cute-DSL on Hopper.
    # The pip wheel currently imports FA4 unconditionally and crashes on H100 with:
    #   ValueError: evaluate_polynomial() requires a code object with 1 free vars, not 0
    .add_local_dir(
        os.path.join(PROJECT_ROOT, "sglang-flashinfer/sgl-kernel/python/sgl_kernel"),
        remote_path="/root/sgl-kernel-src",
        copy=True,
        ignore=[
            "**/__pycache__",
            "**/__pycache__/**",
        ],
    )
    .run_commands(
        "cp -rfv /root/sglang-src/* /usr/local/lib/python3.11/site-packages/sglang/",
        "find /usr/local/lib/python3.11/site-packages/sglang -name '__pycache__' -type d -exec rm -rf {} +",
        "cp -rfv /root/sgl-kernel-src/* /usr/local/lib/python3.11/site-packages/sgl_kernel/",
        "find /usr/local/lib/python3.11/site-packages/sgl_kernel -name '__pycache__' -type d -exec rm -rf {} +",
    )
)

MODEL_PATH = os.getenv("SGLANG_BENCH_MODEL_PATH", "/models/openai/gpt-oss-20b")
DRAFT_MODEL_PATH = os.getenv("SGLANG_BENCH_DRAFT_MODEL_PATH", MODEL_PATH)
PROMPT = USER_PROMPT
TP_SIZE = int(os.getenv("SGLANG_BENCH_TP_SIZE", "1"))


@app.function(
    image=sglang_image,
    timeout=4 * 60 * 60,
    memory=32000,
    volumes={
        "/models": model_volume,
        "/data": data_volume,
        "/root/.cache/huggingface": hf_cache_volume,
        "/logs": log_volume,
    },
    env={
        "HF_HUB_ENABLE_HF_TRANSFER": os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "1"),
        # Pass through auth tokens if present.
        "HF_TOKEN": os.getenv("HF_TOKEN", ""),
        "HUGGINGFACE_HUB_TOKEN": os.getenv("HUGGINGFACE_HUB_TOKEN", ""),
    },
    retries=0,
)
def ensure_assets(
    model_id_or_path: str,
    draft_model_id_or_path: str | None,
    dataset_repo_id: str | None,
    dataset_split: str | None,
    dataset_out_file: str | None,
    dataset_max_rows: int = 20000,
):
    import json

    from huggingface_hub import snapshot_download
    from datasets import load_dataset

    def _infer_repo_id(maybe_path_or_id: str) -> str | None:
        if not maybe_path_or_id:
            return None
        if "/" in maybe_path_or_id and not maybe_path_or_id.startswith("/"):
            return maybe_path_or_id
        if maybe_path_or_id.startswith("/models/"):
            parts = Path(maybe_path_or_id).parts
            # ('/', 'models', '<org>', '<model>', ...)
            if len(parts) >= 4:
                return f"{parts[2]}/{parts[3]}"
        return None

    def _infer_local_dir(maybe_path_or_id: str) -> str:
        if maybe_path_or_id.startswith("/models/"):
            return maybe_path_or_id
        repo_id = _infer_repo_id(maybe_path_or_id)
        if repo_id is None:
            raise ValueError(
                f"Cannot infer HuggingFace repo_id from model_id_or_path={maybe_path_or_id!r}. "
                "Use a HF repo id like 'openai/gpt-oss-20b' or a /models/<org>/<model> path."
            )
        return f"/models/{repo_id}"

    def _has_model_files(local_dir: str) -> bool:
        if not os.path.isdir(local_dir):
            return False
        if not os.path.exists(os.path.join(local_dir, "config.json")):
            return False
        if not (
            os.path.exists(os.path.join(local_dir, "model.safetensors.index.json"))
            or any(name.endswith(".safetensors") for name in os.listdir(local_dir))
        ):
            return False
        return True

    def _download_model(model_id_or_path: str) -> str:
        local_dir = _infer_local_dir(model_id_or_path)
        if _has_model_files(local_dir):
            print(f"[ASSETS] Model already present: {local_dir}")
            return local_dir

        repo_id = _infer_repo_id(model_id_or_path)
        if repo_id is None:
            raise ValueError(
                f"Cannot infer repo_id for model download from {model_id_or_path!r}"
            )

        os.makedirs(local_dir, exist_ok=True)
        print(f"[ASSETS] Downloading model {repo_id} -> {local_dir}")

        sd_sig = inspect.signature(snapshot_download)
        kwargs = dict(
            repo_id=repo_id,
            local_dir=local_dir,
            resume_download=True,
            ignore_patterns=["metal/**", "original/**"],
        )
        if "local_dir_use_symlinks" in sd_sig.parameters:
            # Avoid duplicating huge weights into both hub cache and /models.
            kwargs["local_dir_use_symlinks"] = True
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or None
        if token and "token" in sd_sig.parameters:
            kwargs["token"] = token
        snapshot_download(**kwargs)

        if not _has_model_files(local_dir):
            raise RuntimeError(f"Model download incomplete: {local_dir}")

        model_volume.commit()
        hf_cache_volume.commit()
        print("[ASSETS] Model volume committed.")
        return local_dir

    # Download main + draft model if needed.
    _download_model(model_id_or_path)
    if draft_model_id_or_path and draft_model_id_or_path != model_id_or_path:
        _download_model(draft_model_id_or_path)

    # Download dataset slice only if requested (for PPL).
    if dataset_repo_id and dataset_split and dataset_out_file:
        if os.path.exists(dataset_out_file) and os.path.getsize(dataset_out_file) > 0:
            print(f"[ASSETS] Dataset already present: {dataset_out_file}")
        else:
            os.makedirs(os.path.dirname(dataset_out_file), exist_ok=True)
            print(
                f"[ASSETS] Streaming dataset {dataset_repo_id}:{dataset_split} -> {dataset_out_file}"
            )
            ds = load_dataset(dataset_repo_id, split=dataset_split, streaming=True)
            wrote = 0
            with open(dataset_out_file, "w", encoding="utf-8") as wf:
                for ex in ds:
                    text = ex.get("text")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    wf.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    wrote += 1
                    if wrote >= dataset_max_rows:
                        break
            if wrote == 0:
                raise RuntimeError(
                    f"Dataset download wrote 0 rows: {dataset_repo_id}:{dataset_split}"
                )
            data_volume.commit()
            hf_cache_volume.commit()
            print(f"[ASSETS] Dataset committed (rows={wrote}).")

    return {"ok": True}


@app.function(
    image=sglang_image,
    gpu=GPU_SPEC,
    memory=128000,
    volumes={
        "/models": model_volume,
        "/logs": log_volume,
        "/data": data_volume,
        "/root/.cache/huggingface": hf_cache_volume,
        # Mount a persistent volume for FlashInfer JIT cache. We cannot mount onto
        # "/root/.cache/flashinfer" directly because it may be non-empty in the
        # container image. Instead, mount an empty base dir and point FlashInfer's
        # workspace base there via FLASHINFER_WORKSPACE_BASE.
        "/flashinfer_cache_base": flashinfer_cache_volume,
    },
    timeout=2400,
    env={
        "FLASHINFER_WORKSPACE_BASE": "/flashinfer_cache_base",
        # Avoid FlashInfer JIT file locks on network volumes (NFS) by placing lock
        # files and aggregate ninja manifests in local tmpfs.
        "FLASHINFER_JIT_TMPDIR": "/dev/shm/flashinfer_jit_tmp",
        "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
        "SGLANG_FLASHINFER_FORCE_PAGE1": os.getenv(
            "SGLANG_FLASHINFER_FORCE_PAGE1", "0"
        ),
        "SGLANG_DEBUG_FLASHINFER_ALLOC": os.getenv(
            "SGLANG_DEBUG_FLASHINFER_ALLOC", "0"
        ),
        "SGLANG_DEBUG_FLASHINFER_BACKEND": os.getenv(
            "SGLANG_DEBUG_FLASHINFER_BACKEND", "0"
        ),
        "SGLANG_FLASHINFER_DISABLE_TENSOR_CORES": os.getenv(
            "SGLANG_FLASHINFER_DISABLE_TENSOR_CORES", "0"
        ),
        "SGLANG_BENCH_USE_HARMONY": os.getenv("SGLANG_BENCH_USE_HARMONY", "1"),
        "SGLANG_BENCH_SYSTEM_PROMPT": os.getenv(
            "SGLANG_BENCH_SYSTEM_PROMPT", SYSTEM_PROMPT
        ),
        "SGLANG_BENCH_USER_PROMPT": os.getenv("SGLANG_BENCH_USER_PROMPT", USER_PROMPT),
        "SGLANG_BENCH_DISABLE_CUDA_GRAPH": os.getenv(
            "SGLANG_BENCH_DISABLE_CUDA_GRAPH", "0"
        ),
        "SGLANG_BENCH_RUN_PPL": os.getenv("SGLANG_BENCH_RUN_PPL", "0"),
        "SGLANG_BENCH_PPL_DATASET_FILE": os.getenv(
            "SGLANG_BENCH_PPL_DATASET_FILE", PPL_DATASET_FILE
        ),
        "SGLANG_BENCH_PPL_SEQ_LEN": os.getenv(
            "SGLANG_BENCH_PPL_SEQ_LEN", str(PPL_SEQ_LEN)
        ),
        "SGLANG_BENCH_PPL_NUM_SAMPLES": os.getenv(
            "SGLANG_BENCH_PPL_NUM_SAMPLES", str(PPL_NUM_SAMPLES)
        ),
        "SGLANG_BENCH_PPL_BATCH_SIZE": os.getenv(
            "SGLANG_BENCH_PPL_BATCH_SIZE", str(PPL_BATCH_SIZE)
        ),
        # HuggingFace downloads (models/datasets) live in persistent volumes.
        "HF_HUB_ENABLE_HF_TRANSFER": os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "1"),
        "HF_TOKEN": os.getenv("HF_TOKEN", ""),
        "HUGGINGFACE_HUB_TOKEN": os.getenv("HUGGINGFACE_HUB_TOKEN", ""),
        # Do not enable this env; current FA3 FP8-TC path is for Eagle3 speculative and is broken.
        "SGLANG_ENABLE_FA3_FP8_TC": "0",
    },
    retries=0,
)
def run_benchmark(
    kv_cache_dtype: str,
    attention_backend: str,
    moe_runner_backend: str = "triton_kernel",
    sampling_backend: str | None = None,
    batch_size: int = 1,
    max_new_tokens: int = 256,
    speculative: bool = False,
    speculative_topk: int = 1,
    speculative_num_steps: int = 4,
    speculative_num_draft_tokens: int | None = None,
):
    import sglang as sgl
    import flashinfer
    import torch
    import traceback
    import sys
    import time
    import json
    import hashlib

    from transformers import AutoTokenizer
    from contextlib import redirect_stdout, redirect_stderr

    # Use a unique log file per run
    run_id = int(time.time())
    log_file = f"/logs/benchmark_remote_{run_id}_{kv_cache_dtype}_{attention_backend}_bs{batch_size}.log"
    config_str = (
        f"{kv_cache_dtype}_{attention_backend}_{moe_runner_backend}_bs{batch_size}"
    )

    # Line-buffered so long-running runs can be monitored via `modal volume get`
    # while the function is still running.
    with open(log_file, "a", buffering=1) as f:
        with redirect_stdout(f), redirect_stderr(f):
            print("=" * 60, flush=True)
            print(
                f"Testing: kv_cache_dtype={kv_cache_dtype}, attn={attention_backend}, moe={moe_runner_backend}, sampling={sampling_backend or 'auto'}, bs={batch_size}",
                flush=True,
            )
            print(f"tp_size={TP_SIZE} gpu_spec={GPU_SPEC}", flush=True)
            print(
                f"FlashInfer version: {getattr(flashinfer, '__version__', 'unknown')}",
                flush=True,
            )
            print(f"Torch version: {torch.__version__}", flush=True)
            print(
                f"USE_HARMONY={os.getenv('SGLANG_BENCH_USE_HARMONY', '1')}", flush=True
            )
            print(
                f"DISABLE_CUDA_GRAPH={os.getenv('SGLANG_BENCH_DISABLE_CUDA_GRAPH', '0')}",
                flush=True,
            )
            if speculative:
                print(
                    f"Speculative: alg=EAGLE3 topk={speculative_topk} steps={speculative_num_steps} "
                    f"num_draft_tokens={speculative_num_draft_tokens or 'auto'} draft_model={DRAFT_MODEL_PATH}",
                    flush=True,
                )
            if os.getenv("SGLANG_BENCH_RUN_PPL", "0") == "1":
                print(
                    f"PPL: dataset={PPL_DATASET_FILE} seq_len={PPL_SEQ_LEN} num_samples={PPL_NUM_SAMPLES} bs={PPL_BATCH_SIZE}",
                    flush=True,
                )
            print("=" * 60, flush=True)

            try:
                if not os.path.isdir(MODEL_PATH):
                    raise RuntimeError(
                        f"Model path does not exist: {MODEL_PATH}. "
                        "Run the predownload step (ensure_assets) first."
                    )
                tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

                # FlashInfer FP8/FP4 runs (especially with GPT-OSS attention sinks) can
                # trigger long JIT compilation and/or long-running first forwards. The
                # scheduler watchdog measures "time since last forward()" and can
                # SIGQUIT the engine if it doesn't advance fast enough.
                default_watchdog_timeout = (
                    1800.0 if str(kv_cache_dtype).startswith(("fp8", "fp4")) else 300.0
                )
                watchdog_timeout = float(
                    os.getenv(
                        "SGLANG_BENCH_WATCHDOG_TIMEOUT",
                        str(default_watchdog_timeout),
                    )
                )
                print(f"watchdog_timeout={watchdog_timeout}", flush=True)

                def format_prompt(text: str) -> str:
                    if os.getenv("SGLANG_BENCH_USE_HARMONY", "1") != "1":
                        return text
                    messages = []
                    sys_prompt = os.getenv(
                        "SGLANG_BENCH_SYSTEM_PROMPT", SYSTEM_PROMPT
                    ).strip()
                    if sys_prompt:
                        messages.append({"role": "system", "content": sys_prompt})
                    messages.append({"role": "user", "content": text})
                    return tok.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                print("Initializing engine...", flush=True)
                engine = sgl.Engine(
                    model_path=MODEL_PATH,
                    tp_size=TP_SIZE,
                    kv_cache_dtype=kv_cache_dtype,
                    attention_backend=attention_backend,
                    moe_runner_backend=moe_runner_backend,
                    sampling_backend=sampling_backend,
                    context_length=8192,
                    allow_auto_truncate=True,
                    watchdog_timeout=watchdog_timeout,
                    disable_cuda_graph=(
                        os.getenv("SGLANG_BENCH_DISABLE_CUDA_GRAPH", "0") == "1"
                    ),
                    speculative_algorithm=("EAGLE3" if speculative else None),
                    speculative_draft_model_path=(
                        DRAFT_MODEL_PATH if speculative else None
                    ),
                    speculative_num_steps=(
                        speculative_num_steps if speculative else None
                    ),
                    speculative_eagle_topk=(speculative_topk if speculative else None),
                    speculative_num_draft_tokens=(
                        speculative_num_draft_tokens if speculative else None
                    ),
                    speculative_attention_mode=("decode" if speculative else "prefill"),
                    # Keep CUDA graph enabled, but only capture up to bs=1 for this
                    # benchmark by default; override via `batch_size` for throughput tests.
                    cuda_graph_max_bs=max(1, batch_size),
                )
                print("Engine initialized.", flush=True)

                sampling_params = {"temperature": 0, "max_new_tokens": max_new_tokens}
                warmup_params = dict(sampling_params)
                warmup_params["max_new_tokens"] = min(32, max_new_tokens)

                user_prompt = os.getenv("SGLANG_BENCH_USER_PROMPT", PROMPT)

                if batch_size == 1:
                    prompts = format_prompt(user_prompt)
                    warmup_sampling = warmup_params
                    bench_sampling = sampling_params
                else:
                    # Keep prompts identical for apples-to-apples throughput comparisons.
                    prompts = [format_prompt(user_prompt)] * batch_size
                    warmup_sampling = [warmup_params] * batch_size
                    bench_sampling = [sampling_params] * batch_size

                try:
                    prompt_tokens = len(
                        tok.encode(
                            prompts if isinstance(prompts, str) else prompts[0],
                            add_special_tokens=False,
                        )
                    )
                except Exception:
                    prompt_tokens = -1
                print(f"Prompt tokens: {prompt_tokens}", flush=True)

                print("Warming up...", flush=True)
                warmup = engine.generate(prompts, warmup_sampling)
                print("Warmup output:", flush=True)
                if isinstance(warmup, list):
                    for i, out in enumerate(warmup):
                        print(f"[Warmup {i}]")
                        print(out.get("text", out))
                else:
                    print(warmup.get("text", warmup))

                # Persist any JIT kernels compiled during warmup so a later failure
                # (e.g., during PPL/logprob) still leaves the cache usable.
                try:
                    flashinfer_cache_volume.commit()
                except Exception as e:
                    print(f"[WARN] flashinfer_cache_volume.commit failed: {e}")

                ppl_result = None
                if os.getenv("SGLANG_BENCH_RUN_PPL", "0") == "1":
                    print("\nComputing PPL (small slice)...", flush=True)

                    def iter_token_blocks(
                        dataset_file: str, seq_len_plus_one: int, num_blocks: int
                    ):
                        blocks = []
                        buf = []
                        eos = tok.eos_token_id
                        with open(dataset_file, "r", encoding="utf-8") as rf:
                            for line in rf:
                                try:
                                    ex = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                text = ex.get("text")
                                if not isinstance(text, str) or not text.strip():
                                    continue
                                ids = tok.encode(text, add_special_tokens=False)
                                if not ids:
                                    continue
                                buf.extend(ids)
                                if eos is not None:
                                    buf.append(eos)
                                while (
                                    len(buf) >= seq_len_plus_one
                                    and len(blocks) < num_blocks
                                ):
                                    blocks.append(buf[:seq_len_plus_one])
                                    buf = buf[seq_len_plus_one:]
                                if len(blocks) >= num_blocks:
                                    return blocks
                        return blocks

                    seq_len = int(PPL_SEQ_LEN)
                    num_samples = int(PPL_NUM_SAMPLES)
                    ppl_bs = max(1, int(PPL_BATCH_SIZE))
                    seq_len_plus_one = seq_len + 1

                    token_blocks = iter_token_blocks(
                        PPL_DATASET_FILE, seq_len_plus_one, num_samples
                    )
                    if len(token_blocks) < num_samples:
                        raise RuntimeError(
                            f"Only built {len(token_blocks)} token blocks (need {num_samples}). "
                            "Check PPL dataset volume/path and decrease PPL seq_len/num_samples."
                        )

                    nll_sum = 0.0
                    token_count = 0
                    sampling = {"temperature": 0, "max_new_tokens": 0}

                    for i in range(0, len(token_blocks), ppl_bs):
                        batch = token_blocks[i : i + ppl_bs]
                        t0 = time.time()
                        print(
                            f"[PPL] batch_start={i // ppl_bs} bs={len(batch)} tokens={len(batch[0])} t={t0:.3f}",
                            flush=True,
                        )
                        out = engine.generate(
                            input_ids=batch,
                            sampling_params=[sampling] * len(batch),
                            return_logprob=[True] * len(batch),
                            logprob_start_len=[1] * len(batch),
                        )
                        t1 = time.time()
                        print(
                            f"[PPL] batch_done={i // ppl_bs} dt={t1 - t0:.2f}s t={t1:.3f}",
                            flush=True,
                        )
                        if not isinstance(out, list):
                            raise TypeError(
                                f"Expected list output for batched PPL input, got: {type(out)}"
                            )
                        for req_out in out:
                            meta = req_out.get("meta_info", {})
                            token_logprobs = meta.get("input_token_logprobs", [])
                            lp = []
                            for t in token_logprobs:
                                logprob = t[0]
                                if logprob is None:
                                    continue
                                lp.append(float(logprob))
                            nll_sum += -sum(lp)
                            token_count += len(lp)

                    ppl = math.exp(nll_sum / max(1, token_count))
                    ppl_result = {"ppl": ppl, "tokens": token_count, "nll_sum": nll_sum}
                    print(
                        f"[PPL] tokens={token_count} nll_sum={nll_sum:.6f} ppl={ppl:.6f}"
                    )

                print("Benchmarking...", flush=True)
                start_time = time.time()
                output = engine.generate(prompts, bench_sampling)
                end_time = time.time()

                duration = end_time - start_time
                if isinstance(output, list):
                    num_tokens = sum(
                        o["meta_info"]["completion_tokens"] for o in output
                    )
                else:
                    num_tokens = output["meta_info"]["completion_tokens"]
                tps = num_tokens / max(duration, 1e-6)

                print(
                    f"Generated {num_tokens} tokens in {duration:.2f}s = {tps:.2f} tok/s"
                )
                print("Generated text:")
                out_text_for_hash = ""
                if isinstance(output, list):
                    for i, out in enumerate(output):
                        print(f"[Output {i}]")
                        txt = out.get("text", out)
                        print(txt)
                        out_text_for_hash += str(txt)
                        if "meta_info" in out:
                            print("Meta info:", out["meta_info"])
                else:
                    txt = output.get("text", output)
                    print(txt)
                    out_text_for_hash = str(txt)
                    if "meta_info" in output:
                        print("Meta info:", output["meta_info"])

                out_hash = hashlib.sha256(
                    out_text_for_hash.encode("utf-8", errors="replace")
                ).hexdigest()
                print(f"[OUTPUT_HASH] sha256={out_hash}")

                engine.shutdown()
                log_volume.commit()
                flashinfer_cache_volume.commit()
                ret = {
                    "tps": tps,
                    "success": True,
                    "log": log_file,
                    "out_hash": out_hash,
                }
                if ppl_result is not None:
                    ret["ppl"] = ppl_result
                return ret
            except Exception as e:
                err_out = f"Error in {config_str}: {e}\n{traceback.format_exc()}"
                print(err_out)
                log_volume.commit()
                flashinfer_cache_volume.commit()
                return {"success": False, "error": str(e), "log": log_file}


@app.local_entrypoint()
def main():
    local_log_file = os.path.join(
        LOG_DIR, f"sglang_benchmark_summary_{int(time.time())}.log"
    )

    def log_local(msg):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        with open(local_log_file, "a") as f:
            f.write(line + "\n")

    log_local(f"SGLang Optimization Benchmark Started. Local log: {local_log_file}")
    log_local("Using FlashInfer built from source with FP4 KV cache patches")
    log_local(f"GPU spec: {GPU_SPEC}  TP size: {TP_SIZE}")
    log_local(f"Model path: {MODEL_PATH}")
    log_local(f"Batch size: {BATCH_SIZE}, max_new_tokens: {MAX_NEW_TOKENS}")
    log_local(
        f"USE_HARMONY={int(USE_HARMONY)} disable_cuda_graph={int(DISABLE_CUDA_GRAPH)}"
    )
    log_local(f"system_prompt={SYSTEM_PROMPT!r}")
    log_local(f"user_prompt={USER_PROMPT!r}")
    if RUN_PPL:
        log_local(
            f"PPL enabled: dataset={PPL_DATASET_FILE} seq_len={PPL_SEQ_LEN} num_samples={PPL_NUM_SAMPLES} bs={PPL_BATCH_SIZE}"
        )

    # New workspace: volumes may be empty. Ensure models/dataset exist before running.
    if os.getenv("SGLANG_BENCH_SKIP_ASSET_DOWNLOAD", "0") != "1":
        log_local("Ensuring assets are present in Modal volumes (models/dataset)...")
        try:
            ensure_assets.remote(
                MODEL_PATH,
                (DRAFT_MODEL_PATH if RUN_EAGLE3 else None),
                (PPL_DATASET_REPO_ID if RUN_PPL else None),
                (PPL_DATASET_SPLIT if RUN_PPL else None),
                (PPL_DATASET_FILE if RUN_PPL else None),
                PPL_DATASET_MAX_ROWS,
            )
            log_local("Asset check/download complete.")
        except Exception as e:
            log_local(f"ASSET DOWNLOAD FAILED: {e}")
            raise

    def run_one(
        name: str,
        dtype: str,
        attn_backend: str,
        moe_backend: str,
        sampling_backend: str | None,
        speculative: bool,
    ) -> bool:
        log_local(f"Launching Configuration: {name}")
        try:
            result = run_benchmark.remote(
                dtype,
                attn_backend,
                moe_backend,
                sampling_backend,
                BATCH_SIZE,
                MAX_NEW_TOKENS,
                speculative,
                EAGLE3_TOPK,
                EAGLE3_STEPS,
                EAGLE3_NUM_DRAFT_TOKENS,
            )
        except Exception as e:
            log_local(f"CONFIG FAILED: {name}")
            log_local(f"Error Summary: {e}")
            return False

        if not result.get("success"):
            log_local(f"CONFIG FAILED: {name}")
            log_local(f"Error Summary: {result.get('error')}")
            log_local(f"See remote log for detail: {result.get('log')}")
            return False

        results[name] = result["tps"]
        log_local(f"SUCCESS: {name} -> {result['tps']:.2f} tok/s")
        if result.get("ppl"):
            ppl = result["ppl"]
            log_local(
                f"PPL: {ppl.get('ppl', float('nan')):.6f} (tokens={ppl.get('tokens', 0)})"
            )
            ppl_summaries[name] = ppl
        if result.get("out_hash"):
            output_hashes[name] = str(result["out_hash"])
            log_local(f"Output hash: {result['out_hash']}")
        log_local(f"Remote log: {result['log']}")
        return True

    # CASCADE RULES (IMPORTANT):
    # - FlashInfer must run BF16 -> FP8 -> FP4 (stop on first failure).
    # - FA3 must run BF16 -> FP8 (stop on first failure).
    # - EAGLE3 runs only after the corresponding non-spec FlashInfer config works,
    #   and it also cascades BF16 -> FP8 -> FP4 (stop on first failure).
    results: dict[str, float] = {}
    output_hashes: dict[str, str] = {}
    ppl_summaries: dict[str, dict] = {}

    # ----------------------------
    # FlashInfer (non-spec) cascade
    # ----------------------------
    want_flashinfer_fp8 = RUN_FP8 or RUN_FP4
    want_flashinfer_fp4 = RUN_FP4
    want_flashinfer_bf16 = (
        RUN_BF16 or want_flashinfer_fp8 or want_flashinfer_fp4 or RUN_EAGLE3
    )

    flashinfer_bf16_ok = False
    flashinfer_fp8_ok = False
    flashinfer_fp4_ok = False

    if want_flashinfer_bf16:
        flashinfer_bf16_ok = run_one(
            "BF16 (FlashInfer)", "auto", "flashinfer", "triton_kernel", None, False
        )
    if want_flashinfer_fp8:
        if flashinfer_bf16_ok:
            flashinfer_fp8_ok = run_one(
                "FP8 KV (FlashInfer)",
                "fp8_e4m3",
                "flashinfer",
                "triton_kernel",
                None,
                False,
            )
        else:
            log_local(
                "SKIPPING: FlashInfer FP8/FP4 because FlashInfer BF16 failed (cascade stop)."
            )
    if want_flashinfer_fp4:
        if flashinfer_fp8_ok:
            flashinfer_fp4_ok = run_one(
                "NVFP4 KV (FlashInfer)",
                "fp4_e2m1",
                "flashinfer",
                "triton_kernel",
                None,
                False,
            )
        else:
            log_local(
                "SKIPPING: FlashInfer FP4 because FlashInfer FP8 failed (cascade stop)."
            )

    # ----------------------------
    # FA3 (non-spec) cascade
    # ----------------------------
    if RUN_FP8_FA3:
        fa3_bf16_ok = run_one(
            "BF16 (FA3)", "auto", "fa3", "triton_kernel", "flashinfer", False
        )
        if fa3_bf16_ok:
            _ = run_one(
                "FP8 KV (FA3)",
                "fp8_e4m3",
                "fa3",
                "triton_kernel",
                "flashinfer",
                False,
            )
        else:
            log_local("SKIPPING: FA3 FP8 because FA3 BF16 failed (cascade stop).")

    # ----------------------------
    # EAGLE3 (FlashInfer) cascade
    # ----------------------------
    if RUN_EAGLE3:
        # Speculative decoding variants. For topk>1, strongly consider forcing page_size=1
        # via `SGLANG_FLASHINFER_FORCE_PAGE1=1` to avoid instability.
        spec_suffix = (
            f"steps={EAGLE3_STEPS}, topk={EAGLE3_TOPK}, "
            f"draft={EAGLE3_NUM_DRAFT_TOKENS or 'auto'}"
        )
        if not flashinfer_bf16_ok:
            log_local(
                "SKIPPING: all EAGLE3 configs because FlashInfer BF16 failed (must be working before EAGLE)."
            )
        else:
            eagle_bf16_ok = run_one(
                f"EAGLE3 BF16 ({spec_suffix})",
                "auto",
                "flashinfer",
                "triton_kernel",
                None,
                True,
            )

            eagle_fp8_ok = False
            if want_flashinfer_fp8:
                if eagle_bf16_ok and flashinfer_fp8_ok:
                    eagle_fp8_ok = run_one(
                        f"EAGLE3 FP8 ({spec_suffix})",
                        "fp8_e4m3",
                        "flashinfer",
                        "triton_kernel",
                        None,
                        True,
                    )
                else:
                    log_local(
                        "SKIPPING: EAGLE3 FP8 because prerequisites failed (FlashInfer FP8 + EAGLE3 BF16)."
                    )

            if want_flashinfer_fp4:
                if eagle_fp8_ok and flashinfer_fp4_ok:
                    _ = run_one(
                        f"EAGLE3 FP4 ({spec_suffix})",
                        "fp4_e2m1",
                        "flashinfer",
                        "triton_kernel",
                        None,
                        True,
                    )
                else:
                    log_local(
                        "SKIPPING: EAGLE3 FP4 because prerequisites failed (FlashInfer FP4 + EAGLE3 FP8)."
                    )

    log_local("\n" + "=" * 60)
    log_local("FINAL SUMMARY")
    log_local("=" * 60)
    bf16_best = results.get("BF16 (FlashInfer)")
    baseline = bf16_best or (next(iter(results.values()), None) if results else None)

    for name, tps in results.items():
        speedup = tps / baseline if (baseline and tps) else 1.0
        log_local(f"{name:35}: {tps if tps else 'FAILED':8} tok/s ({speedup:.2f}x)")
    log_local("=" * 60)

    if output_hashes:
        log_local("OUTPUT HASHES (sha256 of generated text)")
        for name, h in output_hashes.items():
            log_local(f"{name:35}: {h}")
        log_local("=" * 60)

    if ppl_summaries:
        log_local("PPL SUMMARY")
        for name, ppl in ppl_summaries.items():
            log_local(
                f"{name:35}: ppl={ppl.get('ppl', float('nan')):.6f} tokens={ppl.get('tokens', 0)}"
            )
        log_local("=" * 60)
