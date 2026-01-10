# modal/sglang_nll_scoring.py
# Compute completion-only NLL/PPL on Harmony-formatted `text` rows using SGLang
# teacher-forcing logprobs (max_new_tokens=0, return_logprob=True).
#
# Candidate Parquet rows come from the CPU pipeline and should include at least:
#   id, text, loss_mode (optional), and optionally completion span offsets.
#
# This job:
# - reads candidate parquet files from an HF dataset repo (by subdir prefix)
# - computes assistant-only token mask (Harmony assistant message body spans)
# - computes NLL/PPL over masked tokens using SGLang Engine input token logprobs
# - writes Parquet score shards to a persistent Modal volume

from __future__ import annotations

import os
import time
from pathlib import Path

import modal

APP_NAME = "harmony-sglang-nll-scoring"
GPU_SPEC = os.environ.get("SGLANG_NLL_GPU", "H100")

# Optional: pass your local `HF_TOKEN` into the Modal container at launch time.
_secrets = []
_hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if not _hf_token:
    try:
        from huggingface_hub import HfFolder

        _hf_token = HfFolder.get_token()
    except Exception:
        _hf_token = None
if _hf_token:
    _secrets.append(
        modal.Secret.from_dict({"HF_TOKEN": _hf_token, "HUGGINGFACE_HUB_TOKEN": _hf_token})
    )

# Modal does not automatically propagate local env vars into containers, so we
# pass configuration explicitly.
_run_env: dict[str, str | None] = {
    # Ensure Python logs flush immediately in Modal.
    "PYTHONUNBUFFERED": "1",
    "CANDIDATE_DATASET_ID": os.environ.get("CANDIDATE_DATASET_ID"),
    "CANDIDATE_SUBDIR": os.environ.get("CANDIDATE_SUBDIR"),
    "MODEL_ID": os.environ.get("MODEL_ID"),
    "MODEL_PATH": os.environ.get("MODEL_PATH"),
    "TP_SIZE": os.environ.get("TP_SIZE"),
    "DTYPE": os.environ.get("DTYPE"),
    "KV_CACHE_DTYPE": os.environ.get("KV_CACHE_DTYPE"),
    "ATTENTION_BACKEND": os.environ.get("ATTENTION_BACKEND"),
    "PREFILL_ATTENTION_BACKEND": os.environ.get("PREFILL_ATTENTION_BACKEND"),
    "DECODE_ATTENTION_BACKEND": os.environ.get("DECODE_ATTENTION_BACKEND"),
    "MEM_FRACTION_STATIC": os.environ.get("MEM_FRACTION_STATIC"),
    "CHUNKED_PREFILL_SIZE": os.environ.get("CHUNKED_PREFILL_SIZE"),
    "CUDA_GRAPH_MAX_BS": os.environ.get("CUDA_GRAPH_MAX_BS"),
    "MAX_RUNNING_REQUESTS": os.environ.get("MAX_RUNNING_REQUESTS"),
    "MAX_TOTAL_TOKENS": os.environ.get("MAX_TOTAL_TOKENS"),
    "SGLANG_WATCHDOG_TIMEOUT": os.environ.get("SGLANG_WATCHDOG_TIMEOUT"),
    "SGLANG_DISABLE_CUDA_GRAPH": os.environ.get("SGLANG_DISABLE_CUDA_GRAPH"),
    "SGLANG_SKIP_SERVER_WARMUP": os.environ.get("SGLANG_SKIP_SERVER_WARMUP"),
    "LOG_EVERY_S": os.environ.get("LOG_EVERY_S"),
    "MAX_LENGTH": os.environ.get("MAX_LENGTH"),
    "BATCH_SIZE": os.environ.get("BATCH_SIZE"),
    "MAX_RECORDS": os.environ.get("MAX_RECORDS"),
    "ROWS_PER_SHARD": os.environ.get("ROWS_PER_SHARD"),
    "RUN_TAG": os.environ.get("RUN_TAG"),
}

data_volume = modal.Volume.from_name("mhc-data-volume", create_if_missing=True)
model_volume = modal.Volume.from_name("gpt-oss-model-weights", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("hf-cache-persistent", create_if_missing=True)
flashinfer_cache_volume = modal.Volume.from_name(
    "flashinfer-jit-cache", create_if_missing=True
)

BASE_IMAGE = "nvidia/cuda:12.8.0-devel-ubuntu24.04"

_repo_root = Path(__file__).resolve().parents[1]
_sglang_src_dir = _repo_root / "sglang-flashinfer" / "python" / "sglang"
_sgl_kernel_src_dir = _repo_root / "sglang-flashinfer" / "sgl-kernel" / "python" / "sgl_kernel"

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
        "wget",
        "ninja-build",
    )
    .run_commands("python -m pip install -U pip setuptools wheel")
    .run_commands("pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128")
    .run_commands(
        "pip install 'sglang[all]' hf_transfer "
        "pyarrow==22.0.0 datasets==3.2.0 "
        "'transformers>=4.57.1,!=4.57.2'"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(
        str(_sglang_src_dir),
        remote_path="/root/sglang-src",
        copy=True,
        ignore=[
            "**/__pycache__",
            "**/__pycache__/**",
        ],
    )
    .add_local_dir(
        str(_sgl_kernel_src_dir),
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

app = modal.App(APP_NAME)


@app.function(
    image=image,
    env=_run_env,
    volumes={
        "/models": model_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
    secrets=_secrets,
    timeout=60 * 60 * 12,
    cpu=8.0,
    memory=32768,
)
def prefetch_model() -> str:
    """CPU stage: download model weights into the persistent model volume."""
    import inspect
    import os
    import time
    from pathlib import Path
    from typing import Any

    from huggingface_hub import snapshot_download

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    model_id = (os.environ.get("MODEL_ID") or "").strip() or "openai/gpt-oss-120b"
    model_path = (os.environ.get("MODEL_PATH") or "").strip() or None
    model_id_or_path = model_path or model_id

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
                "Use a HF repo id like 'openai/gpt-oss-120b' or a /models/<org>/<model> path."
            )
        return f"/models/{repo_id}"

    def _has_model_files(local_dir: str) -> bool:
        if not os.path.isdir(local_dir):
            return False
        if not os.path.exists(os.path.join(local_dir, "config.json")):
            return False
        # Tokenizer is required for both scoring + correct offset mapping.
        if not os.path.exists(os.path.join(local_dir, "tokenizer.json")):
            return False

        index_path = os.path.join(local_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            try:
                import json

                with open(index_path, "r", encoding="utf-8") as f:
                    idx = json.load(f)
                weight_map = idx.get("weight_map") or {}
                shard_files = sorted(set(weight_map.values()))
                if not shard_files:
                    return False
                for sf in shard_files:
                    if not os.path.exists(os.path.join(local_dir, sf)):
                        return False
                return True
            except Exception:
                return False

        # Single-file safetensors fallback.
        return any(name.endswith(".safetensors") for name in os.listdir(local_dir))

    local_dir = _infer_local_dir(model_id_or_path)
    if _has_model_files(local_dir):
        print(f"[prefetch_model] already present: {local_dir}", flush=True)
        return local_dir

    repo_id = _infer_repo_id(model_id_or_path)
    if repo_id is None:
        raise ValueError(f"Cannot infer repo_id for model download from {model_id_or_path!r}")

    os.makedirs(local_dir, exist_ok=True)
    print(f"[prefetch_model] downloading {repo_id} -> {local_dir}", flush=True)
    t0 = time.time()

    sd_sig = inspect.signature(snapshot_download)
    kwargs: dict[str, Any] = dict(
        repo_id=repo_id,
        local_dir=local_dir,
        resume_download=True,
        ignore_patterns=["metal/**", "original/**"],
    )
    if "local_dir_use_symlinks" in sd_sig.parameters:
        kwargs["local_dir_use_symlinks"] = True
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None
    if token and "token" in sd_sig.parameters:
        kwargs["token"] = token

    import threading

    done: list[bool] = []
    err: list[BaseException] = []

    def _do() -> None:
        try:
            snapshot_download(**kwargs)
            done.append(True)
        except BaseException as e:  # noqa: BLE001
            err.append(e)

    th = threading.Thread(target=_do, daemon=True)
    th.start()
    while th.is_alive():
        th.join(timeout=30)
        if th.is_alive():
            print(f"[prefetch_model] still downloading... elapsed_s={time.time() - t0:.0f}", flush=True)
    if err:
        raise err[0]

    dt = time.time() - t0

    if not _has_model_files(local_dir):
        raise RuntimeError(f"Model download incomplete: {local_dir}")

    model_volume.commit()
    hf_cache_volume.commit()
    print(f"[prefetch_model] done dt_s={dt:.1f}", flush=True)
    return local_dir


@app.function(
    image=image,
    env=_run_env,
    volumes={
        "/root/.cache/huggingface": hf_cache_volume,
    },
    secrets=_secrets,
    timeout=60 * 60 * 12,
    cpu=8.0,
    memory=32768,
)
def prefetch_candidates() -> str:
    """CPU stage: snapshot_download the candidate parquet files into HF cache."""
    import inspect
    import os
    import time
    from pathlib import Path
    from typing import Any

    from huggingface_hub import snapshot_download

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    dataset_id = os.environ.get("CANDIDATE_DATASET_ID")
    if not dataset_id:
        raise RuntimeError("Set CANDIDATE_DATASET_ID (HF dataset repo_id with candidate parquet files)")
    subdir = (os.environ.get("CANDIDATE_SUBDIR") or "").strip().strip("/") or None

    allow_patterns = ["**/*.parquet", "README.md"]
    if subdir:
        allow_patterns = [f"{subdir}/**/*.parquet", "README.md"]

    print(f"[prefetch_candidates] snapshot_download {dataset_id} subdir={subdir!r}", flush=True)
    t0 = time.time()

    sd_sig = inspect.signature(snapshot_download)
    kwargs: dict[str, Any] = dict(
        repo_id=dataset_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        resume_download=True,
    )
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None
    if token and "token" in sd_sig.parameters:
        kwargs["token"] = token

    import threading

    result: dict[str, str] = {}
    err: list[BaseException] = []

    def _do() -> None:
        try:
            result["snap_path"] = snapshot_download(**kwargs)
        except BaseException as e:  # noqa: BLE001
            err.append(e)

    th = threading.Thread(target=_do, daemon=True)
    th.start()
    while th.is_alive():
        th.join(timeout=30)
        if th.is_alive():
            print(
                f"[prefetch_candidates] still downloading... elapsed_s={time.time() - t0:.0f}",
                flush=True,
            )
    if err:
        raise err[0]

    snap_path = result.get("snap_path") or ""
    dt = time.time() - t0

    snap = Path(snap_path)
    scan_root = (snap / subdir) if subdir else snap
    n_parquet = sum(1 for _ in scan_root.rglob("*.parquet"))
    hf_cache_volume.commit()
    print(f"[prefetch_candidates] done dt_s={dt:.1f} parquet_files={n_parquet}", flush=True)
    return str(scan_root)


@app.function(
    image=image,
    env=_run_env,
    gpu=GPU_SPEC,
    volumes={
        "/root/data": data_volume,
        "/models": model_volume,
        "/root/.cache/huggingface": hf_cache_volume,
        "/root/.cache/flashinfer": flashinfer_cache_volume,
    },
    secrets=_secrets,
    timeout=60 * 60 * 12,
    cpu=16.0,
    memory=131072,
)
def score_nll() -> str:
    import json
    import math
    from dataclasses import dataclass, field
    from hashlib import sha1
    from typing import Any

    import pyarrow as pa
    import pyarrow.parquet as pq
    import sglang as sgl
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    START_TAG = "<|start|>"
    END_TAG = "<|end|>"
    CALL_TAG = "<|call|>"
    CHANNEL_TAG = "<|channel|>"
    MESSAGE_TAG = "<|message|>"
    RETURN_TAG = "<|return|>"

    def assistant_content_spans(text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        if not isinstance(text, str):
            return spans

        i = 0
        n = len(text)
        while True:
            start = text.find(START_TAG, i)
            if start < 0:
                break

            role_start = start + len(START_TAG)
            msg_tag = text.find(MESSAGE_TAG, role_start)
            if msg_tag < 0:
                raise ValueError("malformed harmony text: missing <|message|>")

            header = text[role_start:msg_tag]
            if CHANNEL_TAG in header:
                role = header.split(CHANNEL_TAG, 1)[0]
            else:
                role = header

            content_start = msg_tag + len(MESSAGE_TAG)
            end_pos = text.find(END_TAG, content_start)
            call_pos = text.find(CALL_TAG, content_start)
            return_pos = text.find(RETURN_TAG, content_start)

            candidates: list[tuple[int, str]] = []
            if end_pos >= 0:
                candidates.append((end_pos, END_TAG))
            if call_pos >= 0:
                candidates.append((call_pos, CALL_TAG))
            if return_pos >= 0:
                candidates.append((return_pos, RETURN_TAG))

            if not candidates:
                raise ValueError(
                    "malformed harmony text: missing <|end|>, <|call|>, or <|return|>"
                )

            delim_pos, delim_tag = min(candidates, key=lambda t: t[0])
            if role == "assistant" or role.startswith("assistant "):
                spans.append((content_start, delim_pos))

            i = delim_pos + len(delim_tag)
            if i >= n:
                break

        return spans

    def _load_spans(
        *,
        text: str,
        loss_mode: str | None,
        spans_json: str | None,
        start_char: int | None,
        end_char: int | None,
    ) -> list[tuple[int, int]]:
        if not isinstance(text, str):
            return []
        if spans_json:
            try:
                raw = json.loads(spans_json)
                spans: list[tuple[int, int]] = []
                for item in raw:
                    if (
                        isinstance(item, (list, tuple))
                        and len(item) == 2
                        and isinstance(item[0], int)
                        and isinstance(item[1], int)
                    ):
                        spans.append((item[0], item[1]))
                if spans:
                    return spans
            except Exception:
                pass
        if start_char is not None and end_char is not None:
            try:
                s = int(start_char)
                e = int(end_char)
            except Exception:
                s, e = -1, -1
            if 0 <= s < e <= len(text):
                return [(s, e)]

        if loss_mode is None or loss_mode == "assistant_all":
            try:
                return assistant_content_spans(text)
            except Exception:
                return []
        return []

    def _token_keep_mask(
        offsets: list[tuple[int, int]], spans: list[tuple[int, int]]
    ) -> list[bool]:
        if not spans:
            return [False] * len(offsets)
        keep = [False] * len(offsets)
        for i, (cs, ce) in enumerate(offsets):
            if cs == 0 and ce == 0:
                continue
            for ss, ee in spans:
                if ce > ss and cs < ee:
                    keep[i] = True
                    break
        return keep

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    def _env_flag(name: str, default: bool) -> bool:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            return default
        return raw.lower() not in {"0", "false", "no", "off"}

    def _stage_start(name: str) -> float:
        t = time.time()
        print(f"[t] {name} start", flush=True)
        return t

    def _stage_done(name: str, t0: float) -> None:
        dt = time.time() - t0
        print(f"[t] {name} done dt_s={dt:.3f}", flush=True)

    dataset_id = os.environ.get("CANDIDATE_DATASET_ID")
    if not dataset_id:
        raise RuntimeError("Set CANDIDATE_DATASET_ID (HF dataset repo_id with candidate parquet files)")
    subdir = (os.environ.get("CANDIDATE_SUBDIR") or "").strip().strip("/") or None

    model_id = (os.environ.get("MODEL_ID") or "").strip() or "openai/gpt-oss-120b"
    model_path = (os.environ.get("MODEL_PATH") or "").strip() or None
    tp_size = int(os.environ.get("TP_SIZE") or "1")
    max_length = int(os.environ.get("MAX_LENGTH") or "4096")
    batch_size = int(os.environ.get("BATCH_SIZE") or "1")
    max_records = int(os.environ.get("MAX_RECORDS") or "0")  # 0 = all
    rows_per_shard = int(os.environ.get("ROWS_PER_SHARD") or "50000")
    log_every_s = float(os.environ.get("LOG_EVERY_S") or "30")

    # SGLang engine memory knobs. GPT-OSS-120B on a single H100 needs a larger
    # mem_fraction_static than the default heuristic in some configurations.
    mem_fraction_static_env = (os.environ.get("MEM_FRACTION_STATIC") or "").strip()
    if mem_fraction_static_env:
        mem_fraction_static: float | None = float(mem_fraction_static_env)
    else:
        mem_fraction_static = None

    chunked_prefill_size_env = (os.environ.get("CHUNKED_PREFILL_SIZE") or "").strip()
    chunked_prefill_size = (
        int(chunked_prefill_size_env)
        if chunked_prefill_size_env
        else None
    )

    cuda_graph_max_bs_env = (os.environ.get("CUDA_GRAPH_MAX_BS") or "").strip()
    cuda_graph_max_bs = int(cuda_graph_max_bs_env) if cuda_graph_max_bs_env else None

    max_running_requests_env = (os.environ.get("MAX_RUNNING_REQUESTS") or "").strip()
    max_running_requests = (
        int(max_running_requests_env)
        if max_running_requests_env
        else max(8, batch_size)
    )

    max_total_tokens_env = (os.environ.get("MAX_TOTAL_TOKENS") or "").strip()
    max_total_tokens = (
        int(max_total_tokens_env)
        if max_total_tokens_env
        else (max_length * batch_size + 512)
    )

    out_tag = (os.environ.get("RUN_TAG") or "").strip()
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_id = out_tag or sha1(f"{dataset_id}|{subdir}|{model_id}|{ts}".encode("utf-8")).hexdigest()[:12]
    out_dir = Path("/root/data/nll_scores") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    def _infer_repo_id(maybe_path_or_id: str) -> str | None:
        if not maybe_path_or_id:
            return None
        if "/" in maybe_path_or_id and not maybe_path_or_id.startswith("/"):
            return maybe_path_or_id
        return None

    def _infer_local_dir(maybe_path_or_id: str) -> str:
        if maybe_path_or_id.startswith("/models/"):
            return maybe_path_or_id
        repo_id = _infer_repo_id(maybe_path_or_id)
        if repo_id is None:
            raise ValueError(
                f"Cannot infer HuggingFace repo_id from model_id_or_path={maybe_path_or_id!r}. "
                "Use a HF repo id like 'openai/gpt-oss-120b' or a /models/<org>/<model> path."
            )
        return f"/models/{repo_id}"

    def _has_model_files(local_dir: str) -> bool:
        if not os.path.isdir(local_dir):
            return False
        if not os.path.exists(os.path.join(local_dir, "config.json")):
            return False
        if not os.path.exists(os.path.join(local_dir, "tokenizer.json")):
            return False

        index_path = os.path.join(local_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            try:
                import json

                with open(index_path, "r", encoding="utf-8") as f:
                    idx = json.load(f)
                weight_map = idx.get("weight_map") or {}
                shard_files = sorted(set(weight_map.values()))
                if not shard_files:
                    return False
                for sf in shard_files:
                    if not os.path.exists(os.path.join(local_dir, sf)):
                        return False
                return True
            except Exception:
                return False

        return any(name.endswith(".safetensors") for name in os.listdir(local_dir))

    def _require_model(model_id_or_path: str) -> str:
        local_dir = _infer_local_dir(model_id_or_path)
        if not _has_model_files(local_dir):
            raise RuntimeError(
                f"Model not found at {local_dir}. Run the CPU prefetch first "
                "(prefetch_model) so we don't spend GPU time downloading weights."
            )
        return local_dir

    model_dir = _require_model(model_path or model_id)
    print(f"[*] model_dir={model_dir}", flush=True)

    t_tok = _stage_start("load_tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Need a fast tokenizer for return_offsets_mapping=True")
    _stage_done("load_tokenizer", t_tok)

    disable_cuda_graph = _env_flag("SGLANG_DISABLE_CUDA_GRAPH", False)
    skip_server_warmup = _env_flag("SGLANG_SKIP_SERVER_WARMUP", False)

    t_eng = _stage_start("init_sglang_engine")
    print(
        "[cfg] "
        f"tp_size={tp_size} max_length={max_length} batch_size={batch_size} "
        f"dtype={(os.environ.get('DTYPE') or '').strip() or 'auto'} "
        f"kv_cache_dtype={(os.environ.get('KV_CACHE_DTYPE') or '').strip() or 'auto'} "
        f"mem_fraction_static={mem_fraction_static} chunked_prefill_size={chunked_prefill_size} "
        f"cuda_graph_max_bs={cuda_graph_max_bs} max_running_requests={max_running_requests} "
        f"max_total_tokens={max_total_tokens} disable_cuda_graph={disable_cuda_graph} "
        f"skip_server_warmup={skip_server_warmup}",
        flush=True,
    )
    engine_kwargs: dict[str, Any] = {
        "model_path": model_dir,
        "tp_size": tp_size,
        "context_length": max_length,
        "allow_auto_truncate": True,
        "watchdog_timeout": float(os.environ.get("SGLANG_WATCHDOG_TIMEOUT") or "1800"),
        "disable_cuda_graph": disable_cuda_graph,
        "skip_server_warmup": skip_server_warmup,
        "max_running_requests": max_running_requests,
        "max_total_tokens": max_total_tokens,
    }
    dtype_env = (os.environ.get("DTYPE") or "").strip()
    if dtype_env:
        engine_kwargs["dtype"] = dtype_env

    kv_cache_dtype_env = (os.environ.get("KV_CACHE_DTYPE") or "").strip()
    if kv_cache_dtype_env:
        engine_kwargs["kv_cache_dtype"] = kv_cache_dtype_env

    attention_backend_env = (os.environ.get("ATTENTION_BACKEND") or "").strip()
    if attention_backend_env:
        engine_kwargs["attention_backend"] = attention_backend_env

    prefill_attention_backend_env = (os.environ.get("PREFILL_ATTENTION_BACKEND") or "").strip()
    if prefill_attention_backend_env:
        engine_kwargs["prefill_attention_backend"] = prefill_attention_backend_env

    decode_attention_backend_env = (os.environ.get("DECODE_ATTENTION_BACKEND") or "").strip()
    if decode_attention_backend_env:
        engine_kwargs["decode_attention_backend"] = decode_attention_backend_env

    if mem_fraction_static is not None:
        engine_kwargs["mem_fraction_static"] = float(mem_fraction_static)
    if chunked_prefill_size is not None:
        engine_kwargs["chunked_prefill_size"] = int(chunked_prefill_size)
    if cuda_graph_max_bs is not None:
        engine_kwargs["cuda_graph_max_bs"] = int(cuda_graph_max_bs)

    engine = sgl.Engine(**engine_kwargs)
    print("[*] engine ready", flush=True)
    # Avoid calling engine.get_server_info() here: it can block on internal state
    # RPCs (scheduler readiness) and stall the whole job before the first warmup.
    try:
        from dataclasses import asdict, is_dataclass

        sa = getattr(engine, "server_args", None)
        if sa is not None:
            sa_dict = asdict(sa) if is_dataclass(sa) else {}
            keys = [
                "dtype",
                "kv_cache_dtype",
                "attention_backend",
                "prefill_attention_backend",
                "decode_attention_backend",
                "chunked_prefill_size",
                "cuda_graph_max_bs",
                "mem_fraction_static",
                "page_size",
                "moe_runner_backend",
                "sampling_backend",
                "max_running_requests",
                "max_total_tokens",
                "disable_cuda_graph",
            ]
            small = {k: sa_dict.get(k) for k in keys if k in sa_dict}
            print(f"[*] server_args (selected): {small}", flush=True)
    except Exception as e:
        print(f"[warn] server_args summary failed: {type(e).__name__}: {e}", flush=True)
    _stage_done("init_sglang_engine", t_eng)

    def _extract_logprob(entry: Any) -> float | None:
        if entry is None:
            return None
        if isinstance(entry, (int, float)):
            return float(entry)
        if isinstance(entry, dict):
            for key in ("logprob", "token_logprob", "lp", "log_prob"):
                if key in entry:
                    try:
                        return float(entry[key])
                    except Exception:
                        return None
            return None
        if isinstance(entry, (list, tuple)) and entry:
            # Some SGLang versions return per-token lists (top-k) where the
            # first element corresponds to the realized token.
            return _extract_logprob(entry[0])
        return None

    # Warmup: force a single forward so compilation/setup doesn't silently
    # delay the first real batch.
    warmup_text = "Hello world. " * 64
    warmup_ids = tokenizer(warmup_text, add_special_tokens=False)["input_ids"][: min(512, max_length)]
    print(f"[*] warmup start tokens={len(warmup_ids)}", flush=True)
    t_warm = time.time()
    warm_out = engine.generate(
        input_ids=[warmup_ids],
        sampling_params=[{"temperature": 0, "max_new_tokens": 0}],
        return_logprob=[True],
        logprob_start_len=[0],
    )
    warm_dt = time.time() - t_warm
    warm_req = warm_out[0] if isinstance(warm_out, list) else warm_out
    warm_meta = warm_req.get("meta_info", {}) if isinstance(warm_req, dict) else {}
    warm_lps = warm_meta.get("input_token_logprobs")
    warm_lp_len = len(warm_lps) if isinstance(warm_lps, list) else 0
    print(
        f"[*] warmup done dt_s={warm_dt:.2f} len(input_ids)={len(warmup_ids)} "
        f"len(input_token_logprobs)={warm_lp_len}",
        flush=True,
    )
    if not warm_lp_len:
        print("[warn] warmup: missing/empty meta_info['input_token_logprobs']", flush=True)
    try:
        flashinfer_cache_volume.commit()
        print("[*] flashinfer cache committed", flush=True)
    except Exception as e:
        print(f"[warn] flashinfer cache commit failed: {type(e).__name__}: {e}", flush=True)

    @dataclass
    class ScoreWriter:
        out_dir: Path
        rows_per_shard: int
        compression: str = "zstd"
        shard_index: int = 0
        rows_in_shard: int = 0
        buf: dict[str, list[Any]] = field(default_factory=dict)

        def add(self, row: dict[str, Any]) -> None:
            for k, v in row.items():
                self.buf.setdefault(k, []).append(v)
            self.rows_in_shard += 1
            if self.rows_in_shard >= self.rows_per_shard:
                self.flush()

        def flush(self) -> None:
            if not self.rows_in_shard:
                return
            table = pa.table(self.buf)
            path = self.out_dir / f"part-{self.shard_index:05d}.parquet"
            pq.write_table(table, path, compression=self.compression)
            print(f"[write] {path} rows={self.rows_in_shard}", flush=True)
            # Persist progress incrementally so a preemption/OOM doesn't lose hours of work.
            data_volume.commit()
            self.shard_index += 1
            self.rows_in_shard = 0
            self.buf = {}

    writer = ScoreWriter(out_dir=out_dir, rows_per_shard=rows_per_shard)

    t_snap = _stage_start(
        f"snapshot_download_dataset(local_files_only) {dataset_id} subdir={subdir or ''}"
    )
    allow_patterns = ["**/*.parquet", "README.md"]
    if subdir:
        allow_patterns = [f"{subdir}/**/*.parquet", "README.md"]
    sd_kwargs: dict[str, Any] = dict(
        repo_id=dataset_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        local_files_only=True,
    )
    try:
        snap_path = snapshot_download(**sd_kwargs)
    except Exception as e:
        raise RuntimeError(
            "Candidate dataset snapshot not found in local HF cache. "
            "Run the CPU prefetch first (prefetch_candidates) so we don't spend GPU time downloading. "
            f"Original error: {type(e).__name__}: {e}"
        ) from e
    _stage_done(f"snapshot_download_dataset {dataset_id}", t_snap)

    snap = Path(snap_path)
    scan_root = (snap / subdir) if subdir else snap
    parquet_files = sorted(scan_root.rglob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"no parquet files found under {scan_root}")
    print(f"[*] parquet files: {len(parquet_files)}", flush=True)

    processed = 0
    processed_input_tokens = 0
    processed_kept_tokens = 0
    t0_run = time.time()
    t_last_log = t0_run
    sampling = {"temperature": 0, "max_new_tokens": 0}
    did_logprob_debug = False

    try:
        for file_idx, pf in enumerate(parquet_files):
            rel_path = str(pf.relative_to(snap))
            parquet = pq.ParquetFile(pf)
            file_rows = int(parquet.metadata.num_rows) if parquet.metadata else -1
            print(
                f"[*] scoring file {file_idx + 1}/{len(parquet_files)} rows={file_rows}: {rel_path}",
                flush=True,
            )
            cols = set(parquet.schema.names)
            if "id" not in cols or "text" not in cols:
                raise RuntimeError(f"missing required columns in {rel_path} (have={sorted(cols)[:20]}...)")

            read_cols = ["id", "text"]
            for opt in ["loss_mode", "assistant_spans_json", "completion_start_char", "completion_end_char"]:
                if opt in cols:
                    read_cols.append(opt)

            parquet_batch_size = max(256, batch_size)
            for batch in parquet.iter_batches(columns=read_cols, batch_size=parquet_batch_size):
                # Materialize to python for tokenizer + SGLang input.
                batch_dict = batch.to_pydict()
                ids: list[str] = batch_dict.get("id", [])
                texts: list[str] = batch_dict.get("text", [])
                loss_modes: list[str] = batch_dict.get("loss_mode") or ["assistant_all"] * len(texts)
                spans_jsons: list[str | None] = batch_dict.get("assistant_spans_json") or [None] * len(texts)
                start_chars: list[int | None] = batch_dict.get("completion_start_char") or [None] * len(texts)
                end_chars: list[int | None] = batch_dict.get("completion_end_char") or [None] * len(texts)

                # `pyarrow.ParquetFile.iter_batches(batch_size=...)` controls the
                # maximum number of rows materialized per iterator step. If we
                # hard-cap it to 256, we cannot test true micro-batches >256.
                #
                # We already chunk to `batch_size` below, so it is safe to read
                # up to the micro-batch size here.
                for start in range(0, len(texts), batch_size):
                    sub_ids = ids[start : start + batch_size]
                    sub_texts = texts[start : start + batch_size]
                    sub_loss_modes = loss_modes[start : start + batch_size]
                    sub_spans_jsons = spans_jsons[start : start + batch_size]
                    sub_start_chars = start_chars[start : start + batch_size]
                    sub_end_chars = end_chars[start : start + batch_size]

                    spans_per_ex: list[list[tuple[int, int]]] = []
                    for t, mode, sj, sc, ec in zip(
                        sub_texts, sub_loss_modes, sub_spans_jsons, sub_start_chars, sub_end_chars
                    ):
                        spans_per_ex.append(
                            _load_spans(
                                text=t,
                                loss_mode=mode,
                                spans_json=sj if isinstance(sj, str) and sj else None,
                                start_char=sc if isinstance(sc, int) else None,
                                end_char=ec if isinstance(ec, int) else None,
                            )
                        )

                    enc = tokenizer(
                        sub_texts,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=max_length,
                        return_offsets_mapping=True,
                    )
                    input_ids_batch: list[list[int]] = enc["input_ids"]
                    offsets_batch: list[list[tuple[int, int]]] = enc["offset_mapping"]
                    processed_input_tokens += sum(len(x) for x in input_ids_batch)

                    keep_masks: list[list[bool]] = []
                    for offs, spans in zip(offsets_batch, spans_per_ex):
                        keep_masks.append(_token_keep_mask(offs, spans))

                    t_batch0 = time.time()
                    out = engine.generate(
                        input_ids=input_ids_batch,
                        sampling_params=[sampling] * len(input_ids_batch),
                        return_logprob=[True] * len(input_ids_batch),
                        logprob_start_len=[0] * len(input_ids_batch),
                    )
                    t_batch1 = time.time()
                    if not isinstance(out, list):
                        out = [out]
                    if len(out) != len(input_ids_batch):
                        raise RuntimeError(f"Engine output batch size mismatch: got {len(out)} expected {len(input_ids_batch)}")

                    for i, req_out in enumerate(out):
                        meta = req_out.get("meta_info", {}) if isinstance(req_out, dict) else {}
                        token_logprobs = meta.get("input_token_logprobs") or []
                        ids_tok = input_ids_batch[i]
                        keep = keep_masks[i]
                        n = min(len(token_logprobs), len(ids_tok), len(keep))

                        if not did_logprob_debug:
                            print(
                                "[dbg] "
                                f"len(input_ids)={len(ids_tok)} len(input_token_logprobs)={len(token_logprobs)} "
                                f"type(entry0)={type(token_logprobs[0]).__name__ if token_logprobs else 'NONE'}",
                                flush=True,
                            )
                            if token_logprobs:
                                print(f"[dbg] entry0={token_logprobs[0]!r}", flush=True)
                                print(f"[dbg] entry1={token_logprobs[1]!r}" if len(token_logprobs) > 1 else "[dbg] entry1=<missing>", flush=True)
                            did_logprob_debug = True

                        nll_sum = 0.0
                        tok_count = 0
                        for ti in range(n):
                            if not keep[ti]:
                                continue
                            lp_f = _extract_logprob(token_logprobs[ti])
                            if lp_f is None or not math.isfinite(lp_f):
                                continue
                            nll_sum += -lp_f
                            tok_count += 1

                        nll_mean = (nll_sum / tok_count) if tok_count else math.nan
                        ppl = math.exp(min(30.0, nll_mean)) if tok_count else math.nan

                        processed_kept_tokens += tok_count
                        writer.add(
                            {
                                "id": str(sub_ids[i]),
                                "model": model_id,
                                "model_dir": model_dir,
                                "dataset": dataset_id,
                                "subdir": subdir or "",
                                "max_length": max_length,
                                "nll_sum": float(nll_sum) if tok_count else math.nan,
                                "nll_mean": float(nll_mean) if tok_count else math.nan,
                                "ppl": float(ppl) if tok_count else math.nan,
                                "completion_token_count": int(tok_count),
                            }
                        )

                    processed += len(out)
                    now = time.time()
                    if log_every_s > 0 and (now - t_last_log) >= log_every_s:
                        dt = max(1e-6, now - t0_run)
                        row_s = processed / dt
                        in_tok_s = processed_input_tokens / dt
                        kept_tok_s = processed_kept_tokens / dt
                        last_batch_s = max(0.0, t_batch1 - t_batch0)
                        print(
                            "[prog] "
                            f"rows={processed} in_tok={processed_input_tokens} kept_tok={processed_kept_tokens} "
                            f"rows/s={row_s:.2f} in_tok/s={in_tok_s:.0f} kept_tok/s={kept_tok_s:.0f} "
                            f"last_batch_s={last_batch_s:.3f}",
                            flush=True,
                        )
                        t_last_log = now
                    if max_records and processed >= max_records:
                        break
                if max_records and processed >= max_records:
                    break
            if max_records and processed >= max_records:
                break
    finally:
        try:
            engine.shutdown()
        except Exception:
            pass

    writer.flush()

    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "dataset_id": dataset_id,
        "subdir": subdir,
        "model_id": model_id,
        "model_dir": model_dir,
        "tp_size": tp_size,
        "max_length": max_length,
        "batch_size": batch_size,
        "processed": processed,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[ok] wrote {out_dir}/run_manifest.json", flush=True)

    data_volume.commit()
    model_volume.commit()
    hf_cache_volume.commit()
    flashinfer_cache_volume.commit()

    return str(out_dir)


@app.local_entrypoint()
def main() -> None:
    prefetch_only = (os.environ.get("PREFETCH_ONLY") or "").strip().lower() in {"1", "true", "yes", "on"}
    prefetch_model.remote()
    prefetch_candidates.remote()
    if prefetch_only:
        print("[ok] prefetch complete (PREFETCH_ONLY=1)")
        return
    out_dir = score_nll.remote()
    print(out_dir)
