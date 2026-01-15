"""
Collect EAFT diagnostics for a single model (no comparison).

Why this exists:
- We want to run each model once on GPU (Modal), save a JSON artifact,
  and later compare any two models on CPU via a dynamic HTML dashboard.

Outputs:
  - artifacts/eaft_models/<run_id>/<model_slug>.json

Run (GPU, always log to unsloth_logs/):
  mkdir -p unsloth_logs
  ts=$(date +%Y%m%d_%H%M%S)
  nohup env MODAL_PROFILE=phamtrinhkien1203 GPU_TYPE=H100:1 \
    modal run modal/collect_calib_packs_eaft_single.py \
      --model-id openai/gpt-oss-20b \
      --seq-lens-csv 65536,131072 \
      --num-blocks 4 --batch-size 1 \
    > "unsloth_logs/eaft_single_${ts}.log" 2>&1 &
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable, Iterable

import modal


APP_NAME = "collect-calib-packs-eaft-single"

_KAGGLE_WORKDIR = Path("/kaggle/working")


def _default_eaft_cache_root() -> Path:
    override = (os.environ.get("EAFT_CACHE_ROOT") or "").strip()
    if override:
        return Path(override)
    if _KAGGLE_WORKDIR.exists():
        return _KAGGLE_WORKDIR / "eaft_cache"
    return Path("/root")


_EAFT_CACHE_ROOT = _default_eaft_cache_root()
_HF_HOME_DIR = Path(os.environ.get("EAFT_HF_HOME", str(_EAFT_CACHE_ROOT / "hf_cache")))
_MODEL_DIR = Path(os.environ.get("EAFT_MODEL_DIR", str(_EAFT_CACHE_ROOT / "model")))
_DATA_DIR = Path(os.environ.get("EAFT_DATA_DIR", str(_EAFT_CACHE_ROOT / "data")))


def _ensure_transformers_sklearn_stub() -> None:
    """
    Kaggle images can ship a broken `scikit-learn` wheel (ABI mismatch vs NumPy),
    which can crash `import transformers` even though we don't use sklearn.

    Transformers 4.57.x may import `sklearn.metrics.roc_curve` from its
    generation utilities during lazy-init. If sklearn raises a non-ImportError
    (e.g. ValueError from a compiled extension), it bubbles up and breaks the
    run.

    Fix: if sklearn cannot be imported cleanly, register a tiny pure-Python
    stub so `from sklearn.metrics import roc_curve` succeeds.
    """
    try:
        import sklearn  # noqa: F401

        return
    except Exception:
        pass

    # Ensure child processes (e.g. SGLang engine workers) also see the stub by
    # placing a minimal `sklearn` package on disk and prepending it to
    # PYTHONPATH. In spawn-based multiprocessing, sys.modules does not carry
    # over, but PYTHONPATH does.
    try:
        override = (os.environ.get("VERSA_SKLEARN_STUB_DIR") or "").strip()
        if override:
            stub_root = Path(override)
        elif Path("/kaggle/working").exists():
            # Kaggle kernels always include CWD in sys.path, even for spawned
            # subprocesses. This is the most reliable way to shadow the broken
            # system sklearn wheel.
            stub_root = Path("/kaggle/working")
        else:
            stub_root = Path("/tmp/versa_sklearn_stub")

        (stub_root / "sklearn" / "metrics").mkdir(parents=True, exist_ok=True)
        (stub_root / "sklearn" / "__init__.py").write_text(
            "'''Minimal sklearn stub to avoid ABI crashes (Kaggle).'''\n"
            "__all__ = ['metrics']\n",
            encoding="utf-8",
        )
        (stub_root / "sklearn" / "metrics" / "__init__.py").write_text(
            "def roc_curve(*args, **kwargs):\n"
            "    raise ImportError('sklearn is unavailable (stubbed).')\n",
            encoding="utf-8",
        )
        prev = os.environ.get("PYTHONPATH", "")
        stub_str = str(stub_root)
        if prev:
            if not prev.split(":")[0] == stub_str and stub_str not in prev.split(":"):
                os.environ["PYTHONPATH"] = stub_str + ":" + prev
        else:
            os.environ["PYTHONPATH"] = stub_str

        # Kaggle's preinstalled SGLang env is often injected via PYTHONPATH.
        # If we created PYTHONPATH ourselves, ensure we don't accidentally drop it.
        kaggle_sglang = Path("/kaggle/usr/lib/fa3-pip-install-sglang")
        if kaggle_sglang.exists():
            parts = [p for p in os.environ.get("PYTHONPATH", "").split(":") if p]
            if str(kaggle_sglang) not in parts:
                os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + str(kaggle_sglang)
    except Exception:
        pass

    import sys
    import types

    import importlib.machinery

    sklearn_mod = types.ModuleType("sklearn")
    sklearn_mod.__path__ = []  # mark as package
    sklearn_mod.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None, is_package=True)
    metrics_mod = types.ModuleType("sklearn.metrics")
    metrics_mod.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics", loader=None, is_package=False)

    def roc_curve(*_args: Any, **_kwargs: Any) -> None:
        raise ImportError("sklearn is unavailable (stubbed).")

    metrics_mod.roc_curve = roc_curve  # type: ignore[attr-defined]
    sklearn_mod.metrics = metrics_mod  # type: ignore[attr-defined]
    sys.modules["sklearn"] = sklearn_mod
    sys.modules["sklearn.metrics"] = metrics_mod


def _maybe_load_repo_dotenv() -> None:
    try:
        dotenv_path = Path(__file__).resolve().parent.parent / ".env"
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

# Kaggle: Transformers can accidentally pull in TF/Keras (and their sklearn
# wrapper), which is frequently broken due to NumPy ABI mismatches. We only need
# tokenization + logprob scoring, so hard-disable TF/Flax/JAX integrations.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_JAX", "0")

# Kaggle: Triton sometimes ships non-executable tool stubs inside the bundled
# wheel; force a working `ptxas` from the CUDA toolkit when available.
try:
    if Path("/usr/local/cuda/bin/ptxas").exists():
        os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
except Exception:
    pass

# Kaggle: keep Triton cache writable and stable.
try:
    if Path("/kaggle/working").exists():
        os.environ.setdefault("TRITON_CACHE_DIR", "/kaggle/working/.triton_cache")
        Path(os.environ["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# Install the sklearn stub early so any subprocess which imports this module
# (e.g. SGLang engine workers) won't crash during Transformers lazy imports.
try:
    _ensure_transformers_sklearn_stub()
except Exception:
    pass

DEFAULT_DATASET_REPO = os.environ.get("CALIB_PACKS_DATASET", "radna0/harmony-qwen3-calib-packs-v2-20260113")
DEFAULT_PACK_FILES = [
    "packs/reasoning_style_10k_v2/reasoning_style_10k_v2.parquet",
    "tool_agentic_10k_v6.parquet",
    "packs/calib_prompt_10000_v2/calib_prompt_10000_v2.parquet",
]

DEFAULT_GPU_TYPE = os.environ.get("GPU_TYPE", "B200:1")

data_volume = modal.Volume.from_name("pruning-data", create_if_missing=True)
model_volume = modal.Volume.from_name("pruning-models", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("hf-cache-persistent", create_if_missing=True)

_secrets: list[modal.Secret] = []
if os.environ.get("HF_TOKEN"):
    _secrets.append(modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}))

BASE_IMAGE = "nvidia/cuda:12.8.0-devel-ubuntu24.04"
_repo_root = Path(__file__).resolve().parents[1]
_sglang_src_dir = _repo_root / "sglang-flashinfer" / "python" / "sglang"
_sgl_kernel_src_dir = _repo_root / "sglang-flashinfer" / "sgl-kernel" / "python" / "sgl_kernel"
image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.11")
    .apt_install(
        "git",
        "python3-dev",
        "build-essential",
        "curl",
        # SGLang / sgl-kernel runtime deps
        "libnuma-dev",
        "numactl",
    )
    .run_commands("python -m pip install -U pip setuptools wheel")
    .run_commands(
        "python -m pip install "
        "numpy==2.2.0 pyarrow==22.0.0 "
        "transformers==4.56.2 tokenizers safetensors "
        "hf_transfer huggingface-hub==0.34.0"
    )
    .run_commands(
        # SGLang runtime for teacher-forcing logprobs (no massive logits materialization).
        "python -m pip install 'sglang[all]'",
        # Keep torch consistent with CUDA 12.8 wheels.
        "python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128",
    )
    # Install SGLang from source (overlay python packages) like modal/verify_sglang_gptoss_transmla.py.
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
    .env({"SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1"})
)

app = modal.App(APP_NAME)


def _ensure_hf_env() -> None:
    os.environ.setdefault("HF_HOME", str(_HF_HOME_DIR))
    os.environ.setdefault("XDG_CACHE_HOME", str(_HF_HOME_DIR / ".cache"))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for p in (_HF_HOME_DIR, _HF_HOME_DIR / ".cache", _DATA_DIR, _MODEL_DIR):
        try:
            Path(p).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


def _get_hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    return tok.strip() if tok else None


def _snapshot_download_model(model_id: str) -> Path:
    from huggingface_hub import snapshot_download

    _ensure_hf_env()
    _ensure_transformers_sklearn_stub()
    token = _get_hf_token()
    cache_dir = _MODEL_DIR / ".hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Hard rule: never burn GPU time downloading weights if they were already
    # CPU-predownloaded into the persistent volume.
    local_dir = _MODEL_DIR / str(model_id)
    try:
        if local_dir.exists():
            # `local_dir` is usually a symlink to a HF snapshot folder.
            probe = local_dir / "config.json"
            if probe.exists():
                return Path(local_dir.resolve() if local_dir.is_symlink() else local_dir)
    except Exception:
        pass
    print(f"[*] snapshot_download(model) start: {model_id}", flush=True)
    # Keep logs alive during long downloads so we can see liveness on Modal.
    try:
        import threading

        stop_evt = threading.Event()

        def _ticker():
            t0 = time.time()
            while not stop_evt.wait(60.0):
                dt = time.time() - t0
                print(f"[*] snapshot_download(model) running: {model_id} elapsed_s={dt:.0f}", flush=True)

        th = threading.Thread(target=_ticker, daemon=True)
        th.start()
    except Exception:
        stop_evt = None  # type: ignore[assignment]
    snap = snapshot_download(
        repo_id=str(model_id),
        repo_type="model",
        cache_dir=str(cache_dir),
        token=token,
        resume_download=True,
        max_workers=8,
    )
    try:
        if stop_evt is not None:
            stop_evt.set()
    except Exception:
        pass
    print(f"[+] snapshot_download(model) done: {model_id} -> {snap}", flush=True)
    local_dir = _MODEL_DIR / str(model_id)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    if local_dir.exists() and local_dir.is_symlink():
        return Path(local_dir.resolve())
    if local_dir.exists() and not local_dir.is_symlink():
        return local_dir
    try:
        local_dir.symlink_to(snap, target_is_directory=True)
    except Exception:
        pass
    return Path(snap)


def _download_dataset_file(dataset_repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    _ensure_hf_env()
    _ensure_transformers_sklearn_stub()
    token = _get_hf_token()
    print(f"[*] hf_hub_download(dataset) start: {dataset_repo}::{filename}", flush=True)
    return Path(
        hf_hub_download(
            repo_id=str(dataset_repo),
            repo_type="dataset",
            filename=str(filename),
            token=token,
        )
    )


def _find_cached_dataset_file(dataset_repo: str, filename: str) -> Path | None:
    # Avoid burning GPU time on dataset downloads: if CPU predownload already
    # fetched the file into the HF cache volume, resolve the cached path
    # directly instead of calling hf_hub_download (which can still do network
    # metadata checks).
    try:
        cache_root = _HF_HOME_DIR / "hub"
        repo_dir = cache_root / f"datasets--{dataset_repo.replace('/', '--')}" / "snapshots"
        if not repo_dir.exists():
            return None
        # Prefer the newest snapshot dir.
        for snap in sorted(repo_dir.iterdir(), reverse=True):
            p = snap / filename
            if p.exists():
                return p
    except Exception:
        return None
    return None


def _resolve_dataset_file(dataset_repo: str, filename: str) -> Path:
    cached = _find_cached_dataset_file(str(dataset_repo), str(filename))
    if cached is not None:
        print(f"[+] using cached dataset file: {dataset_repo}::{filename} -> {cached}", flush=True)
        return cached
    p = _download_dataset_file(str(dataset_repo), str(filename))
    print(f"[+] hf_hub_download(dataset) done: {dataset_repo}::{filename} -> {p}", flush=True)
    return p


@app.function(
    image=image,
    cpu=16.0,
    timeout=21600,
    memory=262144,
    volumes={"/root/data": data_volume, "/root/model": model_volume, "/root/hf_cache": hf_cache_volume},
    secrets=_secrets,
)
def predownload_model(*, model_id: str) -> str:
    _ensure_hf_env()
    try:
        model_volume.reload()
        hf_cache_volume.reload()
    except Exception:
        pass
    p = _snapshot_download_model(str(model_id))
    try:
        model_volume.commit()
        hf_cache_volume.commit()
    except Exception:
        pass
    return str(p)


@app.function(
    image=image,
    cpu=16.0,
    timeout=21600,
    memory=262144,
    volumes={"/root/data": data_volume, "/root/model": model_volume, "/root/hf_cache": hf_cache_volume},
    secrets=_secrets,
)
def predownload_packs(*, dataset_repo: str, pack_files: list[str]) -> list[str]:
    _ensure_hf_env()
    try:
        data_volume.reload()
        hf_cache_volume.reload()
    except Exception:
        pass
    out: list[str] = []
    for f in pack_files:
        p = _download_dataset_file(str(dataset_repo), str(f))
        print(f"[+] hf_hub_download(dataset) done: {dataset_repo}::{f} -> {p}", flush=True)
        out.append(str(p))
    try:
        data_volume.commit()
        hf_cache_volume.commit()
    except Exception:
        pass
    return out


def _iter_parquet_texts(parquet_path: Path, *, text_column: str = "text") -> Iterable[str]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(parquet_path))
    schema_names = set(pf.schema_arrow.names)
    if text_column not in schema_names:
        raise RuntimeError(f"Missing {text_column!r} in parquet schema: {sorted(schema_names)}")
    for rg in range(pf.num_row_groups):
        tab = pf.read_row_group(rg, columns=[text_column])
        col = tab.column(text_column)
        for v in col.to_pylist():
            if isinstance(v, str) and v.strip():
                yield v


START_TAG = "<|start|>"
CHANNEL_TAG = "<|channel|>"
MESSAGE_TAG = "<|message|>"
END_TAG = "<|end|>"
CALL_TAG = "<|call|>"
RETURN_TAG = "<|return|>"


def _assistant_content_spans(text: str) -> list[tuple[int, int]]:
    i = 0
    n = len(text)
    spans: list[tuple[int, int]] = []
    while i < n:
        start = text.find(START_TAG, i)
        if start < 0:
            break
        role_start = start + len(START_TAG)
        msg_tag_pos = text.find(MESSAGE_TAG, role_start)
        if msg_tag_pos < 0:
            break
        channel_pos = text.find(CHANNEL_TAG, role_start, msg_tag_pos)
        if channel_pos >= 0:
            role = text[role_start:channel_pos].strip()
        else:
            role = text[role_start:msg_tag_pos].strip()
        content_start = msg_tag_pos + len(MESSAGE_TAG)
        end_pos = text.find(END_TAG, content_start)
        call_pos = text.find(CALL_TAG, content_start)
        return_pos = text.find(RETURN_TAG, content_start)
        candidates: list[int] = [p for p in (end_pos, call_pos, return_pos) if p >= 0]
        if not candidates:
            break
        delim_pos = min(candidates)
        if role == "assistant" or role.startswith("assistant "):
            spans.append((content_start, delim_pos))
        i = delim_pos + 1
    return spans


def _token_keep_mask(offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> list[bool]:
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


def _iter_gpt_oss_layers(model) -> list[Any]:
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None) if base is not None else None
    return list(layers) if layers is not None else []


def _apply_top_k(model, top_k: int) -> None:
    for layer in _iter_gpt_oss_layers(model):
        router = getattr(getattr(layer, "mlp", None), "router", None)
        if router is None:
            continue
        try:
            router.top_k = int(top_k)
        except Exception:
            pass
    try:
        model.config.num_experts_per_tok = int(top_k)
        model.config.experts_per_token = int(top_k)
    except Exception:
        pass


@dataclass(frozen=True)
class PackedBlocks:
    blocks_ids: list[list[int]]
    blocks_keep: list[list[bool]]
    rows_seen: int
    wall_s: float


def _pack_blocks(
    *,
    text_iter: Callable[[], Iterable[str]],
    tok,
    eos_id: int,
    seq_len: int,
    num_blocks: int,
) -> PackedBlocks:
    buf_ids: list[int] = []
    buf_keep: list[bool] = []
    blocks_ids: list[list[int]] = []
    blocks_keep: list[list[bool]] = []
    rows_seen = 0
    t0 = time.time()
    for text in text_iter():
        rows_seen += 1
        spans = _assistant_content_spans(text)
        enc = tok(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
        ids: list[int] = enc["input_ids"]
        offs: list[tuple[int, int]] = enc["offset_mapping"]
        keep = _token_keep_mask(offs, spans)
        buf_ids.extend(ids)
        buf_keep.extend(keep)
        buf_ids.append(int(eos_id))
        buf_keep.append(False)
        # NOTE: We build blocks of exactly `seq_len` tokens (not `seq_len+1`) so
        # SGLang's `context_length` can be set to `seq_len` without exceeding
        # the model's derived maximum (e.g., 131072). Teacher-forcing logprobs
        # are computed for tokens [1..seq_len-1] via logprob_start_len=1.
        while len(buf_ids) >= seq_len and len(blocks_ids) < num_blocks:
            block_i = buf_ids[:seq_len]
            block_k = buf_keep[:seq_len]
            block_k[0] = False
            blocks_ids.append(block_i)
            blocks_keep.append(block_k)
            del buf_ids[:seq_len]
            del buf_keep[:seq_len]
        if len(blocks_ids) >= num_blocks:
            break
    dt = time.time() - t0
    if len(blocks_ids) < num_blocks:
        raise RuntimeError(f"Only built {len(blocks_ids)}/{num_blocks} blocks (rows_seen={rows_seen}).")
    return PackedBlocks(blocks_ids=blocks_ids, blocks_keep=blocks_keep, rows_seen=rows_seen, wall_s=float(dt))


def _hist1d(values, *, bins: int, vmin: float, vmax: float) -> dict[str, Any]:
    import numpy as np

    if bins <= 0:
        raise ValueError("bins must be > 0")
    edges = np.linspace(float(vmin), float(vmax), int(bins) + 1, dtype=np.float64)
    counts, _ = np.histogram(values, bins=edges)
    return {
        "edges": [float(x) for x in edges.tolist()],
        "counts": [int(x) for x in counts.astype(np.int64).tolist()],
    }


def _hist2d(x, y, *, xbins: int, ybins: int, xmin: float, xmax: float, ymin: float, ymax: float) -> dict[str, Any]:
    import numpy as np

    x_edges = np.linspace(float(xmin), float(xmax), int(xbins) + 1, dtype=np.float64)
    y_edges = np.linspace(float(ymin), float(ymax), int(ybins) + 1, dtype=np.float64)
    counts, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
    counts_i64 = counts.astype(np.int64)
    return {
        "x_edges": [float(v) for v in x_edges.tolist()],
        "y_edges": [float(v) for v in y_edges.tolist()],
        "counts": [int(v) for v in counts_i64.reshape(-1).tolist()],  # row-major
        "xbins": int(xbins),
        "ybins": int(ybins),
    }


def _eaft_collect_for_plots(
    engine,
    *,
    blocks: PackedBlocks,
    batch_size: int,
    entropy_topk: int,
    cc_quantile: float,
    hist_xbins: int,
    hist_ybins: int,
    prob_scale: str,
    logp_min: float,
    logp_max: float,
    sample_points: int,
    tag: str = "",
) -> dict[str, Any]:
    import numpy as np

    meta_debug = os.environ.get("SGLANG_META_DEBUG", "0").lower() in ("1", "true", "yes", "y")
    try:
        progress_every_s = float(os.environ.get("EAFT_PROGRESS_EVERY_S", "30"))
    except Exception:
        progress_every_s = 30.0

    entropy_topk = int(entropy_topk)
    cc_quantile = float(cc_quantile)
    if not (0.0 < cc_quantile < 1.0):
        raise ValueError("cc_quantile must be in (0,1)")

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
            return _extract_logprob(entry[0])
        return None

    def _extract_topk_logprobs(entry: Any) -> list[float]:
        # SGLang has had multiple wire formats for `meta_info["input_token_logprobs"]`.
        # We try to be robust:
        # - Some versions return `[logprob, top_logprobs]` or `[logprob, token_id, top_logprobs]`
        # - Some return just `logprob`
        # - Some return `top_logprobs` embedded in a dict
        #
        # Goal: return logprobs for the top-k candidates (not including the scalar
        # realized token logprob unless it is itself part of the top-k list).
        if isinstance(entry, (list, tuple)) and entry:
            # Structured tuple/list where the first element is the realized token logprob,
            # and a later element contains the top-k list/dict.
            if isinstance(entry[0], (int, float)):
                for idx in range(1, min(len(entry), 4)):
                    cand_blob = entry[idx]
                    if isinstance(cand_blob, (list, tuple)):
                        out: list[float] = []
                        for cand in cand_blob[: max(1, int(entropy_topk))]:
                            lp = _extract_logprob(cand)
                            if lp is None or not math.isfinite(lp):
                                continue
                            out.append(float(lp))
                        if out:
                            return out
                    if isinstance(cand_blob, dict):
                        tlp = (
                            cand_blob.get("top_logprobs")
                            or cand_blob.get("top_logprob")
                            or cand_blob.get("topk_logprobs")
                            or cand_blob.get("top_logprobs_list")
                        )
                        if isinstance(tlp, (list, tuple)):
                            out = []
                            for cand in tlp[: max(1, int(entropy_topk))]:
                                lp = _extract_logprob(cand)
                                if lp is None or not math.isfinite(lp):
                                    continue
                                out.append(float(lp))
                            if out:
                                return out

            # If the list itself is a top-k list, its elements should themselves be structured.
            if isinstance(entry[0], (dict, list, tuple)):
                out = []
                for cand in entry[: max(1, int(entropy_topk))]:
                    lp = _extract_logprob(cand)
                    if lp is None or not math.isfinite(lp):
                        continue
                    out.append(float(lp))
                return out
        if isinstance(entry, dict):
            tlp = entry.get("top_logprobs") or entry.get("top_logprob") or entry.get("topk_logprobs")
            if isinstance(tlp, (list, tuple)):
                out = []
                for cand in tlp[: max(1, int(entropy_topk))]:
                    lp = _extract_logprob(cand)
                    if lp is None or not math.isfinite(lp):
                        continue
                    out.append(float(lp))
                return out
        return []

    def _entropy_from_topk_logprobs(lps: list[float]) -> float:
        if not lps:
            return 0.0
        probs = np.exp(np.array(lps, dtype=np.float64))
        s = float(probs.sum())
        if s <= 0:
            return 0.0
        probs = probs / s
        ent = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
        denom = float(max(1e-12, math.log(float(len(probs)))))
        return float(ent / denom)

    # Histograms + streaming stats (avoid storing per-token logits/arrays).
    prob_scale = str(prob_scale).lower().strip()
    if prob_scale not in ("linear", "log"):
        raise ValueError("prob_scale must be 'linear' or 'log'")
    x_min = 0.0 if prob_scale == "linear" else float(logp_min)
    x_max = 1.0 if prob_scale == "linear" else float(logp_max)
    x_edges = np.linspace(float(x_min), float(x_max), int(hist_xbins) + 1, dtype=np.float64)
    h_edges = np.linspace(0.0, 1.0, int(hist_ybins) + 1, dtype=np.float64)
    hist2d = np.zeros((int(hist_xbins), int(hist_ybins)), dtype=np.int64)
    hist_x = np.zeros((int(hist_xbins),), dtype=np.int64)
    hist_h = np.zeros((int(hist_ybins),), dtype=np.int64)

    total_nll = 0.0
    total_nll_sumsq = 0.0
    sum_prob = 0.0
    sum_entropy = 0.0
    kept_tokens = 0
    total_pred_tokens = 0

    # Reservoir sample points for interactive scatter.
    sample_points = int(sample_points)
    samples: list[list[float]] = []
    rng = np.random.default_rng(0)

    sampling: dict[str, Any] = {"temperature": 0, "max_new_tokens": 0}
    printed_format = False
    t0 = time.time()
    last_log = t0
    total_batches = max(1, int(math.ceil(len(blocks.blocks_ids) / max(1, int(batch_size)))))
    for start in range(0, len(blocks.blocks_ids), int(batch_size)):
        batch_ids = blocks.blocks_ids[start : start + int(batch_size)]
        batch_keep = blocks.blocks_keep[start : start + int(batch_size)]
        # Teacher forcing logprobs for all input tokens; we skip the first token.
        # For entropy we need top-k candidates per token. SGLang exposes this via
        # `top_logprobs_num` which returns meta_info["input_top_logprobs"].
        out = engine.generate(
            input_ids=batch_ids,
            sampling_params=[sampling] * len(batch_ids),
            return_logprob=[True] * len(batch_ids),
            logprob_start_len=[1] * len(batch_ids),
            top_logprobs_num=[int(entropy_topk)] * len(batch_ids),
        )
        if not isinstance(out, list):
            out = [out]
        for bi, req_out in enumerate(out):
            meta = req_out.get("meta_info", {}) if isinstance(req_out, dict) else {}
            token_logprobs = meta.get("input_token_logprobs") or []
            input_top_logprobs = meta.get("input_top_logprobs") or []
            if meta_debug and not printed_format:
                try:
                    print(
                        "[fmt] meta_info keys=",
                        sorted(list(meta.keys()))[:60],
                        flush=True,
                    )
                    if token_logprobs:
                        e0 = token_logprobs[0]
                        e1 = token_logprobs[1] if len(token_logprobs) > 1 else None
                        print(
                            "[fmt] entry0 type=",
                            type(e0).__name__,
                            "repr=",
                            repr(e0)[:800],
                            flush=True,
                        )
                        print(
                            "[fmt] entry1 type=",
                            type(e1).__name__ if e1 is not None else "None",
                            "repr=",
                            (repr(e1)[:800] if e1 is not None else "<missing>"),
                            flush=True,
                        )
                    if input_top_logprobs:
                        print(
                            "[fmt] input_top_logprobs[0] type=",
                            type(input_top_logprobs[0]).__name__,
                            "repr=",
                            repr(input_top_logprobs[0])[:800],
                            flush=True,
                        )
                    printed_format = True
                except Exception:
                    printed_format = True
            keep = batch_keep[bi][1:]  # align with logprob_start_len=1
            n = min(len(token_logprobs), len(keep))
            total_pred_tokens += n
            for ti in range(n):
                if not keep[ti]:
                    continue
                entry = token_logprobs[ti]
                lp = _extract_logprob(entry)
                if lp is None or not math.isfinite(lp):
                    continue
                p = float(math.exp(lp))
                topk_lps: list[float] = []
                if ti < len(input_top_logprobs):
                    cand_list = input_top_logprobs[ti]
                    if isinstance(cand_list, (list, tuple)):
                        for cand in cand_list[: max(1, int(entropy_topk))]:
                            if isinstance(cand, (list, tuple)) and cand:
                                try:
                                    topk_lps.append(float(cand[0]))
                                except Exception:
                                    pass
                            else:
                                lp2 = _extract_logprob(cand)
                                if lp2 is not None and math.isfinite(lp2):
                                    topk_lps.append(float(lp2))
                if not topk_lps:
                    topk_lps = _extract_topk_logprobs(entry)
                h = float(_entropy_from_topk_logprobs(topk_lps))

                # map x
                x = p if prob_scale == "linear" else float(math.log10(max(1e-12, p)))
                if x < x_min or x > x_max or h < 0.0 or h > 1.0:
                    continue

                kept_tokens += 1
                nll = -float(lp)
                total_nll += nll
                total_nll_sumsq += nll * nll
                sum_prob += p
                sum_entropy += h

                ix = int(np.searchsorted(x_edges, x, side="right") - 1)
                iy = int(np.searchsorted(h_edges, h, side="right") - 1)
                ix = max(0, min(ix, int(hist_xbins) - 1))
                iy = max(0, min(iy, int(hist_ybins) - 1))
                hist2d[ix, iy] += 1
                hist_x[ix] += 1
                hist_h[iy] += 1

                if sample_points > 0:
                    if len(samples) < sample_points:
                        samples.append([float(x), float(h)])
                    else:
                        j = int(rng.integers(0, kept_tokens))
                        if j < sample_points:
                            samples[j] = [float(x), float(h)]

        now = time.time()
        if progress_every_s > 0 and (now - last_log) >= progress_every_s:
            batch_i = int(start / max(1, int(batch_size))) + 1
            dt = max(1e-9, now - t0)
            tok_s_pred = float(total_pred_tokens) / dt
            kept_frac = float(kept_tokens) / float(max(1, total_pred_tokens))
            tag_s = f" {tag}" if tag else ""
            print(
                f"[progress]{tag_s} batch={batch_i}/{total_batches} "
                f"pred_tokens={total_pred_tokens} kept_tokens={kept_tokens} kept_frac={kept_frac:.3f} "
                f"tok_s_pred={tok_s_pred:.1f} elapsed_s={dt:.1f}",
                flush=True,
            )
            last_log = now

    dt = max(1e-9, time.time() - t0)
    if kept_tokens <= 0:
        raise RuntimeError("No kept tokens; cannot compute EAFT diagnostics.")

    # Quantile thresholds from histograms.
    def _quantile_from_hist(counts: np.ndarray, edges: np.ndarray, q: float) -> float:
        total = int(counts.sum())
        if total <= 0:
            return float(edges[0])
        target = q * total
        cum = 0
        for i, c in enumerate(counts.tolist()):
            cum += int(c)
            if cum >= target:
                return float(edges[i + 1])
        return float(edges[-1])

    x_thr = _quantile_from_hist(hist_x, x_edges, cc_quantile)
    h_thr = _quantile_from_hist(hist_h, h_edges, cc_quantile)

    # CC stats from 2D hist using bin midpoints.
    total_hist = int(hist2d.sum())
    cc_count = 0
    ll = lh = hl = hh = 0
    for ix in range(int(hist_xbins)):
        x_mid = 0.5 * (x_edges[ix] + x_edges[ix + 1])
        for iy in range(int(hist_ybins)):
            y_mid = 0.5 * (h_edges[iy] + h_edges[iy + 1])
            c = int(hist2d[ix, iy])
            if c <= 0:
                continue
            if x_mid <= x_thr and y_mid <= h_thr:
                cc_count += c
                ll += c
            elif x_mid <= x_thr and y_mid > h_thr:
                lh += c
            elif x_mid > x_thr and y_mid <= h_thr:
                hl += c
            else:
                hh += c

    cc_rate = float(cc_count / max(1, total_hist))
    mean_nll = float(total_nll / kept_tokens)
    var_nll = max(0.0, float(total_nll_sumsq / kept_tokens) - (mean_nll * mean_nll))
    ppl = float(math.exp(min(30.0, mean_nll)))
    mean_prob = float(sum_prob / kept_tokens)
    mean_entropy = float(sum_entropy / kept_tokens)

    x_q05 = _quantile_from_hist(hist_x, x_edges, 0.05)
    x_q50 = _quantile_from_hist(hist_x, x_edges, 0.50)
    x_q95 = _quantile_from_hist(hist_x, x_edges, 0.95)
    h_q05 = _quantile_from_hist(hist_h, h_edges, 0.05)
    h_q50 = _quantile_from_hist(hist_h, h_edges, 0.50)
    h_q95 = _quantile_from_hist(hist_h, h_edges, 0.95)

    return {
        "ppl": float(ppl),
        "kept_tokens": int(kept_tokens),
        "tok_s_pred": float(total_pred_tokens / dt),
        "mean_prob": float(mean_prob),
        "mean_entropy": float(mean_entropy),
        "mean_nll": float(mean_nll),
        "var_nll": float(var_nll),
        # For long-seq runs we estimate quantiles from histograms (streaming).
        "p_thr": float(x_thr if prob_scale == "linear" else math.pow(10.0, x_thr)),
        "h_thr": float(h_thr),
        "x_thr": float(x_thr),
        "p_q05": float(x_q05 if prob_scale == "linear" else math.pow(10.0, x_q05)),
        "p_q50": float(x_q50 if prob_scale == "linear" else math.pow(10.0, x_q50)),
        "p_q95": float(x_q95 if prob_scale == "linear" else math.pow(10.0, x_q95)),
        "h_q05": float(h_q05),
        "h_q50": float(h_q50),
        "h_q95": float(h_q95),
        "cc_rate": float(cc_rate),
        "cc_count": int(cc_count),
        "quadrants": {
            "LL": float(ll / max(1, total_hist)),
            "LH": float(lh / max(1, total_hist)),
            "HL": float(hl / max(1, total_hist)),
            "HH": float(hh / max(1, total_hist)),
        },
        "hist2d_x_H": {
            "x_edges": [float(v) for v in x_edges.tolist()],
            "y_edges": [float(v) for v in h_edges.tolist()],
            "counts": [int(v) for v in hist2d.reshape(-1).tolist()],
            "xbins": int(hist_xbins),
            "ybins": int(hist_ybins),
        },
        "hist1d_x": {"edges": [float(v) for v in x_edges.tolist()], "counts": [int(v) for v in hist_x.tolist()]},
        "hist1d_H": {"edges": [float(v) for v in h_edges.tolist()], "counts": [int(v) for v in hist_h.tolist()]},
        "samples": samples,
        "x_scale": prob_scale,
        "x_min": float(x_min),
        "x_max": float(x_max),
    }


@app.function(
    image=image,
    gpu=DEFAULT_GPU_TYPE,
    timeout=21600,
    cpu=16.0,
    memory=262144,
    volumes={"/root/data": data_volume, "/root/model": model_volume, "/root/hf_cache": hf_cache_volume},
    secrets=_secrets,
)
def collect_calib_packs_eaft_single(
    *,
    dataset_repo: str,
    pack_files: list[str],
    model_id: str,
    tp_size: int,
    attn_backend: str,
    quantization: str,
    trust_remote_code: bool,
    top_k: int,
    entropy_topk: int,
    cc_quantile: float,
    num_blocks: int,
    batch_size: int,
    hist_xbins: int,
    hist_ybins: int,
    prob_scale: str,
    logp_min: float,
    logp_max: float,
    sample_points: int,
    seq_lens: list[int],
    device_map: str,
    max_gpu_mem_gb: float,
    max_cpu_mem_gb: float,
    offload_folder: str,
) -> dict[str, Any]:
    _ensure_hf_env()
    _ensure_transformers_sklearn_stub()
    import sglang as sgl
    from transformers import AutoTokenizer
    try:
        model_volume.reload()
        hf_cache_volume.reload()
        data_volume.reload()
    except Exception:
        pass

    seq_lens = [int(s) for s in seq_lens if int(s) > 0]
    if not seq_lens:
        raise RuntimeError("seq_lens must be non-empty")

    model_dir = _snapshot_download_model(str(model_id))
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=bool(trust_remote_code))
    if not getattr(tok, "is_fast", False):
        raise RuntimeError("Need a fast tokenizer for return_offsets_mapping=True")
    eos = tok.eos_token_id
    if eos is None:
        raise RuntimeError("Tokenizer missing eos_token_id")

    pack_paths: dict[str, Path] = {}
    for f in pack_files:
        pack_paths[str(f)] = _resolve_dataset_file(str(dataset_repo), str(f))

    device_map = str(device_map or "").strip().lower()
    dm = {"": 0} if not device_map or device_map == "cuda" else device_map
    max_memory = None
    if float(max_gpu_mem_gb) > 0 or float(max_cpu_mem_gb) > 0:
        max_memory = {}
        if float(max_gpu_mem_gb) > 0:
            max_memory[0] = f"{float(max_gpu_mem_gb):.0f}GiB"
        if float(max_cpu_mem_gb) > 0:
            max_memory["cpu"] = f"{float(max_cpu_mem_gb):.0f}GiB"

    extra = {
        "torch_dtype": "auto",
        "device_map": dm,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if max_memory:
        extra["max_memory"] = max_memory
    if offload_folder:
        extra["offload_folder"] = str(offload_folder)
        extra["offload_state_dict"] = True

    # SGLang engine (logprob teacher-forcing) avoids materializing full vocab logits
    # for long sequences, which is what caused the 512GiB allocation attempt.
    tp_size = int(tp_size) if int(tp_size) > 0 else 1
    # Attention backend selection:
    # - H100 (SM90): FA3 is typically fastest/stable.
    # - B200 (SM100): FA3 is not supported; SGLang's GPT-OSS gating currently
    #   allows: triton/trtllm_mha/fa3/fa4. Use triton by default.
    #
    # Users can override with ATTN_BACKEND.
    try:
        import torch

        sm = int(torch.cuda.get_device_capability()[0])
    except Exception:
        sm = 0
    default_attn_backend = "trtllm_mha" if sm >= 10 else "fa3"
    attn_backend = (str(attn_backend).strip() if str(attn_backend).strip() else default_attn_backend)
    kv_cache_dtype = os.environ.get("KV_CACHE_DTYPE", "auto")
    sglang_dtype = os.environ.get("SGLANG_DTYPE", "bfloat16")
    quantization = str(quantization or "").strip()
    # Match model's derived max context length (e.g., 131072). We purposely do
    # not set context_length to seq_len+1.
    context_length = int(os.environ.get("CONTEXT_LENGTH", "0")) or max(seq_lens)
    # User requested: allow overwrite/truncation for safety at long context.
    os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")

    # Memory knobs (especially important for 120B @ 65K/131K on single GPU).
    # These follow the same conventions used in modal/sglang_nll_scoring.py.
    mem_fraction_static_env = (os.environ.get("MEM_FRACTION_STATIC") or "").strip()
    chunked_prefill_size_env = (os.environ.get("CHUNKED_PREFILL_SIZE") or "").strip()
    max_total_tokens_env = (os.environ.get("MAX_TOTAL_TOKENS") or "").strip()
    watchdog_timeout_env = (os.environ.get("SGLANG_WATCHDOG_TIMEOUT") or "").strip()

    engine_kwargs: dict[str, Any] = {
        "model_path": str(model_dir),
        "tp_size": tp_size,
        "attention_backend": attn_backend,
        "kv_cache_dtype": kv_cache_dtype,
        "dtype": sglang_dtype,
        "context_length": int(context_length),
        "max_running_requests": 1,
        "max_total_tokens": int(max_total_tokens_env) if max_total_tokens_env else max(8192, int(context_length + 1024)),
        "disable_cuda_graph": True,
        "allow_auto_truncate": True,
    }
    if bool(trust_remote_code):
        engine_kwargs["trust_remote_code"] = True
    if quantization:
        engine_kwargs["quantization"] = quantization
    if mem_fraction_static_env:
        try:
            engine_kwargs["mem_fraction_static"] = float(mem_fraction_static_env)
        except Exception:
            pass
    else:
        # Conservative default for very long contexts on single-GPU. SGLang's
        # internal heuristic is OK, but explicit helps avoid fragmentation/OOM.
        if int(context_length) >= 65536:
            engine_kwargs["mem_fraction_static"] = 0.86

    if chunked_prefill_size_env:
        try:
            engine_kwargs["chunked_prefill_size"] = int(chunked_prefill_size_env)
        except Exception:
            pass
    else:
        if int(context_length) >= 65536:
            engine_kwargs["chunked_prefill_size"] = 16384

    if watchdog_timeout_env:
        try:
            engine_kwargs["watchdog_timeout"] = float(watchdog_timeout_env)
        except Exception:
            pass

    engine = sgl.Engine(**engine_kwargs)

    def _eval_pack(name: str, text_it: Callable[[], Iterable[str]]) -> dict[str, Any]:
        out: dict[str, Any] = {"pack": name, "seq": {}}
        for seq_len in seq_lens:
            print(f"[*] pack={name} seq_len={seq_len} packing...", flush=True)
            blocks = _pack_blocks(
                text_iter=text_it,
                tok=tok,
                eos_id=int(eos),
                seq_len=int(seq_len),
                num_blocks=int(num_blocks),
            )
            print(
                f"[*] pack={name} seq_len={seq_len} rows_seen={blocks.rows_seen} wall_s={blocks.wall_s:.2f}",
                flush=True,
            )
            metrics = _eaft_collect_for_plots(
                engine,
                blocks=blocks,
                batch_size=int(batch_size),
                entropy_topk=int(entropy_topk),
                cc_quantile=float(cc_quantile),
                hist_xbins=int(hist_xbins),
                hist_ybins=int(hist_ybins),
                prob_scale=str(prob_scale),
                logp_min=float(logp_min),
                logp_max=float(logp_max),
                sample_points=int(sample_points),
                tag=f"{name} seq{seq_len}",
            )
            out["seq"][str(int(seq_len))] = {
                "rows_seen": int(blocks.rows_seen),
                "pack_wall_s": float(blocks.wall_s),
                "model": metrics,
            }
        return out

    results: list[dict[str, Any]] = []
    for f, p in pack_paths.items():
        name = Path(f).stem
        results.append(_eval_pack(name, lambda p=p: _iter_parquet_texts(p, text_column="text")))

    def union_iter() -> Iterable[str]:
        iters = [_iter_parquet_texts(pack_paths[str(f)], text_column="text") for f in pack_files]
        for row in zip_longest(*iters, fillvalue=None):
            for t in row:
                if isinstance(t, str) and t.strip():
                    yield t

    results.append(_eval_pack("UNION", lambda: union_iter()))
    try:
        engine.shutdown()
    except Exception:
        pass

    return {
        "meta": {
            "dataset_repo": str(dataset_repo),
            "pack_files": list(pack_files),
            "model_id": str(model_id),
            "gpu_type": str(DEFAULT_GPU_TYPE),
            "engine": "sglang",
            "top_k": int(top_k),
            "entropy_topk": int(entropy_topk),
            "cc_quantile": float(cc_quantile),
            "num_blocks": int(num_blocks),
            "batch_size": int(batch_size),
            "hist_xbins": int(hist_xbins),
            "hist_ybins": int(hist_ybins),
            "prob_scale": str(prob_scale),
            "logp_min": float(logp_min),
            "logp_max": float(logp_max),
            "sample_points": int(sample_points),
            "seq_lens": [int(s) for s in seq_lens],
            "axes": {"x": "p_t" if str(prob_scale) == "linear" else "log10(p_t)", "y": "H_topK/ln(K)"},
            "x_min": 0.0 if str(prob_scale) == "linear" else float(logp_min),
            "x_max": 1.0 if str(prob_scale) == "linear" else float(logp_max),
        },
        "packs": results,
    }


@app.local_entrypoint()
def main(
    dataset_repo: str = DEFAULT_DATASET_REPO,
    pack_files_csv: str = ",".join(DEFAULT_PACK_FILES),
    model_id: str = "",
    tp_size: int = 1,
    attn_backend: str = "",
    quantization: str = "",
    trust_remote_code: bool = False,
    top_k: int = 4,
    entropy_topk: int = 20,
    cc_quantile: float = 0.15,
    num_blocks: int = 4,
    batch_size: int = 1,
    hist_xbins: int = 160,
    hist_ybins: int = 120,
    prob_scale: str = "linear",
    logp_min: float = -12.0,
    logp_max: float = 0.0,
    sample_points: int = 2000,
    device_map: str = "",
    max_gpu_mem_gb: float = 0.0,
    max_cpu_mem_gb: float = 0.0,
    offload_folder: str = "/root/model/offload",
    seq_lens_csv: str = "65536,131072",
    predownload_only: bool = False,
    skip_predownload: bool = False,
):
    pack_files = [x.strip() for x in (pack_files_csv or "").split(",") if x.strip()]
    if not pack_files:
        raise SystemExit("Empty --pack-files-csv")
    if not model_id:
        raise SystemExit("Empty --model-id")

    seq_lens = [int(s.strip()) for s in str(seq_lens_csv).split(",") if s.strip()]
    if not seq_lens:
        raise SystemExit("Empty --seq-lens-csv")

    # Kaggle/Versa: keep offload paths writable (the default Modal path is /root).
    if str(offload_folder).rstrip("/") == "/root/model/offload":
        offload_folder = str(_MODEL_DIR / "offload")

    # CPU-first: download weights + packs into Modal volumes before any GPU container starts.
    if not bool(skip_predownload):
        print(f"[*] CPU predownload model: {model_id}", flush=True)
        predownload_model.remote(model_id=str(model_id))
        print(f"[*] CPU predownload packs: {dataset_repo} ({len(pack_files)} files)", flush=True)
        predownload_packs.remote(dataset_repo=str(dataset_repo), pack_files=pack_files)
        if bool(predownload_only):
            print("[+] Predownload complete (predownload-only).", flush=True)
            return

    res = collect_calib_packs_eaft_single.remote(
        dataset_repo=str(dataset_repo),
        pack_files=pack_files,
        model_id=str(model_id),
        tp_size=int(tp_size),
        attn_backend=str(attn_backend),
        quantization=str(quantization),
        trust_remote_code=bool(trust_remote_code),
        top_k=int(top_k),
        entropy_topk=int(entropy_topk),
        cc_quantile=float(cc_quantile),
        num_blocks=int(num_blocks),
        batch_size=int(batch_size),
        hist_xbins=int(hist_xbins),
        hist_ybins=int(hist_ybins),
        prob_scale=str(prob_scale),
        logp_min=float(logp_min),
        logp_max=float(logp_max),
        sample_points=int(sample_points),
        seq_lens=seq_lens,
        device_map=str(device_map),
        max_gpu_mem_gb=float(max_gpu_mem_gb),
        max_cpu_mem_gb=float(max_cpu_mem_gb),
        offload_folder=str(offload_folder),
    )

    run_id = time.strftime("%Y%m%d_%H%M%S")
    safe_name = str(model_id).replace("/", "_")
    out_dir = Path(__file__).resolve().parents[1] / "artifacts" / "eaft_models" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / f"{safe_name}.json"
    # Attach run_id so a dashboard can disambiguate multiple runs of the same model_id.
    try:
        if isinstance(res, dict):
            meta = res.get("meta") if isinstance(res.get("meta"), dict) else None
            if meta is not None:
                meta.setdefault("run_id", run_id)
    except Exception:
        pass
    data_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[+] Wrote {data_path}")
