# modal/gpt_oss_pruning_track.py
#
# Pruning Track: GPT-OSS MoE expert profiling + soft pruning + structural pruning.
#
# This file is intentionally "analysis-first":
# - No big training runs
# - Focus on router/expert usage, cheap evals, and pruning IO feasibility

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterable

import modal

APP_NAME = "gpt-oss-pruning-track"

_KAGGLE_WORKDIR = Path("/kaggle/working")


def _default_pruning_cache_root() -> Path:
    override = (os.environ.get("PRUNING_CACHE_ROOT") or "").strip()
    if override:
        return Path(override)
    if _KAGGLE_WORKDIR.exists():
        # Kaggle: /kaggle/working is only ~20GB; keep large artifacts in /tmp.
        return Path("/tmp/harmony_pruning_cache")
    return Path("/root")


_PRUNING_CACHE_ROOT = _default_pruning_cache_root()
_HF_HOME_DIR = Path(os.environ.get("PRUNING_HF_HOME", str(_PRUNING_CACHE_ROOT / "hf_cache")))
_MODEL_DIR = Path(os.environ.get("PRUNING_MODEL_DIR", str(_PRUNING_CACHE_ROOT / "model")))
_DATA_DIR = Path(os.environ.get("PRUNING_DATA_DIR", str(_PRUNING_CACHE_ROOT / "data")))
_ARTIFACTS_DIR = Path(os.environ.get("PRUNING_ARTIFACTS_DIR", str(_MODEL_DIR / "artifacts")))


def _ensure_transformers_sklearn_stub() -> None:
    """
    Kaggle images can ship a broken `scikit-learn` wheel (ABI mismatch vs NumPy),
    which can crash `import transformers` even though we don't use sklearn.
    Ensure `from sklearn.metrics import roc_curve` succeeds via a tiny stub.
    """
    try:
        import sklearn  # noqa: F401

        return
    except Exception:
        pass

    # Put a minimal `sklearn` package on disk so spawned subprocesses can import it.
    try:
        stub_root = _KAGGLE_WORKDIR if _KAGGLE_WORKDIR.exists() else Path("/tmp/versa_sklearn_stub")
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
            parts = [p for p in prev.split(":") if p]
            if stub_str not in parts:
                os.environ["PYTHONPATH"] = stub_str + ":" + prev
        else:
            os.environ["PYTHONPATH"] = stub_str
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


os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_JAX", "0")

try:
    _ensure_transformers_sklearn_stub()
except Exception:
    pass

def _maybe_load_repo_dotenv() -> None:
    # Keep local runs reproducible without requiring manual `source .env` before
    # `modal run ...`. Never overrides already-set environment variables.
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

DEFAULT_DATASET_ID = os.environ.get("DATASET_ID", "radna0/harmony-nemotron-cpu-artifacts")
DEFAULT_DATASET_SPLIT = os.environ.get("DATASET_SPLIT", "train")
DEFAULT_TEXT_COLUMN = os.environ.get("TEXT_COLUMN", "text")
DEFAULT_DOMAIN = os.environ.get("DOMAIN", "")  # optional filter on `meta_domain`
DEFAULT_DOMAIN_COLUMN = os.environ.get("DOMAIN_COLUMN", "meta_domain")

# Domain-specific dataset defaults (used for math-targeted pruning without scanning `meta_domain`).
DEFAULT_MATH_DATASET_ID = os.environ.get("MATH_DATASET_ID", "radna0/nemotron-math-v2-harmony-tools")
DEFAULT_MATH_DATASET_SPLIT = os.environ.get("MATH_DATASET_SPLIT", "high_part00")
DEFAULT_MATH_TEXT_COLUMN = os.environ.get("MATH_TEXT_COLUMN", "text")

DEFAULT_20B_MODEL_ID = os.environ.get("MODEL_ID_20B", "openai/gpt-oss-20b")
DEFAULT_120B_MODEL_ID = os.environ.get("MODEL_ID_120B", "openai/gpt-oss-120b")
DEFAULT_20B_MODEL_DIR = (os.environ.get("MODEL_DIR_20B") or "").strip()

# Curated calibration packs (used for parity EAFT/PPL + recommended for pruning calibration).
DEFAULT_CALIB_PACKS_REPO = os.environ.get("CALIB_PACKS_DATASET", "radna0/harmony-qwen3-calib-packs-v2-20260113")
DEFAULT_CALIB_PACK_FILES = [
    "packs/reasoning_style_10k_v2/reasoning_style_10k_v2.parquet",
    "tool_agentic_10k_v6.parquet",
    "packs/calib_prompt_10000_v2/calib_prompt_10000_v2.parquet",
]

_secrets = []
if os.environ.get("HF_TOKEN"):
    _secrets.append(modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}))

# Persistent caches (per Modal profile)
data_volume = modal.Volume.from_name("pruning-data", create_if_missing=True)
model_volume = modal.Volume.from_name("pruning-models", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("hf-cache-persistent", create_if_missing=True)

BASE_IMAGE = "nvidia/cuda:12.8.0-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.12")
    .apt_install("git", "python3-dev", "build-essential", "curl")
    .run_commands("python -m pip install -U pip setuptools wheel")
    .run_commands(
        "python -m pip install "
        "torch==2.9.0 "
        "--extra-index-url https://download.pytorch.org/whl/cu128"
    )
    .run_commands(
        "python -m pip install "
        "numpy==2.2.0 datasets==3.2.0 accelerate==1.10.1 "
        "transformers==4.56.2 tokenizers safetensors "
        "pyarrow==21.0.0 pandas==2.2.3 "
        # Transformers MXFP4 integration for GPT-OSS.
        "kernels==0.11.7 "
        "hf_transfer huggingface-hub==0.34.0"
    )
)

# For REAP-lite saliency profiling we need access to plain PyTorch expert weights
# to compute per-expert output norms. The `kernels` package can replace the MoE
# MLP with a hub kernel (MegaBlocks) which hides/shims expert parameters.
# Use a dedicated image without `kernels` so GPT-OSS falls back to the reference
# PyTorch implementation.
image_no_kernels = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.12")
    .apt_install("git", "python3-dev", "build-essential", "curl")
    .run_commands("python -m pip install -U pip setuptools wheel")
    .run_commands(
        "python -m pip install "
        "torch==2.9.0 "
        "--extra-index-url https://download.pytorch.org/whl/cu128"
    )
    .run_commands(
        "python -m pip install "
        "numpy==2.2.0 datasets==3.2.0 accelerate==1.10.1 "
        "transformers==4.56.2 tokenizers safetensors "
        "pyarrow==21.0.0 pandas==2.2.3 "
        "hf_transfer huggingface-hub==0.34.0"
    )
    .run_commands("python -m pip uninstall -y kernels || true")
)

app = modal.App(APP_NAME)

def _local_mode_enabled() -> bool:
    # When this file runs inside Kaggle (via Versa / remote Jupyter), we must
    # not submit Modal jobs. Instead, execute Modal-decorated functions in-process
    # via `.local()` so the exact same codepath runs on the Kaggle GPU/CPU.
    flag = (os.environ.get("PRUNING_LOCAL_MODE") or "").strip().lower()
    if flag in ("1", "true", "yes", "y"):
        return True
    try:
        return _KAGGLE_WORKDIR.exists()
    except Exception:
        return False


_PRUNING_LOCAL_MODE = _local_mode_enabled()


def _invoke(fn: Any, *args: Any, **kwargs: Any) -> Any:
    if _PRUNING_LOCAL_MODE and hasattr(fn, "local"):
        return fn.local(*args, **kwargs)
    return fn.remote(*args, **kwargs)


def _ensure_hf_env() -> None:
    os.environ.setdefault("HF_HOME", str(_HF_HOME_DIR))
    os.environ.setdefault("XDG_CACHE_HOME", str(_HF_HOME_DIR / ".cache"))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    # Xet can attempt to write logs under HF_HOME/xet/logs, which sometimes fails
    # under volume mounts. We don't need Xet for this pruning track.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for p in (
        str(_HF_HOME_DIR),
        str(_HF_HOME_DIR / ".cache"),
        str(_DATA_DIR),
        str(_MODEL_DIR),
        str(_ARTIFACTS_DIR),
    ):
        try:
            Path(p).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


def _get_hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    return tok.strip() if tok else None


def _parse_csv(s: str) -> list[str]:
    out: list[str] = []
    for part in (s or "").split(","):
        p = part.strip()
        if p:
            out.append(p)
    return out


def _snapshot_download_model(model_id: str) -> Path:
    from huggingface_hub import snapshot_download

    _ensure_hf_env()
    token = _get_hf_token()

    cache_dir = _MODEL_DIR / ".hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Kaggle/Versa: allow using a pre-mounted local model directory (no hub access).
    # - If MODEL_DIR_20B is set and model_id matches the configured 20B ID, prefer it.
    # - If `model_id` itself is a valid local path, accept it.
    try:
        if DEFAULT_20B_MODEL_DIR and str(model_id) == str(DEFAULT_20B_MODEL_ID):
            p = Path(DEFAULT_20B_MODEL_DIR)
            if (p / "config.json").exists():
                return p
        p2 = Path(str(model_id))
        if p2.exists() and (p2 / "config.json").exists():
            return p2
    except Exception:
        pass
    # Avoid re-downloading in GPU containers: if the model was already
    # predownloaded into the persistent volume, use it directly.
    local_dir = _MODEL_DIR / str(model_id)
    try:
        if local_dir.exists():
            probe = local_dir / "config.json"
            if probe.exists():
                return Path(local_dir.resolve() if local_dir.is_symlink() else local_dir)
    except Exception:
        pass
    snap = snapshot_download(
        repo_id=str(model_id),
        repo_type="model",
        cache_dir=str(cache_dir),
        token=token,
        resume_download=True,
    )
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


@app.function(
    image=image,
    timeout=21600,
    cpu=4.0,
    memory=32768,
    volumes={
        "/root/model": model_volume,
        "/root/hf_cache": hf_cache_volume,
    },
    secrets=_secrets,
)
def predownload_model(model_id: str) -> str:
    """CPU-only: download a HF model snapshot into the persistent volumes."""
    _ensure_hf_env()
    try:
        model_volume.reload()
        hf_cache_volume.reload()
    except Exception:
        pass
    model_dir = _snapshot_download_model(str(model_id))
    try:
        model_volume.commit()
        hf_cache_volume.commit()
    except Exception:
        pass
    return str(model_dir)


@app.function(
    image=image,
    timeout=21600,
    cpu=16.0,
    memory=65536,
    volumes={
        "/root/data": data_volume,
        "/root/hf_cache": hf_cache_volume,
    },
    secrets=_secrets,
)
def predownload_calib_packs_cpu(*, dataset_repo: str, pack_files: list[str]) -> dict[str, Any]:
    """CPU-only: download calib pack parquet files into the HF cache volume."""
    from huggingface_hub import hf_hub_download

    _ensure_hf_env()
    try:
        data_volume.reload()
        hf_cache_volume.reload()
    except Exception:
        pass

    token = _get_hf_token()
    repo = str(dataset_repo)
    files = [str(p) for p in (pack_files or []) if str(p).strip()]
    if not files:
        raise ValueError("pack_files must be a non-empty list.")

    paths: list[str] = []
    for pf in files:
        local = hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            filename=str(pf),
            token=token,
        )
        paths.append(str(local))

    try:
        hf_cache_volume.commit()
        data_volume.commit()
    except Exception:
        pass

    return {"dataset_repo": repo, "pack_files": files, "downloaded_paths": paths}


@app.function(
    image=image,
    timeout=21600,
    cpu=2.0,
    memory=8192,
    volumes={"/root/hf_cache": hf_cache_volume},
    secrets=_secrets,
)
def read_model_cfg_meta(model_id: str) -> dict[str, int]:
    """Remote helper: read minimal config metadata for a model."""
    from huggingface_hub import hf_hub_download

    _ensure_hf_env()
    try:
        hf_cache_volume.reload()
    except Exception:
        pass
    token = _get_hf_token()
    cfg_path = hf_hub_download(str(model_id), filename="config.json", token=token)
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    num_layers = int(cfg.get("num_hidden_layers") or 0)
    num_experts = int(cfg.get("num_local_experts") or 0)
    return {"num_layers": num_layers, "num_experts": num_experts}


def _stream_text_rows(
    *,
    dataset_id: str,
    split: str,
    text_column: str,
    limit: int,
    domain: str = "",
    domain_column: str = "meta_domain",
) -> list[str]:
    from datasets import load_dataset

    _ensure_hf_env()
    token = _get_hf_token()

    ds = load_dataset(
        str(dataset_id),
        split=str(split),
        streaming=True,
        token=token,
    )
    out: list[str] = []
    for row in ds:
        if len(out) >= int(limit):
            break
        if domain:
            try:
                if str(row.get(domain_column, "")).strip() != str(domain).strip():
                    continue
            except Exception:
                continue
        if text_column not in row:
            raise KeyError(
                f"Dataset row missing text column {text_column!r}. Available keys: {sorted(row.keys())}"
            )
        text = row[text_column]
        if isinstance(text, str) and text.strip():
            out.append(text)
    if len(out) < int(limit):
        raise RuntimeError(f"Only got {len(out)} rows from dataset; expected {limit}.")
    return out


def _stream_rows_for_reap(
    *,
    dataset_id: str,
    split: str,
    text_column: str,
    limit: int,
    domain: str = "",
    domain_column: str = "meta_domain",
) -> list[dict[str, Any]]:
    """
    Stream rows for REAP-lite saliency profiling.

    Returns dicts with at least: {"id": <optional>, "text": <str>, "meta_domain": <optional>}
    """
    from datasets import load_dataset

    _ensure_hf_env()
    token = _get_hf_token()

    ds = load_dataset(
        str(dataset_id),
        split=str(split),
        streaming=True,
        token=token,
    )
    out: list[dict[str, Any]] = []
    for row in ds:
        if len(out) >= int(limit):
            break
        if domain:
            try:
                if str(row.get(domain_column, "")).strip() != str(domain).strip():
                    continue
            except Exception:
                continue
        text = row.get(text_column)
        if not isinstance(text, str) or not text.strip():
            continue
        out.append(
            {
                "id": row.get("id"),
                "text": text,
                "meta_domain": row.get(domain_column),
            }
        )
    if len(out) < int(limit):
        raise RuntimeError(f"Only got {len(out)} rows from dataset; expected {limit}.")
    return out


def _load_rows_jsonl(path: str, *, limit: int | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    p = Path(str(path))
    if not p.exists():
        raise FileNotFoundError(f"rows_jsonl_path not found: {p}")
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        text = row.get("text")
        if isinstance(text, str) and text.strip():
            out.append(row)
        if limit is not None and len(out) >= int(limit):
            break
    if not out:
        raise RuntimeError(f"No usable rows loaded from {p}")
    return out


@app.function(
    image=image,
    timeout=21600,
    cpu=16.0,
    memory=65536,
    volumes={
        "/root/data": data_volume,
        "/root/hf_cache": hf_cache_volume,
    },
    secrets=_secrets,
)
def sample_calib_packs_rows_cpu(
    *,
    dataset_repo: str,
    pack_files: list[str],
    text_column: str,
    num_rows: int,
    seed: int,
    strategy: str = "per_file",
    pack_weights: list[float] | None = None,
) -> dict[str, Any]:
    """
    Deterministically sample `num_rows` rows from a set of parquet pack files in
    a HF dataset repo.

    Sampling strategies:
    - per_file (default): allocate ~equal quota per pack file, then merge.
      This prevents one pack dominating the pruning signal.
    - per_file_weighted: like per_file, but allocate per-file quotas according to
      `pack_weights` (same order as `pack_files`). Useful when one pack's
      downstream metric (e.g. tool/agentic JS2D) dominates and must be preserved.
    - global: a single global min-hash sample across all files.

    Writes a JSONL file into the data volume and returns its path.
    Each JSONL row has at least: {"text": <harmony packed str>, "source_file": <pack path>}.
    """
    import hashlib
    import heapq
    import math

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    _ensure_hf_env()
    try:
        data_volume.reload()
        hf_cache_volume.reload()
    except Exception:
        pass

    token = _get_hf_token()
    repo = str(dataset_repo)
    files = [str(p) for p in (pack_files or []) if str(p).strip()]
    if not files:
        raise ValueError("pack_files must be a non-empty list.")

    num_rows = int(num_rows)
    if num_rows <= 0:
        raise ValueError("num_rows must be > 0.")
    seed = int(seed)
    text_col = str(text_column or "text")
    strategy = str(strategy or "per_file").strip().lower()
    if strategy not in ("per_file", "per_file_weighted", "global"):
        raise ValueError("strategy must be one of: per_file, per_file_weighted, global")

    weights: list[float] = []
    if strategy == "per_file_weighted":
        raw = list(pack_weights or [])
        if len(raw) != len(files):
            raise ValueError("pack_weights must match pack_files length for per_file_weighted")
        for w in raw:
            try:
                wf = float(w)
            except Exception:
                wf = 0.0
            weights.append(max(0.0, wf))
        if sum(weights) <= 0:
            raise ValueError("pack_weights must sum to > 0 for per_file_weighted")

    # Keep a fixed-size "max heap" (implemented as a min-heap over negative scores).
    # Include a numeric tiebreaker so Python never tries to compare dicts.
    heap: list[tuple[int, int, dict[str, Any]]] = []
    scanned = 0
    kept = 0
    valid_by_file: dict[str, int] = {str(p): 0 for p in files}
    nonstr_by_file: dict[str, int] = {str(p): 0 for p in files}
    empty_by_file: dict[str, int] = {str(p): 0 for p in files}
    sample_types: dict[str, str] = {}

    def _score(s: str) -> int:
        # Use a stable 64-bit score to select the smallest hashes.
        h = hashlib.blake2b((str(seed) + "\n" + s).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "big", signed=False)

    def _push(heap_ref: list[tuple[int, int, dict[str, Any]]], score: int, row: dict[str, Any], cap: int) -> None:
        nonlocal kept
        if len(heap_ref) < int(cap):
            heapq.heappush(heap_ref, (-int(score), int(scanned), row))
            kept += 1
            return
        worst_score = -heap_ref[0][0]
        if int(score) < int(worst_score):
            heapq.heapreplace(heap_ref, (-int(score), int(scanned), row))

    # Compute per-file caps for stratified sampling.
    per_file_cap = int(num_rows)
    per_file_caps: dict[str, int] = {}
    if strategy == "per_file":
        per_file_cap = int(math.ceil(float(num_rows) / float(len(files))))
        for p in files:
            per_file_caps[str(p)] = int(per_file_cap)
    elif strategy == "per_file_weighted":
        total_w = float(sum(weights))
        for p, w in zip(files, weights):
            cap = int(math.ceil(float(num_rows) * float(w) / total_w))
            per_file_caps[str(p)] = max(1, int(cap))

    # We keep small per-file heaps when stratifying, then merge into a global heap
    # of size num_rows to enforce the exact requested count.
    per_file_heaps: dict[str, list[tuple[int, int, dict[str, Any]]]] = {}

    for pack_path in files:
        local_path = hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            filename=str(pack_path),
            token=token,
        )
        pqf = pq.ParquetFile(local_path)
        row_i = 0
        for batch in pqf.iter_batches(batch_size=8192, columns=[text_col]):
            col = batch.column(0)
            for j in range(batch.num_rows):
                scanned += 1
                try:
                    text = col[j].as_py()
                except Exception:
                    row_i += 1
                    continue
                if not isinstance(text, str):
                    nonstr_by_file[str(pack_path)] = int(nonstr_by_file.get(str(pack_path), 0)) + 1
                    if str(pack_path) not in sample_types:
                        sample_types[str(pack_path)] = str(type(text))
                    row_i += 1
                    continue
                if not text.strip():
                    empty_by_file[str(pack_path)] = int(empty_by_file.get(str(pack_path), 0)) + 1
                    row_i += 1
                    continue
                valid_by_file[str(pack_path)] = int(valid_by_file.get(str(pack_path), 0)) + 1
                row = {"text": text, "source_file": str(pack_path), "row_i": int(row_i)}
                score = _score(text)
                if strategy == "global":
                    _push(heap, score, row, cap=num_rows)
                else:
                    h = per_file_heaps.setdefault(str(pack_path), [])
                    cap = int(per_file_caps.get(str(pack_path), per_file_cap))
                    _push(h, score, row, cap=cap)
                row_i += 1

    if strategy in ("per_file", "per_file_weighted"):
        for pack_path, h in per_file_heaps.items():
            for s, t, r in h:
                # `s` is negative score (max-heap stored as min-heap), restore.
                _push(heap, -int(s), r, cap=num_rows)

    if len(heap) < num_rows:
        raise RuntimeError(
            "Only sampled "
            f"{len(heap)} rows from calib packs; expected {num_rows}. scanned={scanned} "
            f"valid_by_file={valid_by_file} nonstr_by_file={nonstr_by_file} empty_by_file={empty_by_file} "
            f"sample_types={sample_types}"
        )

    # Sort by score asc for determinism.
    out_rows = [r for _, _, r in sorted([(-s, t, r) for (s, t, r) in heap], key=lambda x: (x[0], x[1]))]
    prompt_hash = hashlib.blake2b(
        ("\n".join(r["text"][:512] for r in out_rows)).encode("utf-8"), digest_size=16
    ).hexdigest()

    out_dir = _DATA_DIR / "calib_packs_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"calib_packs_sample_{prompt_hash}_n{num_rows}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    try:
        data_volume.commit()
        hf_cache_volume.commit()
    except Exception:
        pass

    return {
        "rows_jsonl_path": str(out_path),
        "dataset_repo": repo,
        "pack_files": files,
        "pack_weights": weights if strategy == "per_file_weighted" else None,
        "text_column": text_col,
        "num_rows": num_rows,
        "seed": seed,
        "strategy": strategy,
        "scanned": int(scanned),
        "prompt_hash": str(prompt_hash),
    }


def _compute_layer_neff(mass: list[float]) -> float:
    # Effective number of experts (inverse Simpson index).
    s = float(sum(float(x) for x in mass)) or 0.0
    if s <= 0:
        return 0.0
    p2 = 0.0
    for x in mass:
        px = float(x) / s
        p2 += px * px
    return float((1.0 / max(1e-12, p2)))


def _budgeted_keep_counts(
    *,
    mass_by_layer: list[list[float]],
    num_experts: int,
    keep_frac: float,
    min_keep: int,
    max_keep: int,
) -> list[int]:
    """
    Allocate a global keep budget across layers (no finetune) based on how
    "spread" saliency mass is within each layer.

    Output: keep_n per layer such that sum(keep_n) == round(keep_frac * L * E)
    with per-layer clamps [min_keep, max_keep].
    """
    L = len(mass_by_layer)
    if L <= 0:
        raise ValueError("mass_by_layer is empty")
    E = int(num_experts)
    if E <= 0:
        raise ValueError("num_experts must be > 0")
    keep_frac = float(keep_frac)
    if not (0.0 < keep_frac <= 1.0):
        raise ValueError("keep_frac must be in (0,1]")

    min_keep = max(1, min(int(min_keep), E))
    max_keep = max(min_keep, min(int(max_keep), E))

    budget = int(round(keep_frac * float(L * E)))
    budget = max(L * min_keep, min(L * max_keep, budget))

    neffs = [max(1e-6, _compute_layer_neff(m)) for m in mass_by_layer]
    mean_neff = float(sum(neffs)) / float(max(1, L))

    # Start near uniform, then scale by neff ratio.
    base = float(budget) / float(L)
    raw = []
    for neff in neffs:
        scale = float(neff) / max(1e-6, mean_neff)
        raw.append(base * scale)

    # Round + clamp, then fix sum by distributing deltas deterministically.
    keep = [max(min_keep, min(max_keep, int(round(x)))) for x in raw]
    cur = sum(keep)
    if cur == budget:
        return keep

    # Order layers by neff descending: add extra experts to the most spread layers first,
    # and remove from the most concentrated layers first.
    order_add = sorted(range(L), key=lambda i: neffs[i], reverse=True)
    order_sub = sorted(range(L), key=lambda i: neffs[i], reverse=False)

    if cur < budget:
        need = budget - cur
        i = 0
        while need > 0 and i < 10_000:
            li = order_add[i % L]
            if keep[li] < max_keep:
                keep[li] += 1
                need -= 1
            i += 1
    else:
        need = cur - budget
        i = 0
        while need > 0 and i < 10_000:
            li = order_sub[i % L]
            if keep[li] > min_keep:
                keep[li] -= 1
                need -= 1
            i += 1

    if sum(keep) != budget:
        raise RuntimeError(f"budgeting failed: got {sum(keep)} expected {budget}")
    return keep


def _read_reap_saliency_mass_parquet(path: Path, *, mass_column: str = "gate_norm_sum") -> dict[int, list[float]]:
    import pyarrow.parquet as pq

    mass_col = str(mass_column or "gate_norm_sum")
    table = pq.read_table(str(path), columns=["layer", "expert", mass_col])
    layers = table.column("layer").to_pylist()
    experts = table.column("expert").to_pylist()
    masses = table.column(mass_col).to_pylist()
    by_layer: dict[int, dict[int, float]] = {}
    max_expert = 0
    for li, ei, mi in zip(layers, experts, masses):
        li_i = int(li)
        ei_i = int(ei)
        max_expert = max(max_expert, ei_i)
        by_layer.setdefault(li_i, {})[ei_i] = float(mi)
    out: dict[int, list[float]] = {}
    for li, d in by_layer.items():
        vec = [0.0] * (max_expert + 1)
        for e, m in d.items():
            if 0 <= int(e) < len(vec):
                vec[int(e)] = float(m)
        out[int(li)] = vec
    return out


def _core_experts_from_eaft_parquet(
    path: Path,
    *,
    num_layers: int,
    num_experts: int,
    pos_top_m: int,
    count_top_m: int = 0,
) -> list[list[int]]:
    """
    Build a conservative per-layer "always keep" core from EAFT-REAP profiling.

    We intentionally base this on *positive* contribution mass (pos_gate_norm_sum)
    and optionally raw selection count, so we don't accidentally prune experts
    that are important on easy / high-confidence tokens.
    """
    import pyarrow.parquet as pq

    pos_top_m = max(0, min(int(pos_top_m), int(num_experts)))
    count_top_m = max(0, min(int(count_top_m), int(num_experts)))
    if pos_top_m == 0 and count_top_m == 0:
        return [[] for _ in range(int(num_layers))]

    table = pq.read_table(str(path), columns=["layer", "expert", "pos_gate_norm_sum", "count"])
    layers = table.column("layer").to_pylist()
    experts = table.column("expert").to_pylist()
    pos_mass = table.column("pos_gate_norm_sum").to_pylist()
    counts = table.column("count").to_pylist()

    by_layer: dict[int, list[tuple[int, float, int]]] = {}
    for li, ei, pm, c in zip(layers, experts, pos_mass, counts):
        by_layer.setdefault(int(li), []).append((int(ei), float(pm), int(c)))

    core: list[list[int]] = []
    for li in range(int(num_layers)):
        rows = by_layer.get(int(li), [])
        core_set: set[int] = set()
        if pos_top_m > 0:
            for e, _, _ in sorted(rows, key=lambda t: t[1], reverse=True)[:pos_top_m]:
                if 0 <= int(e) < int(num_experts):
                    core_set.add(int(e))
        if count_top_m > 0:
            for e, _, _ in sorted(rows, key=lambda t: t[2], reverse=True)[:count_top_m]:
                if 0 <= int(e) < int(num_experts):
                    core_set.add(int(e))
        core.append(sorted(core_set))
    return core


def _coverage_set(mass: list[float], cov: float) -> list[int]:
    cov = float(cov)
    if not mass:
        return []
    order = sorted(range(len(mass)), key=lambda e: float(mass[e]), reverse=True)
    total = float(sum(float(x) for x in mass)) or 0.0
    if total <= 0:
        return order[:0]
    kept: list[int] = []
    cum = 0.0
    for e in order:
        kept.append(int(e))
        cum += float(mass[e])
        if (cum / total) >= cov:
            break
    return kept


def _union_keep_sets(
    domain_masses: dict[str, dict[int, list[float]]],
    *,
    cov: float,
    weights: dict[str, float],
    core_n: int = 4,
    cap: int | None = None,
) -> list[list[int]]:
    # Determine number of layers and experts from the first domain.
    if not domain_masses:
        raise ValueError("domain_masses is empty")
    any_dom = next(iter(domain_masses.keys()))
    layers = sorted(domain_masses[any_dom].keys())
    keep_by_layer: list[list[int]] = []
    for li in layers:
        # Build union of per-domain coverage sets.
        union: set[int] = set()
        e_count = len(domain_masses[any_dom][li])
        # Global weighted saliency per expert.
        gmass = [0.0] * e_count
        for dom, masses_by_layer in domain_masses.items():
            mass = masses_by_layer.get(li, [0.0] * e_count)
            w = float(weights.get(dom, 1.0))
            for e in range(e_count):
                gmass[e] += w * float(mass[e])
            cov_set = _coverage_set(mass, cov=float(cov))
            union.update(int(e) for e in cov_set)

        core = sorted(range(e_count), key=lambda e: float(gmass[e]), reverse=True)[: max(0, int(core_n))]
        union.update(int(e) for e in core)

        if cap is not None and int(cap) > 0 and len(union) > int(cap):
            ranked = sorted(range(e_count), key=lambda e: float(gmass[e]), reverse=True)
            kept = set(ranked[: int(cap)])
            kept.update(int(e) for e in core)
            # If core blew past cap, keep it anyway (deterministic).
            union = kept

        keep_by_layer.append(sorted(union))
    return keep_by_layer


def _iter_gpt_oss_layers(model) -> list[Any]:
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None) if base is not None else None
    if layers is None:
        raise RuntimeError("Could not locate GPT-OSS layers at model.model.layers")
    return list(layers)


def _patch_router_soft_prune(
    *,
    router,
    allowed_experts: list[int] | None,
    top_k: int,
    num_experts: int,
):
    import torch

    orig_forward = router.forward

    allowed_mask = None
    if allowed_experts is not None:
        allowed_mask = torch.zeros((num_experts,), dtype=torch.bool)
        for e in allowed_experts:
            if 0 <= int(e) < num_experts:
                allowed_mask[int(e)] = True
        if int(allowed_mask.sum().item()) == 0:
            allowed_mask = None

    def _forward(*args, **kwargs):
        out = orig_forward(*args, **kwargs)
        if not isinstance(out, (tuple, list)) or len(out) != 2:
            return out
        a, b = out
        if not torch.is_tensor(a) or not torch.is_tensor(b):
            return out

        scores_first = True
        scores, indices = a, b
        if a.dtype in (torch.int32, torch.int64) and b.dtype.is_floating_point:
            scores, indices = b, a
            scores_first = False
        elif b.dtype in (torch.int32, torch.int64) and a.dtype.is_floating_point:
            scores, indices = a, b
            scores_first = True
        else:
            # Fallback to shape-based identification.
            try:
                if a.shape and a.shape[-1] == num_experts and b.shape and b.shape[-1] != num_experts:
                    scores, indices = a, b
                    scores_first = True
                elif b.shape and b.shape[-1] == num_experts and a.shape and a.shape[-1] != num_experts:
                    scores, indices = b, a
                    scores_first = False
            except Exception:
                pass

        if scores.dim() == 3:
            if int(scores.shape[-1]) != int(num_experts):
                return out
            scores_view = scores.reshape(-1, int(num_experts))
        elif scores.dim() == 2:
            if int(scores.shape[1]) != int(num_experts):
                return out
            scores_view = scores
        elif scores.dim() == 1:
            if int(scores.numel()) != int(num_experts):
                return out
            scores_view = scores.unsqueeze(0)
        else:
            return out

        if allowed_mask is not None:
            s_view = scores_view.clone()
            s_view[:, ~allowed_mask.to(device=s_view.device)] = 0
            denom = s_view.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            s_view = s_view / denom
        else:
            s_view = scores_view

        k = int(top_k)
        k = max(1, min(k, num_experts))
        idx_view = torch.topk(s_view, k=k, dim=-1).indices

        if scores.dim() == 3:
            scores_out = s_view.reshape(scores.shape)
        elif scores.dim() == 2:
            scores_out = s_view
        else:
            scores_out = s_view.squeeze(0)

        if indices.dim() == 3:
            bs, seq, _k0 = indices.shape
            idx_out = idx_view.reshape(int(bs), int(seq), int(k))
        elif indices.dim() == 2:
            idx_out = idx_view
        elif indices.dim() == 1:
            idx_out = idx_view.squeeze(0)
        else:
            idx_out = idx_view

        if scores_first:
            return scores_out, idx_out
        return idx_out, scores_out

    return orig_forward, _forward


# ---- Harmony parsing (completion-only masking) -------------------------------------

START_TAG = "<|start|>"
CHANNEL_TAG = "<|channel|>"
MESSAGE_TAG = "<|message|>"
END_TAG = "<|end|>"
CALL_TAG = "<|call|>"
RETURN_TAG = "<|return|>"


def _assistant_content_spans(text: str) -> list[tuple[int, int]]:
    # Minimal Harmony parser: same behavior as `harmony_text.py:assistant_content_spans`.
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


@dataclass(frozen=True)
class ProfileArgs:
    model_id: str
    dataset_id: str
    dataset_split: str
    text_column: str
    num_rows: int
    max_seq_length: int


@app.function(
    image=image_no_kernels,
    gpu="B200:1",
    timeout=21600,
    cpu=16.0,
    memory=262144,
    volumes={
        "/root/data": data_volume,
        "/root/model": model_volume,
        "/root/hf_cache": hf_cache_volume,
    },
    secrets=_secrets,
)
def profile_20b_expert_usage(
    model_id: str = DEFAULT_20B_MODEL_ID,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_split: str = DEFAULT_DATASET_SPLIT,
    text_column: str = DEFAULT_TEXT_COLUMN,
    domain: str = DEFAULT_DOMAIN,
    num_rows: int = 500,
    max_seq_length: int = 4096,
):
    import math

    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _ensure_hf_env()

    model_dir = _snapshot_download_model(model_id)
    texts = _stream_text_rows(
        dataset_id=dataset_id,
        split=dataset_split,
        text_column=text_column,
        limit=int(num_rows),
        domain=str(domain or ""),
    )

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype="auto",
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()

    layers = _iter_gpt_oss_layers(model)
    num_layers = len(layers)
    num_experts = int(getattr(model.config, "num_local_experts", 0) or 0)
    top_k = int(getattr(model.config, "num_experts_per_tok", 0) or getattr(model.config, "experts_per_token", 4))
    if num_experts <= 0:
        raise RuntimeError("Could not determine num_local_experts from config.")

    def _as_probs(scores: torch.Tensor) -> torch.Tensor:
        s = scores.float()
        # If already looks like probabilities, avoid softmax.
        try:
            row_sum = float(s.sum(dim=-1).mean().detach().cpu().item())
            s_min = float(s.min().detach().cpu().item())
            s_max = float(s.max().detach().cpu().item())
            if (s_min >= -1e-3) and (s_max <= 1.0 + 1e-3) and (0.98 <= row_sum <= 1.02):
                return s
        except Exception:
            pass
        return torch.softmax(s, dim=-1)

    hist = torch.zeros((num_layers, num_experts), dtype=torch.int64, device="cpu")
    coact = torch.zeros((num_layers, num_experts, num_experts), dtype=torch.int64, device="cpu")
    bins = 200
    prob_hist = torch.zeros((num_layers, bins), dtype=torch.int64, device="cpu")
    prob_sum = torch.zeros((num_layers,), dtype=torch.float64, device="cpu")
    prob_count = torch.zeros((num_layers,), dtype=torch.int64, device="cpu")
    tok_count = torch.zeros((num_layers,), dtype=torch.int64, device="cpu")
    top1_sum = torch.zeros((num_layers,), dtype=torch.float64, device="cpu")
    top2_sum = torch.zeros((num_layers,), dtype=torch.float64, device="cpu")

    hooks = []

    def _make_mlp_prehook(layer_idx: int):
        router = layers[layer_idx].mlp.router

        def _hook(_module, inputs):
            if not inputs:
                return
            hidden = inputs[0]
            if not torch.is_tensor(hidden):
                return
            hs = hidden
            if hs.dim() == 3:
                hs2 = hs.reshape(-1, int(hs.shape[-1]))
            elif hs.dim() == 2:
                hs2 = hs
            else:
                return

            out = router(hs2)
            if not isinstance(out, (tuple, list)) or len(out) != 2:
                return
            scores, idx = out
            if not torch.is_tensor(scores) or not torch.is_tensor(idx):
                return
            if idx.numel() == 0:
                return

            # [N, num_experts]
            if scores.dim() == 1:
                scores2 = scores.unsqueeze(0)
            elif scores.dim() == 2:
                scores2 = scores
            else:
                return
            if int(scores2.shape[-1]) != int(num_experts):
                return

            # [N, top_k]
            if idx.dim() == 1:
                idx2 = idx.unsqueeze(0)
            elif idx.dim() == 2:
                idx2 = idx
            else:
                return

            k = int(idx2.shape[1])
            if k <= 0:
                return

            probs = _as_probs(scores2)
            kk = max(1, min(int(k), 8))
            idxk = idx2[:, :kk].to(torch.int64)

            flat = idxk.reshape(-1)
            h = torch.bincount(flat, minlength=num_experts).to("cpu")
            hist[layer_idx] += h

            tok_count[layer_idx] += int(idxk.shape[0])
            for a in range(kk):
                for b in range(a + 1, kk):
                    lo = torch.minimum(idxk[:, a], idxk[:, b])
                    hi = torch.maximum(idxk[:, a], idxk[:, b])
                    pair_ids = lo * num_experts + hi
                    pc = torch.bincount(pair_ids, minlength=num_experts * num_experts).reshape(
                        num_experts, num_experts
                    )
                    coact[layer_idx] += pc.to("cpu")

            sel = probs.gather(1, idxk).reshape(-1)
            sel = torch.clamp(sel, 0.0, 1.0)
            if sel.numel():
                prob_sum[layer_idx] += float(sel.sum().item())
                prob_count[layer_idx] += int(sel.numel())
                bi = torch.clamp((sel * bins).to(torch.int64), 0, bins - 1)
                prob_hist[layer_idx] += torch.bincount(bi, minlength=bins).to("cpu")

            top_vals, _ = probs.gather(1, idxk).sort(dim=1, descending=True)
            top1_sum[layer_idx] += float(top_vals[:, 0].sum().item())
            if top_vals.shape[1] > 1:
                top2_sum[layer_idx] += float(top_vals[:, 1].sum().item())

        return _hook

    for li in range(num_layers):
        hooks.append(layers[li].mlp.register_forward_pre_hook(_make_mlp_prehook(li)))

    t0 = time.time()
    total_tokens = 0
    try:
        for i, text in enumerate(texts, start=1):
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=int(max_seq_length),
            )
            enc = {k: v.to("cuda") for k, v in enc.items()}
            with torch.inference_mode():
                _ = model(**enc, use_cache=False, return_dict=False)
            total_tokens += int(enc["input_ids"].numel())
            if i % 25 == 0:
                dt_i = max(1e-9, time.time() - t0)
                print(
                    f"[*] profile_20b_expert_usage rows={i}/{len(texts)} tokens={total_tokens} tok/s={total_tokens/dt_i:.0f}",
                    flush=True,
                )
    finally:
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

    dt = max(1e-9, time.time() - t0)
    toks_s = total_tokens / dt

    # Build a compact parquet summary (long-form).
    rows: list[dict[str, Any]] = []
    for li in range(num_layers):
        layer_hist = hist[li].tolist()
        for e, c in enumerate(layer_hist):
            rows.append(
                {
                    "kind": "usage_count",
                    "layer": int(li),
                    "expert_i": int(e),
                    "expert_j": None,
                    "value": float(c),
                }
            )
        for i in range(num_experts):
            for j in range(i, num_experts):
                c = int(coact[li, i, j].item())
                if c:
                    rows.append(
                        {
                            "kind": "coact_count",
                            "layer": int(li),
                            "expert_i": int(i),
                            "expert_j": int(j),
                            "value": float(c),
                        }
                    )
        mean_sel = float(prob_sum[li].item()) / max(1.0, float(prob_count[li].item()))
        rows.append(
            {
                "kind": "selected_prob_mean",
                "layer": int(li),
                "expert_i": None,
                "expert_j": None,
                "value": float(mean_sel),
            }
        )

        tok_n = int(tok_count[li].item())
        rows.append(
            {
                "kind": "selected_prob_top1_mean",
                "layer": int(li),
                "expert_i": None,
                "expert_j": None,
                "value": float(top1_sum[li].item()) / max(1.0, float(tok_n)),
            }
        )
        rows.append(
            {
                "kind": "selected_prob_top2_mean",
                "layer": int(li),
                "expert_i": None,
                "expert_j": None,
                "value": float(top2_sum[li].item()) / max(1.0, float(tok_n)),
            }
        )

        layer_prob_hist = prob_hist[li].tolist()
        for bi, c in enumerate(layer_prob_hist):
            if not c:
                continue
            rows.append(
                {
                    "kind": "selected_prob_hist",
                    "layer": int(li),
                    "expert_i": int(bi),
                    "expert_j": None,
                    "value": float(c),
                }
            )

    table = pa.Table.from_pylist(rows)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")

    top_experts = []
    for li in range(num_layers):
        counts = hist[li].tolist()
        order = sorted(range(num_experts), key=lambda e: counts[e], reverse=True)
        top_experts.append(order)

    return {
        "meta": {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "dataset_split": dataset_split,
            "text_column": text_column,
            "domain": str(domain or ""),
            "num_rows": int(num_rows),
            "max_seq_length": int(max_seq_length),
            "num_layers": int(num_layers),
            "num_experts": int(num_experts),
            "top_k": int(top_k),
            "total_tokens": int(total_tokens),
            "tokens_per_s": float(toks_s),
            "hist_total": int(hist.sum().item()),
            "prob_count_total": int(prob_count.sum().item()),
        },
        "expert_ranking_by_layer": top_experts,
        "usage_counts_by_layer": [hist[li].tolist() for li in range(num_layers)],
        "parquet_bytes": sink.getvalue().to_pybytes(),
    }


@app.function(
    image=image,
    timeout=21600,
    cpu=16.0,
    memory=65536,
    volumes={
        "/root/data": data_volume,
        "/root/hf_cache": hf_cache_volume,
    },
    secrets=_secrets,
)
def sample_domain_rows_cpu(
    *,
    dataset_id: str,
    dataset_split: str,
    text_column: str,
    domain: str,
    domain_column: str,
    num_rows: int,
    seed: int,
    max_scan_rows: int,
) -> dict[str, Any]:
    """
    Deterministically sample `num_rows` examples from a streaming HF dataset by
    keeping the rows with the smallest hash under `seed`.

    Writes a JSONL file into the data volume and returns its path.
    """
    import hashlib
    import heapq

    from datasets import load_dataset

    _ensure_hf_env()
    try:
        data_volume.reload()
        hf_cache_volume.reload()
    except Exception:
        pass

    token = _get_hf_token()
    ds = load_dataset(
        str(dataset_id),
        split=str(dataset_split),
        streaming=True,
        token=token,
    )

    dom = str(domain or "").strip()
    dom_col = str(domain_column or "").strip() or DEFAULT_DOMAIN_COLUMN
    num_rows = int(num_rows)
    seed = int(seed)
    max_scan_rows = int(max_scan_rows)

    heap: list[tuple[int, dict[str, Any]]] = []
    scanned = 0
    matched = 0

    def _score(row_id: Any, text: str) -> int:
        h = hashlib.sha256()
        h.update(str(seed).encode("utf-8"))
        h.update(b"\n")
        h.update(str(row_id).encode("utf-8", errors="ignore"))
        h.update(b"\n")
        h.update(text.encode("utf-8", errors="ignore"))
        return int.from_bytes(h.digest()[:8], "big", signed=False)

    for row in ds:
        scanned += 1
        if max_scan_rows > 0 and scanned > max_scan_rows:
            break
        if scanned % 50000 == 0:
            print(
                f"[*] sample_domain_rows_cpu domain={dom!r} scanned={scanned} matched={matched} heap={len(heap)}/{num_rows}",
                flush=True,
            )
        if dom:
            try:
                if str(row.get(dom_col, "")).strip() != dom:
                    continue
            except Exception:
                continue
        text = row.get(str(text_column))
        if not isinstance(text, str) or not text.strip():
            continue
        matched += 1
        s = _score(row.get("id"), text)
        payload = {
            "id": row.get("id"),
            "text": text,
            "meta_domain": row.get(dom_col),
            "score_u64": s,
        }
        if len(heap) < num_rows:
            heapq.heappush(heap, (-s, payload))
        else:
            if -s > heap[0][0]:
                heapq.heapreplace(heap, (-s, payload))

    if len(heap) < num_rows:
        raise RuntimeError(
            f"Only sampled {len(heap)}/{num_rows} rows for domain={dom!r} after scanning {scanned} rows "
            f"(matched={matched}). Increase max_scan_rows or verify domain values."
        )

    selected = [p for _neg, p in heap]
    selected.sort(key=lambda r: int(r.get("score_u64", 0)))

    prompt_hash = hashlib.sha256(
        ("\n".join(str(r.get("id") or "") for r in selected)).encode("utf-8", errors="ignore")
        + b"\n"
        + ("\n".join(r["text"] for r in selected)).encode("utf-8", errors="ignore")
    ).hexdigest()

    out_dir = _DATA_DIR / "reap_domain_samples" / str(dataset_id).replace("/", "__") / str(dataset_split)
    out_dir.mkdir(parents=True, exist_ok=True)
    dom_safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (dom or "all"))
    out_path = out_dir / f"{dom_safe}_seed{seed}_n{num_rows}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    try:
        data_volume.commit()
    except Exception:
        pass

    return {
        "dataset_id": str(dataset_id),
        "dataset_split": str(dataset_split),
        "text_column": str(text_column),
        "domain": str(dom),
        "domain_column": str(dom_col),
        "seed": int(seed),
        "num_rows": int(num_rows),
        "max_scan_rows": int(max_scan_rows),
        "scanned": int(scanned),
        "matched": int(matched),
        "prompt_hash": str(prompt_hash),
        "rows_jsonl_path": str(out_path),
    }


@app.function(
    image=image,
    timeout=21600,
    cpu=16.0,
    memory=65536,
    volumes={
        "/root/hf_cache": hf_cache_volume,
    },
    secrets=_secrets,
)
def scan_domain_counts_cpu(
    *,
    dataset_id: str,
    dataset_split: str,
    domain_column: str,
    max_scan_rows: int,
) -> dict[str, Any]:
    from collections import Counter

    from datasets import load_dataset

    _ensure_hf_env()
    try:
        hf_cache_volume.reload()
    except Exception:
        pass

    token = _get_hf_token()
    ds = load_dataset(
        str(dataset_id),
        split=str(dataset_split),
        streaming=True,
        token=token,
    )
    dom_col = str(domain_column or "").strip() or DEFAULT_DOMAIN_COLUMN
    max_scan_rows = int(max_scan_rows)
    scanned = 0
    counts: Counter[str] = Counter()
    missing = 0
    for row in ds:
        scanned += 1
        if max_scan_rows > 0 and scanned > max_scan_rows:
            break
        v = row.get(dom_col)
        if v is None:
            missing += 1
            continue
        counts[str(v).strip()] += 1
    return {
        "dataset_id": str(dataset_id),
        "dataset_split": str(dataset_split),
        "domain_column": str(dom_col),
        "max_scan_rows": int(max_scan_rows),
        "scanned": int(scanned),
        "missing": int(missing),
        "counts": dict(counts),
    }


@app.function(
    image=image_no_kernels,
    gpu="B200:1",
    timeout=21600,
    cpu=16.0,
    memory=262144,
    volumes={
        "/root/data": data_volume,
        "/root/model": model_volume,
        "/root/hf_cache": hf_cache_volume,
    },
    secrets=_secrets,
)
def profile_20b_reap_saliency(
    *,
    model_id: str,
    dataset_id: str,
    dataset_split: str,
    text_column: str,
    domain: str,
    num_rows: int,
    max_seq_length: int,
    batch_size: int,
    rows_jsonl_path: str = "",
) -> dict[str, Any]:
    """
    REAP-lite saliency profiling for GPT-OSS MoE (Transformers).

    For each layer, and each token in assistant spans only:
    - selected experts (top_k)
    - gate weight g_j(x) for selected experts
    - expert output norm ||f_j(x)||_2 for selected experts
    - saliency accumulation: g_j(x) * ||f_j(x)||_2
    """
    import hashlib
    import math

    # Mitigate allocator fragmentation for large batches.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _ensure_hf_env()

    try:
        data_volume.reload()
        model_volume.reload()
        hf_cache_volume.reload()
    except Exception:
        pass

    if str(rows_jsonl_path or "").strip():
        rows = _load_rows_jsonl(str(rows_jsonl_path), limit=int(num_rows))
    else:
        rows = _stream_rows_for_reap(
            dataset_id=dataset_id,
            split=dataset_split,
            text_column=text_column,
            limit=int(num_rows),
            domain=str(domain or ""),
            domain_column=DEFAULT_DOMAIN_COLUMN,
        )
    prompt_hash = hashlib.sha256(
        ("\n".join(str(r.get("id") or "") for r in rows)).encode("utf-8", errors="ignore")
        + b"\n"
        + ("\n".join(r["text"] for r in rows)).encode("utf-8", errors="ignore")
    ).hexdigest()

    model_dir = _snapshot_download_model(model_id)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # REAP-lite is meant to measure "generated" (assistant) tokens. For long
    # Harmony transcripts, assistant turns are often near the end; use left
    # truncation so we keep the tail instead of chopping it off.
    tokenizer.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype="auto",
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()

    layers = _iter_gpt_oss_layers(model)
    num_layers = len(layers)
    num_experts = int(getattr(model.config, "num_local_experts", 0) or 0)
    top_k = int(getattr(model.config, "num_experts_per_tok", 0) or getattr(model.config, "experts_per_token", 4))
    if num_experts <= 0 or top_k <= 0:
        raise RuntimeError("Could not determine num_local_experts / top_k from config.")

    # Global per-batch token mask used by hooks (CUDA bool [batch, seq]).
    current_keep_mask = {"mask": None}

    # GPU accumulators (small).
    count = torch.zeros((num_layers, num_experts), dtype=torch.int64, device="cuda")
    gate_sum = torch.zeros((num_layers, num_experts), dtype=torch.float32, device="cuda")
    norm_sum = torch.zeros((num_layers, num_experts), dtype=torch.float32, device="cuda")
    gate_norm_sum = torch.zeros((num_layers, num_experts), dtype=torch.float32, device="cuda")

    hooks = []

    def _make_mlp_prehook(layer_idx: int):
        router = layers[layer_idx].mlp.router
        experts = layers[layer_idx].mlp.experts

        def _hook(_module, inputs):
            if not inputs:
                return
            hidden = inputs[0]
            if not torch.is_tensor(hidden):
                return
            if hidden.dim() != 3:
                return

            keep_mask = current_keep_mask.get("mask")
            if keep_mask is None or not torch.is_tensor(keep_mask):
                return
            if keep_mask.shape[:2] != hidden.shape[:2]:
                return

            hs = hidden  # [bs, seq, hidden]
            bs, seq, hd = hs.shape
            hs2 = hs.reshape(-1, int(hd))
            keep2 = keep_mask.reshape(-1)
            if int(keep2.sum().item()) == 0:
                return
            hs_sel = hs2[keep2]

            # Cap tokens per batch to keep profiling cheap and avoid OOM from
            # computing per-expert outputs for all experts.
            max_tokens = int(os.environ.get("REAP_MAX_TOKENS_PER_BATCH", "128"))
            if max_tokens > 0 and int(hs_sel.shape[0]) > max_tokens:
                hs_sel = hs_sel[:max_tokens]

            out = router(hs_sel)
            if not isinstance(out, (tuple, list)) or len(out) != 2:
                return
            router_scores, router_idx = out
            if not torch.is_tensor(router_scores) or not torch.is_tensor(router_idx):
                return
            if router_scores.dim() != 2 or int(router_scores.shape[1]) != int(num_experts):
                return
            if router_idx.dim() != 2:
                return
            kk = int(router_idx.shape[1])
            if kk <= 0:
                return

            # Gather top-k weights for the selected experts.
            router_idx = router_idx.to(torch.int64)
            topk_w = router_scores.gather(1, router_idx).to(torch.float32)  # [N, kk]
            # Most MoE implementations renormalize top-k probabilities to sum to 1.
            denom = topk_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            topk_w = topk_w / denom

            gate_up_proj = getattr(experts, "gate_up_proj", None)
            down_proj = getattr(experts, "down_proj", None)
            if gate_up_proj is None or down_proj is None:
                return
            gate_up_proj_bias = getattr(experts, "gate_up_proj_bias", None)
            down_proj_bias = getattr(experts, "down_proj_bias", None)
            trace = os.environ.get("TRACE_REAP", "0").lower() in ("1", "true", "yes", "y")
            if trace and layer_idx == 0:
                try:
                    import importlib.util

                    print(
                        "[trace] kernels_installed=",
                        bool(importlib.util.find_spec("kernels")),
                        flush=True,
                    )
                except Exception:
                    pass
                print(
                    f"[trace] experts={experts.__class__.__name__} "
                    f"gate_up_proj_type={type(gate_up_proj)} down_proj_type={type(down_proj)} "
                    f"gate_up_bias_type={type(gate_up_proj_bias)} down_bias_type={type(down_proj_bias)}",
                    flush=True,
                )

            # REAP-lite requires access to raw PyTorch weights; if a hub kernel
            # swaps in a wrapper type, fail fast with a clear message.
            if not torch.is_tensor(gate_up_proj) or not torch.is_tensor(down_proj):
                raise TypeError(
                    "REAP-lite requires PyTorch expert weights, but got non-torch tensors for "
                    f"gate_up_proj={type(gate_up_proj)} down_proj={type(down_proj)}. "
                    "Try running on `image_no_kernels` and ensure `kernels` is not installed."
                )
            if gate_up_proj_bias is not None and not torch.is_tensor(gate_up_proj_bias):
                raise TypeError(
                    "REAP-lite requires PyTorch expert bias tensors, but got "
                    f"gate_up_proj_bias={type(gate_up_proj_bias)}."
                )
            if down_proj_bias is not None and not torch.is_tensor(down_proj_bias):
                raise TypeError(
                    "REAP-lite requires PyTorch expert bias tensors, but got "
                    f"down_proj_bias={type(down_proj_bias)}."
                )
            alpha = float(getattr(experts, "alpha", 1.702))
            limit = float(getattr(experts, "limit", 7.0))

            # Compute expert output norms ||f_j(x)||_2 for selected experts only.
            #
            # Shapes expected (Transformers GPT-OSS reference):
            # - gate_up_proj: [E, H, 2D]
            # - gate_up_proj_bias: [E, 2D] (optional)
            # - down_proj: [E, D, H]
            # - down_proj_bias: [E, H] (optional)
            n_tokens = int(hs_sel.shape[0])
            token_ids = torch.arange(n_tokens, device=hs_sel.device).repeat_interleave(kk)  # [N*kk]
            expert_ids = router_idx.reshape(-1)  # [N*kk]

            expert_ids_sorted, perm = torch.sort(expert_ids)
            token_ids_sorted = token_ids[perm]
            w_sorted = topk_w.reshape(-1)[perm]
            n_sorted = torch.empty_like(w_sorted, dtype=torch.float32)

            unique_e, counts_e = torch.unique_consecutive(expert_ids_sorted, return_counts=True)
            offset = 0
            for e_t, c_t in zip(unique_e.tolist(), counts_e.tolist()):
                e = int(e_t)
                c = int(c_t)
                tok = token_ids_sorted[offset : offset + c]
                x = hs_sel.index_select(0, tok)

                W_gu = gate_up_proj[e]
                W_down = down_proj[e]
                b_gu = gate_up_proj_bias[e] if gate_up_proj_bias is not None else None
                b_down = down_proj_bias[e] if down_proj_bias is not None else None

                gate_up = torch.matmul(x, W_gu)
                if b_gu is not None:
                    gate_up = gate_up + b_gu

                gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                gate = gate.clamp(max=limit)
                up = up.clamp(min=-limit, max=limit)
                glu = gate * torch.sigmoid(gate * alpha)
                act = (up + 1) * glu

                out_e = torch.matmul(act, W_down)
                if b_down is not None:
                    out_e = out_e + b_down
                n_sorted[offset : offset + c] = torch.linalg.norm(out_e.float(), dim=-1)
                offset += c

            norms_flat = torch.empty_like(n_sorted)
            norms_flat[perm] = n_sorted
            norms_sel = norms_flat.reshape(n_tokens, kk)
            flat_e = router_idx.reshape(-1)
            flat_w = topk_w.reshape(-1)
            flat_n = norms_sel.reshape(-1)

            ones = torch.ones_like(flat_e, dtype=torch.int64)
            count[layer_idx].index_add_(0, flat_e, ones)
            gate_sum[layer_idx].index_add_(0, flat_e, flat_w)
            norm_sum[layer_idx].index_add_(0, flat_e, flat_n)
            gate_norm_sum[layer_idx].index_add_(0, flat_e, flat_w * flat_n)

        return _hook

    for li in range(num_layers):
        hooks.append(layers[li].mlp.register_forward_pre_hook(_make_mlp_prehook(li)))

    t0 = time.time()
    total_tokens = 0
    total_kept_tokens = 0

    try:
        for batch_i, start in enumerate(range(0, len(rows), max(1, int(batch_size))), start=1):
            batch = rows[start : start + int(batch_size)]
            texts = [r["text"] for r in batch]

            tok = tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=int(max_seq_length),
                return_offsets_mapping=True,
            )
            offsets = tok.pop("offset_mapping")  # [bs, seq, 2] on CPU

            keep_masks: list[list[bool]] = []
            for row_i, text in enumerate(texts):
                spans = _assistant_content_spans(text)
                keep = _token_keep_mask(offsets[row_i].tolist(), spans)
                keep_masks.append(keep)

            enc = {k: v.to("cuda") for k, v in tok.items()}
            attn = enc.get("attention_mask")
            keep = torch.tensor(keep_masks, dtype=torch.bool, device="cuda")
            if attn is not None and torch.is_tensor(attn):
                keep = keep & attn.to(torch.bool)
            current_keep_mask["mask"] = keep

            with torch.inference_mode():
                _ = model(**enc, use_cache=False, return_dict=False)

            total_tokens += int(enc["input_ids"].numel())
            total_kept_tokens += int(keep.sum().item())
            if batch_i % 10 == 0:
                dt_i = max(1e-9, time.time() - t0)
                print(
                    f"[*] profile_20b_reap_saliency batches={batch_i} rows={min(len(rows), start+int(batch_size))}/{len(rows)} "
                    f"tokens={total_tokens} kept={total_kept_tokens} tok/s={total_tokens/dt_i:.0f}",
                    flush=True,
                )
    finally:
        current_keep_mask["mask"] = None
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

    dt = max(1e-9, time.time() - t0)
    toks_s = total_tokens / dt

    # Move to CPU for serialization.
    count_cpu = count.detach().to("cpu")
    gate_sum_cpu = gate_sum.detach().to("cpu")
    norm_sum_cpu = norm_sum.detach().to("cpu")
    gate_norm_sum_cpu = gate_norm_sum.detach().to("cpu")

    # Build ranking by mean saliency.
    ranking: list[list[int]] = []
    concentration: list[dict[str, float]] = []
    for li in range(num_layers):
        c = count_cpu[li].to(torch.float32)
        mean = gate_norm_sum_cpu[li] / c.clamp_min(1.0)
        order = torch.argsort(mean, descending=True).tolist()
        ranking.append([int(x) for x in order])
        mass = gate_norm_sum_cpu[li].to(torch.float64)
        mass_order = torch.argsort(mass, descending=True)
        total_mass = float(mass.sum().item()) or 1.0
        def _cov(n: int) -> float:
            nn = max(0, min(int(n), int(num_experts)))
            return float(mass[mass_order[:nn]].sum().item()) / total_mass
        concentration.append(
            {
                "top_4": _cov(4),
                "top_8": _cov(8),
                "top_16": _cov(16),
            }
        )

    # Build parquet summary (long-form).
    parquet_rows: list[dict[str, Any]] = []
    for li in range(num_layers):
        for e in range(num_experts):
            c = int(count_cpu[li, e].item())
            gsum = float(gate_sum_cpu[li, e].item())
            nsum = float(norm_sum_cpu[li, e].item())
            gnsum = float(gate_norm_sum_cpu[li, e].item())
            parquet_rows.append(
                {
                    "layer": int(li),
                    "expert": int(e),
                    "count": int(c),
                    "gate_sum": float(gsum),
                    "norm_sum": float(nsum),
                    "gate_norm_sum": float(gnsum),
                    "gate_mean": float(gsum / max(1.0, float(c))),
                    "norm_mean": float(nsum / max(1.0, float(c))),
                    "saliency_mean": float(gnsum / max(1.0, float(c))),
                }
            )

    table = pa.Table.from_pylist(parquet_rows)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")

    return {
        "meta": {
            "model_id": str(model_id),
            "dataset_id": str(dataset_id),
            "dataset_split": str(dataset_split),
            "text_column": str(text_column),
            "domain": str(domain or ""),
            "prompt_hash": str(prompt_hash),
            "num_rows": int(num_rows),
            "max_seq_length": int(max_seq_length),
            "batch_size": int(batch_size),
            "num_layers": int(num_layers),
            "num_experts": int(num_experts),
            "top_k": int(top_k),
            "total_tokens": int(total_tokens),
            "total_kept_tokens": int(total_kept_tokens),
            "tokens_per_s": float(toks_s),
        },
        "ranking_by_layer": ranking,
        "concentration_by_layer": concentration,
        "parquet_bytes": sink.getvalue().to_pybytes(),
    }


@app.function(
    image=image_no_kernels,
    gpu="B200:1",
    timeout=21600,
    cpu=16.0,
    memory=262144,
    volumes={
        "/root/data": data_volume,
        "/root/model": model_volume,
        "/root/hf_cache": hf_cache_volume,
    },
    secrets=_secrets,
)
def profile_20b_eaftreap_saliency(
    *,
    model_id: str,
    dataset_id: str,
    dataset_split: str,
    text_column: str,
    domain: str,
    num_rows: int,
    max_seq_length: int,
    batch_size: int,
    cc_quantile: float = 0.15,
    uncertain_quantile: float = 0.85,
    entropy_topk: int = 20,
    w_good: float = 1.0,
    w_uncertain: float = 0.25,
    w_conflict: float = -2.0,
    rows_jsonl_path: str = "",
) -> dict[str, Any]:
    """
    EAFT-REAP (correctness-aware) saliency profiling for GPT-OSS MoE (Transformers).

    REAP-lite uses: g_j(x) * ||f_j(x)||_2 (magnitude-only).
    EAFT-REAP conditions that contribution on token outcome quality:
      - p_t: probability assigned to the reference token
      - H_t: predictive entropy (Top-K approx, normalized by ln(K))

    We compute global thresholds over sampled tokens:
      p_lo = quantile(p_t, cc_quantile)
      H_lo = quantile(H_t, cc_quantile)
      H_hi = quantile(H_t, uncertain_quantile)

    Then per token we assign a weight w_t:
      - good: (p_t >= p_lo) & (H_t <= H_lo)                 => w_good
      - confident conflict: (p_t < p_lo) & (H_t <= H_lo)    => w_conflict (negative)
      - otherwise (including uncertain/hard):               => w_uncertain

    Expert score (per selection) is the signed mean:
      E[w_t * g_j(x) * ||f_j(x)||_2]
    """
    import hashlib
    import math

    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _ensure_hf_env()

    try:
        data_volume.reload()
        model_volume.reload()
        hf_cache_volume.reload()
    except Exception:
        pass

    if str(rows_jsonl_path or "").strip():
        rows = _load_rows_jsonl(str(rows_jsonl_path), limit=int(num_rows))
    else:
        rows = _stream_rows_for_reap(
            dataset_id=dataset_id,
            split=dataset_split,
            text_column=text_column,
            limit=int(num_rows),
            domain=str(domain or ""),
            domain_column=DEFAULT_DOMAIN_COLUMN,
        )
    prompt_hash = hashlib.sha256(
        ("\n".join(str(r.get("id") or "") for r in rows)).encode("utf-8", errors="ignore")
        + b"\n"
        + ("\n".join(r["text"] for r in rows)).encode("utf-8", errors="ignore")
    ).hexdigest()

    model_dir = _snapshot_download_model(model_id)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype="auto",
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()

    layers = _iter_gpt_oss_layers(model)
    num_layers = len(layers)
    num_experts = int(getattr(model.config, "num_local_experts", 0) or 0)
    top_k = int(getattr(model.config, "num_experts_per_tok", 0) or getattr(model.config, "experts_per_token", 4))
    if num_experts <= 0 or top_k <= 0:
        raise RuntimeError("Could not determine num_local_experts / top_k from config.")

    # We only compute p/H/weights for the same sampled token subset that REAP-lite
    # uses for expert-norm computation (first N kept tokens), for efficiency.
    max_tokens = int(os.environ.get("REAP_MAX_TOKENS_PER_BATCH", "128"))
    max_tokens = int(max_tokens) if int(max_tokens) > 0 else 0
    entropy_topk = int(entropy_topk)
    if entropy_topk <= 1:
        raise ValueError("entropy_topk must be >= 2")

    # ---- Pass 1: collect p_t and H_t samples (and store per-batch) ------------
    p_all: list[float] = []
    h_all: list[float] = []
    batch_samples: list[dict[str, Any]] = []

    total_tokens = 0
    total_kept_tokens = 0
    t0 = time.time()

    for batch_i, start in enumerate(range(0, len(rows), max(1, int(batch_size))), start=1):
        batch = rows[start : start + int(batch_size)]
        texts = [r["text"] for r in batch]

        tok = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=int(max_seq_length),
            return_offsets_mapping=True,
        )
        offsets = tok.pop("offset_mapping")  # CPU

        keep_masks: list[list[bool]] = []
        for row_i, text in enumerate(texts):
            spans = _assistant_content_spans(text)
            keep = _token_keep_mask(offsets[row_i].tolist(), spans)
            keep_masks.append(keep)

        enc = {k: v.to("cuda") for k, v in tok.items()}
        attn = enc.get("attention_mask")
        keep = torch.tensor(keep_masks, dtype=torch.bool, device="cuda")
        if attn is not None and torch.is_tensor(attn):
            keep = keep & attn.to(torch.bool)

        bs, seq = keep.shape
        keep_flat = keep.reshape(-1)
        sel_flat = torch.nonzero(keep_flat, as_tuple=False).squeeze(-1)
        if max_tokens and int(sel_flat.numel()) > max_tokens:
            sel_flat = sel_flat[:max_tokens]

        # Placeholder lists aligned to `sel_flat` order; pos==0 has no prediction.
        p_list: list[float | None] = [None] * int(sel_flat.numel())
        h_list: list[float | None] = [None] * int(sel_flat.numel())

        if int(sel_flat.numel()) > 0:
            with torch.inference_mode():
                out = model(**enc, use_cache=False, return_dict=True)
            logits = out.logits  # [bs, seq, vocab]

            sel_b = (sel_flat // int(seq)).to(torch.int64)
            sel_t = (sel_flat % int(seq)).to(torch.int64)
            pred_mask = sel_t > 0
            if bool(pred_mask.any().item()):
                b = sel_b[pred_mask]
                t = sel_t[pred_mask]
                logits_sel = logits[b, t - 1, :]  # predicts token at t
                tgt = enc["input_ids"][b, t].to(torch.int64)

                logits_sel_f = logits_sel.float()
                logit_ref = logits_sel_f.gather(1, tgt.unsqueeze(-1)).squeeze(-1)
                lse = torch.logsumexp(logits_sel_f, dim=-1)
                p = torch.exp((logit_ref - lse).clamp(min=-50.0, max=0.0)).clamp(min=0.0, max=1.0)

                topk_vals = torch.topk(logits_sel_f, k=int(entropy_topk), dim=-1).values
                topk_p = torch.softmax(topk_vals, dim=-1)
                ent = -(topk_p * torch.log(topk_p.clamp_min(1e-12))).sum(dim=-1) / math.log(float(entropy_topk))
                ent = ent.clamp(min=0.0, max=1.0)

                p_cpu = p.detach().to("cpu").tolist()
                h_cpu = ent.detach().to("cpu").tolist()
                pred_mask_cpu = pred_mask.detach().to("cpu").tolist()
                it = 0
                for i, ok in enumerate(pred_mask_cpu):
                    if not ok:
                        continue
                    pv = float(p_cpu[it])
                    hv = float(h_cpu[it])
                    p_list[i] = pv
                    h_list[i] = hv
                    p_all.append(pv)
                    h_all.append(hv)
                    it += 1

            # Free logits ASAP.
            del logits
            del out

        batch_samples.append({"n_sel": int(sel_flat.numel()), "p": p_list, "h": h_list})
        total_tokens += int(enc["input_ids"].numel())
        total_kept_tokens += int(keep.sum().item())

        if batch_i % 25 == 0:
            dt_i = max(1e-9, time.time() - t0)
            print(
                f"[*] eaftreap pass1 batches={batch_i} rows={min(len(rows), start+int(batch_size))}/{len(rows)} "
                f"tokens={total_tokens} kept={total_kept_tokens} tok/s={total_tokens/dt_i:.0f} p_samples={len(p_all)}",
                flush=True,
            )

    if not p_all or not h_all:
        raise RuntimeError("No EAFT samples collected; check that assistant spans exist in your data.")

    import numpy as np

    p_np = np.asarray(p_all, dtype=np.float64)
    h_np = np.asarray(h_all, dtype=np.float64)
    p_lo = float(np.quantile(p_np, float(cc_quantile)))
    h_lo = float(np.quantile(h_np, float(cc_quantile)))
    h_hi = float(np.quantile(h_np, float(uncertain_quantile)))

    # Assign weights (CPU lists aligned to per-batch selection order).
    weights_by_batch: list[list[float]] = []
    region_counts = {"good": 0, "conflict": 0, "uncertain": 0}
    for b in batch_samples:
        ws: list[float] = []
        for pv, hv in zip(b["p"], b["h"], strict=True):
            if pv is None or hv is None:
                ws.append(0.0)
                continue
            p_v = float(pv)
            h_v = float(hv)
            if p_v >= p_lo and h_v <= h_lo:
                ws.append(float(w_good))
                region_counts["good"] += 1
            elif p_v < p_lo and h_v <= h_lo:
                ws.append(float(w_conflict))
                region_counts["conflict"] += 1
            else:
                ws.append(float(w_uncertain))
                region_counts["uncertain"] += 1
        weights_by_batch.append(ws)

    # ---- Pass 2: run REAP-lite hooks with w_t weighting ------------------------
    current_keep_mask = {"mask": None}
    current_token_weights = {"weights": None}

    count = torch.zeros((num_layers, num_experts), dtype=torch.int64, device="cuda")
    gate_sum = torch.zeros((num_layers, num_experts), dtype=torch.float32, device="cuda")
    norm_sum = torch.zeros((num_layers, num_experts), dtype=torch.float32, device="cuda")
    gate_norm_sum = torch.zeros((num_layers, num_experts), dtype=torch.float32, device="cuda")
    gate_norm_sum_weighted = torch.zeros((num_layers, num_experts), dtype=torch.float32, device="cuda")
    pos_count = torch.zeros((num_layers, num_experts), dtype=torch.int64, device="cuda")
    neg_count = torch.zeros((num_layers, num_experts), dtype=torch.int64, device="cuda")
    pos_gate_norm_sum = torch.zeros((num_layers, num_experts), dtype=torch.float32, device="cuda")
    neg_gate_norm_sum = torch.zeros((num_layers, num_experts), dtype=torch.float32, device="cuda")

    hooks = []

    def _make_mlp_prehook(layer_idx: int):
        router = layers[layer_idx].mlp.router
        experts = layers[layer_idx].mlp.experts

        def _hook(_module, inputs):
            if not inputs:
                return
            hidden = inputs[0]
            if not torch.is_tensor(hidden) or hidden.dim() != 3:
                return

            keep_mask = current_keep_mask.get("mask")
            w_mask = current_token_weights.get("weights")
            if keep_mask is None or not torch.is_tensor(keep_mask):
                return
            if w_mask is None or not torch.is_tensor(w_mask):
                return
            if keep_mask.shape[:2] != hidden.shape[:2] or w_mask.shape[:2] != hidden.shape[:2]:
                return

            hs = hidden  # [bs, seq, hidden]
            bs, seq, hd = hs.shape
            hs2 = hs.reshape(-1, int(hd))
            keep2 = keep_mask.reshape(-1)
            if int(keep2.sum().item()) == 0:
                return
            hs_sel = hs2[keep2]
            w2 = w_mask.reshape(-1)[keep2].to(torch.float32)

            if max_tokens > 0 and int(hs_sel.shape[0]) > max_tokens:
                hs_sel = hs_sel[:max_tokens]
                w2 = w2[:max_tokens]

            out = router(hs_sel)
            if not isinstance(out, (tuple, list)) or len(out) != 2:
                return
            router_scores, router_idx = out
            if not torch.is_tensor(router_scores) or not torch.is_tensor(router_idx):
                return
            if router_scores.dim() != 2 or int(router_scores.shape[1]) != int(num_experts):
                return
            if router_idx.dim() != 2:
                return
            kk = int(router_idx.shape[1])
            if kk <= 0:
                return

            router_idx = router_idx.to(torch.int64)
            topk_w = router_scores.gather(1, router_idx).to(torch.float32)
            denom = topk_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            topk_w = topk_w / denom

            gate_up_proj = getattr(experts, "gate_up_proj", None)
            down_proj = getattr(experts, "down_proj", None)
            if gate_up_proj is None or down_proj is None:
                return
            gate_up_proj_bias = getattr(experts, "gate_up_proj_bias", None)
            down_proj_bias = getattr(experts, "down_proj_bias", None)
            if not torch.is_tensor(gate_up_proj) or not torch.is_tensor(down_proj):
                raise TypeError(
                    "EAFT-REAP requires PyTorch expert weights, but got non-torch tensors for "
                    f"gate_up_proj={type(gate_up_proj)} down_proj={type(down_proj)}."
                )
            if gate_up_proj_bias is not None and not torch.is_tensor(gate_up_proj_bias):
                raise TypeError("EAFT-REAP requires PyTorch expert bias tensors, but got non-torch.")
            if down_proj_bias is not None and not torch.is_tensor(down_proj_bias):
                raise TypeError("EAFT-REAP requires PyTorch expert bias tensors, but got non-torch.")

            # GPT-OSS gated-swish clamp params (from its implementation).
            alpha = float(getattr(experts, "alpha", 1.702))
            limit = float(getattr(experts, "limit", 7.0))

            # Build per-expert norm for the selected top-k experts.
            n_tokens = int(hs_sel.shape[0])
            expert_ids = router_idx.reshape(-1)
            token_ids = (
                torch.arange(n_tokens, device=expert_ids.device, dtype=torch.int64)
                .unsqueeze(1)
                .expand(n_tokens, kk)
                .reshape(-1)
            )
            perm = torch.argsort(expert_ids)
            expert_ids_sorted = expert_ids[perm]
            token_ids_sorted = token_ids[perm]
            w_sorted = topk_w.reshape(-1)[perm]
            n_sorted = torch.empty_like(w_sorted, dtype=torch.float32)

            unique_e, counts_e = torch.unique_consecutive(expert_ids_sorted, return_counts=True)
            offset = 0
            for e_t, c_t in zip(unique_e.tolist(), counts_e.tolist()):
                e = int(e_t)
                c = int(c_t)
                tok = token_ids_sorted[offset : offset + c]
                x = hs_sel.index_select(0, tok)

                W_gu = gate_up_proj[e]
                W_down = down_proj[e]
                b_gu = gate_up_proj_bias[e] if gate_up_proj_bias is not None else None
                b_down = down_proj_bias[e] if down_proj_bias is not None else None

                gate_up = torch.matmul(x, W_gu)
                if b_gu is not None:
                    gate_up = gate_up + b_gu

                gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                gate = gate.clamp(max=limit)
                up = up.clamp(min=-limit, max=limit)
                glu = gate * torch.sigmoid(gate * alpha)
                act = (up + 1) * glu

                out_e = torch.matmul(act, W_down)
                if b_down is not None:
                    out_e = out_e + b_down
                n_sorted[offset : offset + c] = torch.linalg.norm(out_e.float(), dim=-1)
                offset += c

            norms_flat = torch.empty_like(n_sorted)
            norms_flat[perm] = n_sorted
            norms_sel = norms_flat.reshape(n_tokens, kk)
            flat_e = router_idx.reshape(-1)
            flat_w = topk_w.reshape(-1)
            flat_n = norms_sel.reshape(-1)

            w_rep = w2.unsqueeze(1).expand(n_tokens, kk).reshape(-1).to(torch.float32)

            ones = torch.ones_like(flat_e, dtype=torch.int64)
            count[layer_idx].index_add_(0, flat_e, ones)
            gate_sum[layer_idx].index_add_(0, flat_e, flat_w)
            norm_sum[layer_idx].index_add_(0, flat_e, flat_n)

            gate_norm = flat_w * flat_n
            gate_norm_sum[layer_idx].index_add_(0, flat_e, gate_norm)
            gate_norm_sum_weighted[layer_idx].index_add_(0, flat_e, gate_norm * w_rep)

            pos_m = (w_rep > 0).to(torch.float32)
            neg_m = (w_rep < 0).to(torch.float32)
            pos_count[layer_idx].index_add_(0, flat_e, pos_m.to(torch.int64))
            neg_count[layer_idx].index_add_(0, flat_e, neg_m.to(torch.int64))
            pos_gate_norm_sum[layer_idx].index_add_(0, flat_e, gate_norm * pos_m)
            neg_gate_norm_sum[layer_idx].index_add_(0, flat_e, gate_norm * neg_m)

        return _hook

    for li in range(num_layers):
        hooks.append(layers[li].mlp.register_forward_pre_hook(_make_mlp_prehook(li)))

    t1 = time.time()
    total_tokens_pass2 = 0
    total_kept_tokens_pass2 = 0

    try:
        for batch_i, start in enumerate(range(0, len(rows), max(1, int(batch_size))), start=1):
            batch = rows[start : start + int(batch_size)]
            texts = [r["text"] for r in batch]

            tok = tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=int(max_seq_length),
                return_offsets_mapping=True,
            )
            offsets = tok.pop("offset_mapping")  # CPU

            keep_masks = []
            for row_i, text in enumerate(texts):
                spans = _assistant_content_spans(text)
                keep_masks.append(_token_keep_mask(offsets[row_i].tolist(), spans))

            enc = {k: v.to("cuda") for k, v in tok.items()}
            attn = enc.get("attention_mask")
            keep = torch.tensor(keep_masks, dtype=torch.bool, device="cuda")
            if attn is not None and torch.is_tensor(attn):
                keep = keep & attn.to(torch.bool)

            bs, seq = keep.shape
            keep_flat = keep.reshape(-1)
            sel_flat = torch.nonzero(keep_flat, as_tuple=False).squeeze(-1)
            if max_tokens and int(sel_flat.numel()) > max_tokens:
                sel_flat = sel_flat[:max_tokens]

            ws = weights_by_batch[batch_i - 1]
            if int(sel_flat.numel()) != len(ws):
                raise RuntimeError(
                    f"EAFT-REAP selection mismatch at batch {batch_i}: "
                    f"sel={int(sel_flat.numel())} stored={len(ws)}"
                )

            w_full = torch.zeros((bs, seq), dtype=torch.float32, device="cuda")
            if int(sel_flat.numel()) > 0:
                w_full.reshape(-1)[sel_flat] = torch.tensor(ws, dtype=torch.float32, device="cuda")

            current_keep_mask["mask"] = keep
            current_token_weights["weights"] = w_full
            with torch.inference_mode():
                _ = model(**enc, use_cache=False, return_dict=False)

            total_tokens_pass2 += int(enc["input_ids"].numel())
            total_kept_tokens_pass2 += int(keep.sum().item())
            if batch_i % 25 == 0:
                dt_i = max(1e-9, time.time() - t1)
                print(
                    f"[*] eaftreap pass2 batches={batch_i} rows={min(len(rows), start+int(batch_size))}/{len(rows)} "
                    f"tok/s={total_tokens_pass2/dt_i:.0f}",
                    flush=True,
                )
    finally:
        current_keep_mask["mask"] = None
        current_token_weights["weights"] = None
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

    dt = max(1e-9, time.time() - t1)
    toks_s = total_tokens_pass2 / dt

    # Move to CPU for serialization.
    count_cpu = count.detach().to("cpu")
    gate_sum_cpu = gate_sum.detach().to("cpu")
    norm_sum_cpu = norm_sum.detach().to("cpu")
    gate_norm_sum_cpu = gate_norm_sum.detach().to("cpu")
    gate_norm_sum_weighted_cpu = gate_norm_sum_weighted.detach().to("cpu")
    pos_count_cpu = pos_count.detach().to("cpu")
    neg_count_cpu = neg_count.detach().to("cpu")
    pos_gate_norm_sum_cpu = pos_gate_norm_sum.detach().to("cpu")
    neg_gate_norm_sum_cpu = neg_gate_norm_sum.detach().to("cpu")

    ranking: list[list[int]] = []
    for li in range(num_layers):
        # Rank by *total* EAFT-weighted saliency mass (not mean-per-hit).
        # For pruning, frequency matters: we prefer to keep experts that
        # contribute across many tokens rather than rare high-norm spikes.
        score = gate_norm_sum_weighted_cpu[li].to(torch.float32)
        order = torch.argsort(score, descending=True).tolist()
        ranking.append([int(x) for x in order])

    parquet_rows: list[dict[str, Any]] = []
    for li in range(num_layers):
        for e in range(num_experts):
            c = int(count_cpu[li, e].item())
            gsum = float(gate_sum_cpu[li, e].item())
            nsum = float(norm_sum_cpu[li, e].item())
            gnsum = float(gate_norm_sum_cpu[li, e].item())
            gnsum_w = float(gate_norm_sum_weighted_cpu[li, e].item())
            pc = int(pos_count_cpu[li, e].item())
            nc = int(neg_count_cpu[li, e].item())
            pg = float(pos_gate_norm_sum_cpu[li, e].item())
            ng = float(neg_gate_norm_sum_cpu[li, e].item())
            parquet_rows.append(
                {
                    "layer": int(li),
                    "expert": int(e),
                    "count": int(c),
                    "gate_sum": float(gsum),
                    "norm_sum": float(nsum),
                    "gate_norm_sum": float(gnsum),
                    "gate_mean": float(gsum / max(1.0, float(c))),
                    "norm_mean": float(nsum / max(1.0, float(c))),
                    "saliency_mean": float(gnsum / max(1.0, float(c))),
                    "eaft_gate_norm_sum": float(gnsum_w),
                    "eaft_saliency_mean": float(gnsum_w / max(1.0, float(c))),
                    "pos_count": int(pc),
                    "neg_count": int(nc),
                    "pos_gate_norm_sum": float(pg),
                    "neg_gate_norm_sum": float(ng),
                }
            )

    table = pa.Table.from_pylist(parquet_rows)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")

    return {
        "meta": {
            "model_id": str(model_id),
            "dataset_id": str(dataset_id),
            "dataset_split": str(dataset_split),
            "text_column": str(text_column),
            "domain": str(domain or ""),
            "prompt_hash": str(prompt_hash),
            "num_rows": int(num_rows),
            "max_seq_length": int(max_seq_length),
            "batch_size": int(batch_size),
            "num_layers": int(num_layers),
            "num_experts": int(num_experts),
            "top_k": int(top_k),
            "entropy_topk": int(entropy_topk),
            "cc_quantile": float(cc_quantile),
            "uncertain_quantile": float(uncertain_quantile),
            "p_lo": float(p_lo),
            "h_lo": float(h_lo),
            "h_hi": float(h_hi),
            "w_good": float(w_good),
            "w_uncertain": float(w_uncertain),
            "w_conflict": float(w_conflict),
            "region_counts": dict(region_counts),
            "samples": {"p": int(len(p_all)), "H": int(len(h_all))},
            "tokens_per_s_pass2": float(toks_s),
            "total_tokens_pass2": int(total_tokens_pass2),
            "total_kept_tokens_pass2": int(total_kept_tokens_pass2),
        },
        "ranking_by_layer": ranking,
        "parquet_bytes": sink.getvalue().to_pybytes(),
    }


@app.function(
    image=image,
    gpu="B200:1",
    timeout=21600,
    cpu=16.0,
    memory=262144,
    volumes={
        "/root/data": data_volume,
        "/root/model": model_volume,
        "/root/hf_cache": hf_cache_volume,
    },
    secrets=_secrets,
)
def soft_prune_20b_eval(
    *,
    model_id: str = DEFAULT_20B_MODEL_ID,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_split: str = DEFAULT_DATASET_SPLIT,
    text_column: str = DEFAULT_TEXT_COLUMN,
    expert_ranking_by_layer_json: str,
    keep_fracs_csv: str = "1.0,0.5,0.25",
    top_ks_csv: str = "4,2",
    eval_rows: int = 256,
    max_seq_length: int = 4096,
):
    import math

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _ensure_hf_env()

    expert_ranking_by_layer = json.loads(expert_ranking_by_layer_json)
    if not isinstance(expert_ranking_by_layer, list) or not expert_ranking_by_layer:
        raise ValueError("expert_ranking_by_layer_json must be a non-empty JSON list.")

    keep_fracs = []
    for part in (keep_fracs_csv or "").split(","):
        part = part.strip()
        if not part:
            continue
        keep_fracs.append(float(part))
    top_ks = []
    for part in (top_ks_csv or "").split(","):
        part = part.strip()
        if not part:
            continue
        top_ks.append(int(part))

    model_dir = _snapshot_download_model(model_id)
    texts = _stream_text_rows(
        dataset_id=dataset_id,
        split=dataset_split,
        text_column=text_column,
        limit=int(eval_rows),
    )

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype="auto",
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()

    layers = _iter_gpt_oss_layers(model)
    num_layers = len(layers)
    num_experts = int(getattr(model.config, "num_local_experts", 0) or 0)
    if num_experts <= 0:
        raise RuntimeError("Could not determine num_local_experts from config.")
    if len(expert_ranking_by_layer) != num_layers:
        raise ValueError(
            f"expert_ranking_by_layer has {len(expert_ranking_by_layer)} layers, expected {num_layers}"
        )

    orig_forward_by_router = {}
    for li, layer in enumerate(layers):
        router = getattr(getattr(layer, "mlp", None), "router", None)
        if router is None:
            raise RuntimeError(f"Layer {li} has no mlp.router")
        orig_forward_by_router[li] = router.forward

    def _get_router_bias(router):
        b = getattr(router, "bias", None)
        if torch.is_tensor(b):
            return b
        linear = getattr(router, "linear", None)
        if linear is not None:
            bb = getattr(linear, "bias", None)
            if torch.is_tensor(bb):
                return bb
        return None

    orig_bias_by_layer: dict[int, torch.Tensor] = {}
    orig_cfg_top_k = int(getattr(model.config, "num_experts_per_tok", 0) or getattr(model.config, "experts_per_token", 4) or 4)

    def eval_one(*, keep_frac: float, top_k: int) -> dict[str, Any]:
        keep_n = max(1, min(num_experts, int(math.ceil(float(keep_frac) * num_experts))))
        allowed_by_layer = [
            [int(e) for e in expert_ranking_by_layer[li][:keep_n]] for li in range(num_layers)
        ]

        # Patch routing policy:
        # - keep_frac: mask disallowed experts via router bias (forces router away from them).
        # - top_k: reduce experts_per_token (if supported by this model).
        eff_top_k = int(max(1, min(int(top_k), int(keep_n))))
        for li, layer in enumerate(layers):
            router = layer.mlp.router
            router.forward = orig_forward_by_router[li]

            try:
                router.top_k = int(eff_top_k)
            except Exception:
                pass
            try:
                router.num_experts_per_tok = int(eff_top_k)
            except Exception:
                pass
            try:
                model.config.num_experts_per_tok = int(eff_top_k)
                model.config.experts_per_token = int(eff_top_k)
            except Exception:
                pass

            bias = _get_router_bias(router)
            if bias is not None:
                if li not in orig_bias_by_layer:
                    orig_bias_by_layer[li] = bias.detach().clone()
                with torch.no_grad():
                    bias.copy_(orig_bias_by_layer[li])
                    if keep_n < num_experts:
                        allowed = torch.zeros((num_experts,), dtype=torch.bool, device=bias.device)
                        for e in allowed_by_layer[li]:
                            if 0 <= int(e) < num_experts:
                                allowed[int(e)] = True
                        # Large negative shift to push disallowed experts out of top-k.
                        bias[~allowed] = bias[~allowed] - torch.tensor(
                            1e9, device=bias.device, dtype=bias.dtype
                        )

        total_loss = 0.0
        total_tokens = 0
        t0 = time.time()
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=int(max_seq_length),
                padding=False,
            )
            input_ids = enc["input_ids"]
            attn = torch.ones_like(input_ids, dtype=torch.long)
            labels = input_ids.clone()
            enc = {
                "input_ids": input_ids.to("cuda"),
                "attention_mask": attn.to("cuda"),
                "labels": labels.to("cuda"),
            }
            with torch.inference_mode():
                out = model(**enc, use_cache=False)
                loss = float(out.loss.detach().float().item())
            n = int(enc["labels"].numel())
            total_loss += loss * n
            total_tokens += n

        torch.cuda.synchronize()
        dt = max(1e-9, time.time() - t0)
        mean_loss = total_loss / max(1, total_tokens)
        ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")

        return {
            "keep_frac": float(keep_frac),
            "keep_n": int(keep_n),
            "top_k": int(eff_top_k),
            "eval_rows": int(eval_rows),
            "max_seq_length": int(max_seq_length),
            "tokens": int(total_tokens),
            "wall_s": float(dt),
            "tokens_per_s": float(total_tokens / dt),
            "mean_loss": float(mean_loss),
            "ppl": float(ppl),
        }

    results: list[dict[str, Any]] = []
    try:
        for keep_frac in keep_fracs:
            for top_k in top_ks:
                results.append(eval_one(keep_frac=float(keep_frac), top_k=int(top_k)))
    finally:
        # Restore original forwards + router biases + config.
        for li, layer in enumerate(layers):
            layer.mlp.router.forward = orig_forward_by_router[li]
            bias = _get_router_bias(layer.mlp.router)
            if bias is not None and li in orig_bias_by_layer:
                with torch.no_grad():
                    bias.copy_(orig_bias_by_layer[li])
        try:
            model.config.num_experts_per_tok = int(orig_cfg_top_k)
            model.config.experts_per_token = int(orig_cfg_top_k)
        except Exception:
            pass

    return {
        "meta": {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "dataset_split": dataset_split,
            "text_column": text_column,
            "num_layers": int(num_layers),
            "num_experts": int(num_experts),
        },
        "results": results,
    }


@app.function(
    image=image,
    timeout=21600,
    cpu=32.0,
    memory=262144,
    volumes={
        "/root/data": data_volume,
        "/root/model": model_volume,
        "/root/hf_cache": hf_cache_volume,
    },
    secrets=_secrets,
)
def structural_prune_20b_build(
    *,
    model_id: str,
    variant_name: str,
    keep_experts_by_layer_json: str,
    out_subdir: str = "20b_pruned_models",
    max_shard_size_gb: float = 5.0,
):
    from safetensors.torch import safe_open
    from safetensors.torch import save_file
    from huggingface_hub import hf_hub_download

    _ensure_hf_env()

    keep_experts_by_layer = json.loads(keep_experts_by_layer_json)
    if not isinstance(keep_experts_by_layer, list) or not keep_experts_by_layer:
        raise ValueError("keep_experts_by_layer_json must be a non-empty JSON list.")

    t0 = time.time()

    snapshot_dir = _snapshot_download_model(model_id)
    # NOTE: We previously symlinked snapshot files into the pruned output dir and
    # then overwrote `config.json` / `model.safetensors.index.json`, which can
    # mutate the underlying HF snapshot via symlink-following writes.
    #
    # If we are using an on-disk model override (Kaggle inputs, local mirror),
    # do NOT call the Hub at all.
    using_local_override = False
    try:
        if DEFAULT_20B_MODEL_DIR and str(model_id) == str(DEFAULT_20B_MODEL_ID):
            using_local_override = Path(DEFAULT_20B_MODEL_DIR).resolve() == snapshot_dir.resolve()
        if Path(str(model_id)).exists():
            using_local_override = True
    except Exception:
        using_local_override = False

    if not using_local_override:
        token = _get_hf_token()
        cache_dir = _MODEL_DIR / ".hf_cache"
        for fname in ("config.json", "model.safetensors.index.json"):
            hf_hub_download(
                repo_id=str(model_id),
                repo_type="model",
                filename=fname,
                cache_dir=str(cache_dir),
                token=token,
                force_download=True,
            )
    idx_path = snapshot_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        raise RuntimeError(f"Missing safetensors index at {idx_path}")
    cfg_path = snapshot_dir / "config.json"
    if not cfg_path.exists():
        raise RuntimeError(f"Missing config.json at {cfg_path}")

    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    weight_map = idx.get("weight_map") or {}
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("Invalid model.safetensors.index.json: missing weight_map")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    num_layers = int(cfg.get("num_hidden_layers") or 0)
    num_experts = int(cfg.get("num_local_experts") or 0)
    if num_layers <= 0 or num_experts <= 0:
        raise RuntimeError("Could not determine num_hidden_layers/num_local_experts from config.")
    if len(keep_experts_by_layer) != num_layers:
        raise ValueError(
            f"keep_experts_by_layer has {len(keep_experts_by_layer)} layers, expected {num_layers}"
        )

    keep_n = len(keep_experts_by_layer[0])
    if keep_n <= 0 or keep_n > num_experts:
        raise ValueError(f"Invalid keep_n={keep_n} for num_experts={num_experts}")
    for li in range(num_layers):
        if len(keep_experts_by_layer[li]) != keep_n:
            raise ValueError("All layers must keep the same number of experts.")

    out_dir = _ARTIFACTS_DIR / str(out_subdir) / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove previously-generated shards if re-running in the same volume path.
    for p in out_dir.glob("*.safetensors"):
        if p.name.startswith(("base_", "base_shard_", "pruned_layer_", "pruned_shard_", "model-")):
            try:
                p.unlink()
            except Exception:
                pass

    # Symlink non-weight snapshot files into output dir (cheap overlay).
    # Skip files we will rewrite, and skip original `.safetensors` shards since
    # we will emit base shards that exclude the pruned expert/router keys.
    skip_names = {"config.json", "model.safetensors.index.json"}
    for src in snapshot_dir.iterdir():
        if not src.is_file():
            continue
        if src.name in skip_names:
            continue
        if src.suffix == ".safetensors":
            continue
        dst = out_dir / src.name
        if dst.exists():
            continue
        try:
            dst.symlink_to(src)
        except Exception:
            pass

    # Update config for new expert count.
    cfg["num_local_experts"] = int(keep_n)
    cfg["num_experts_per_tok"] = int(min(int(cfg.get("num_experts_per_tok", 4)), keep_n))
    cfg["experts_per_token"] = int(cfg["num_experts_per_tok"])
    cfg_out = out_dir / "config.json"
    if cfg_out.exists() and cfg_out.is_symlink():
        cfg_out.unlink()
    cfg_out.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")

    # Remap expert tensors per layer into per-layer safetensors shards.
    new_weight_map = dict(weight_map)
    mapping_json: dict[str, dict[str, int]] = {}

    def keys_for_layer(li: int) -> list[str]:
        prefix = f"model.layers.{li}.mlp."
        candidates = []
        for k in weight_map.keys():
            if not k.startswith(prefix):
                continue
            if ".router." in k:
                candidates.append(k)
                continue
            if ".experts." in k:
                candidates.append(k)
                continue
        return sorted(candidates)

    # Collect all pruned keys up front so we can build "base shards" that
    # exclude them. This avoids Transformers loading a referenced shard and
    # encountering extra (now shape-mismatched) tensors.
    pruned_keys: set[str] = set()
    pruned_keys_by_layer: dict[int, list[str]] = {}
    for li in range(num_layers):
        ks = keys_for_layer(li)
        pruned_keys_by_layer[li] = ks
        pruned_keys.update(ks)

    print(
        f"[*] structural_prune start variant={variant_name} keep_n={keep_n}/{num_experts} "
        f"layers={num_layers} pruned_keys={len(pruned_keys)}",
        flush=True,
    )

    # Interpret "GB" as decimal (1e9 bytes) so a default of 5.0 produces shards
    # that are <= ~5GB, matching downstream filesystem constraints better than
    # GiB sizing.
    max_bytes = int(float(max_shard_size_gb) * 1_000_000_000.0)
    if max_bytes <= 0:
        raise ValueError("max_shard_size_gb must be > 0")

    def _tensor_nbytes(t) -> int:
        try:
            return int(t.numel()) * int(t.element_size())
        except Exception:
            return 0

    def _flush_shard(*, name: str, tensors: dict[str, Any], keys: list[str], meta_tag: str) -> None:
        if not tensors:
            return
        save_file(tensors, str(out_dir / name), metadata={"format": "pt", "harmony": meta_tag})
        for k in keys:
            new_weight_map[k] = name
        print(f"[*] wrote {name} keys={len(keys)}", flush=True)

    # Build base shards (packed up to ~max_shard_size_gb) containing only the
    # remaining (non-pruned) keys. This reduces shard count for faster IO.
    remaining_by_file: dict[str, list[str]] = {}
    for k, src_file in weight_map.items():
        if k in pruned_keys:
            continue
        remaining_by_file.setdefault(str(src_file), []).append(str(k))

    print(
        f"[*] base shards: {len(remaining_by_file)} files, "
        f"remaining_keys={sum(len(v) for v in remaining_by_file.values())}",
        flush=True,
    )

    base_shard_i = 0
    base_tensors: dict[str, Any] = {}
    base_keys: list[str] = []
    base_bytes = 0

    # Deterministic order: by source shard filename then key.
    for src_file in sorted(remaining_by_file.keys()):
        keys = sorted(remaining_by_file[src_file])
        src_path = snapshot_dir / src_file
        if not src_path.exists():
            raise RuntimeError(f"Missing shard file {src_path}")
        with safe_open(str(src_path), framework="pt", device="cpu") as f:
            for key in keys:
                t = f.get_tensor(key)
                nb = _tensor_nbytes(t)
                if base_tensors and (base_bytes + nb) > max_bytes:
                    shard_name = f"base_shard_{base_shard_i:05d}.safetensors"
                    _flush_shard(name=shard_name, tensors=base_tensors, keys=base_keys, meta_tag="base")
                    base_shard_i += 1
                    base_tensors = {}
                    base_keys = []
                    base_bytes = 0
                base_tensors[key] = t
                base_keys.append(key)
                base_bytes += nb

    if base_tensors:
        shard_name = f"base_shard_{base_shard_i:05d}.safetensors"
        _flush_shard(name=shard_name, tensors=base_tensors, keys=base_keys, meta_tag="base")

    # Pack pruned expert/router keys into shards up to ~max_shard_size_gb.
    pruned_shard_i = 0
    pruned_tensors: dict[str, Any] = {}
    pruned_keys_accum: list[str] = []
    pruned_bytes = 0

    for li in range(num_layers):
        keep = [int(x) for x in keep_experts_by_layer[li]]
        mapping_json[str(li)] = {str(old): int(new) for new, old in enumerate(keep)}

        for key in pruned_keys_by_layer[li]:
            src_file = weight_map.get(key)
            if not src_file:
                continue
            src_path = snapshot_dir / src_file
            if not src_path.exists():
                raise RuntimeError(f"Missing shard file {src_path} for key {key}")
            with safe_open(str(src_path), framework="pt", device="cpu") as f:
                t = f.get_tensor(key)
            if not hasattr(t, "shape") or not t.shape:
                raise RuntimeError(f"Unexpected tensor for key {key}: shape={getattr(t,'shape',None)}")
            if int(t.shape[0]) != int(num_experts):
                raise RuntimeError(
                    f"Unexpected leading dim for {key}: shape={tuple(t.shape)} expected dim0={num_experts}"
                )
            t2 = t[keep].contiguous()
            nb = _tensor_nbytes(t2)
            if pruned_tensors and (pruned_bytes + nb) > max_bytes:
                shard_name = f"pruned_shard_{pruned_shard_i:05d}.safetensors"
                _flush_shard(name=shard_name, tensors=pruned_tensors, keys=pruned_keys_accum, meta_tag="pruned")
                pruned_shard_i += 1
                pruned_tensors = {}
                pruned_keys_accum = []
                pruned_bytes = 0
            pruned_tensors[key] = t2
            pruned_keys_accum.append(key)
            pruned_bytes += nb

    if pruned_tensors:
        shard_name = f"pruned_shard_{pruned_shard_i:05d}.safetensors"
        _flush_shard(name=shard_name, tensors=pruned_tensors, keys=pruned_keys_accum, meta_tag="pruned")

    idx["weight_map"] = new_weight_map
    idx_out = out_dir / "model.safetensors.index.json"
    if idx_out.exists() and idx_out.is_symlink():
        idx_out.unlink()
    idx_out.write_text(
        json.dumps(idx, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "expert_mapping.json").write_text(
        json.dumps(mapping_json, indent=2, sort_keys=True), encoding="utf-8"
    )

    model_volume.commit()
    hf_cache_volume.commit()
    print(f"[+] structural_prune done variant={variant_name} dt_s={time.time()-t0:.1f}", flush=True)
    return str(out_dir)


@app.function(
    image=image,
    gpu="B200:1",
    timeout=21600,
    cpu=16.0,
    memory=262144,
    volumes={
        "/root/data": data_volume,
        "/root/model": model_volume,
        "/root/hf_cache": hf_cache_volume,
    },
    secrets=_secrets,
)
def sanity_infer_model_dir(model_dir: str, prompt: str = "Explain MoE routing briefly."):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _ensure_hf_env()
    try:
        model_volume.reload()
        hf_cache_volume.reload()
    except Exception:
        pass
    model_path = Path(model_dir)
    if not model_path.exists():
        raise RuntimeError(f"model_dir not found: {model_dir}")

    # Debug: verify the pruned shard contains the expected expert dimension.
    try:
        from safetensors.torch import safe_open

        idx = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
        cfg = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        key0 = "model.layers.0.mlp.experts.down_proj_bias"
        mapped0 = (idx.get("weight_map") or {}).get(key0)
        if mapped0:
            with safe_open(str(model_path / mapped0), framework="pt", device="cpu") as f:
                t0 = f.get_tensor(key0)
            print(
                f"[*] sanity precheck num_local_experts={cfg.get('num_local_experts')} "
                f"{key0} file={mapped0} shape={tuple(t0.shape)}",
                flush=True,
            )
    except Exception as e:
        print(f"[!] sanity precheck failed: {type(e).__name__}: {e}", flush=True)

    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype="auto",
            device_map={"": 0},
            trust_remote_code=True,
        )
    except Exception as e:
        # Transformers loads *all* tensors present in each referenced shard file
        # (not only the ones mapped in weight_map). If a referenced shard still
        # contains 32-expert tensors, we can get a size-mismatch even if
        # weight_map points to pruned_layer_*.safetensors.
        try:
            from safetensors.torch import safe_open

            idx = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
            wm = idx.get("weight_map") or {}
            referenced = sorted(set(wm.values()))
            key = "model.layers.0.mlp.experts.down_proj_bias"
            hits = []
            for fname in referenced:
                p = model_path / fname
                if not p.exists():
                    continue
                with safe_open(str(p), framework="pt", device="cpu") as f:
                    if key in f.keys():
                        t = f.get_tensor(key)
                        hits.append((fname, tuple(t.shape)))
            print(f"[!] sanity load failed: {type(e).__name__}: {e}", flush=True)
            print(f"[!] referenced_shards={len(referenced)} key_hits={hits}", flush=True)
        except Exception as e2:
            print(f"[!] sanity debug failed: {type(e2).__name__}: {e2}", flush=True)
        raise
    model.eval()

    inputs = tok(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0], skip_special_tokens=False)
    return {"ok": True, "generated": text[:2000]}


@app.function(
    image=image,
    timeout=21600,
    cpu=8.0,
    memory=65536,
    volumes={
        "/root/model": model_volume,
        "/root/hf_cache": hf_cache_volume,
    },
    secrets=_secrets,
)
def inspect_pruned_checkpoint(model_dir: str, layer_idx: int = 0):
    from safetensors.torch import safe_open

    try:
        model_volume.reload()
        hf_cache_volume.reload()
    except Exception:
        pass
    model_path = Path(model_dir)
    if not model_path.exists():
        raise RuntimeError(f"model_dir not found: {model_dir}")
    idx_path = model_path / "model.safetensors.index.json"
    cfg_path = model_path / "config.json"
    if not idx_path.exists() or not cfg_path.exists():
        raise RuntimeError("Missing config/index in model_dir")

    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    wm = idx.get("weight_map") or {}
    layer = int(layer_idx)
    key = f"model.layers.{layer}.mlp.experts.down_proj_bias"
    src = wm.get(key)
    if not src:
        raise RuntimeError(f"Missing key in weight_map: {key}")
    src_path = model_path / src
    if not src_path.exists():
        raise RuntimeError(f"Mapped file missing: {src_path}")
    with safe_open(str(src_path), framework="pt", device="cpu") as f:
        t = f.get_tensor(key)
    return {
        "model_dir": str(model_path),
        "config_num_local_experts": int(cfg.get("num_local_experts") or 0),
        "key": key,
        "mapped_file": str(src),
        "tensor_shape": list(t.shape),
        "tensor_dtype": str(t.dtype),
    }


@app.function(
    image=image,
    timeout=21600,
    cpu=8.0,
    memory=65536,
    volumes={
        "/root/model": model_volume,
        "/root/hf_cache": hf_cache_volume,
    },
    secrets=_secrets,
)
def validate_pruned_expert_shards(model_dir: str):
    try:
        model_volume.reload()
        hf_cache_volume.reload()
    except Exception:
        pass

    model_path = Path(model_dir)
    idx_path = model_path / "model.safetensors.index.json"
    cfg_path = model_path / "config.json"
    if not idx_path.exists() or not cfg_path.exists():
        raise RuntimeError("Missing config/index in model_dir")

    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    wm = idx.get("weight_map") or {}

    num_layers = int(cfg.get("num_hidden_layers") or 0)
    mismatches = []
    ok = 0
    checked = 0
    for li in range(num_layers):
        prefix = f"model.layers.{li}.mlp."
        for k, v in wm.items():
            if not k.startswith(prefix):
                continue
            if ".router." not in k and ".experts." not in k:
                continue
            checked += 1
            # New packed format: pruned keys live in pruned_shard_*.safetensors.
            # Legacy format: pruned keys lived in pruned_layer_{li}.safetensors.
            is_ok = False
            if isinstance(v, str):
                if v.startswith("pruned_shard_") and v.endswith(".safetensors"):
                    is_ok = True
                if v == f"pruned_layer_{li}.safetensors":
                    is_ok = True
            if is_ok:
                ok += 1
                continue
            mismatches.append({"layer": li, "key": k, "file": v})
            if len(mismatches) >= 50:
                break
        if len(mismatches) >= 50:
            break

    return {
        "model_dir": str(model_path),
        "config_num_local_experts": int(cfg.get("num_local_experts") or 0),
        "config_num_layers": int(num_layers),
        "checked": int(checked),
        "ok": int(ok),
        "mismatch_count_capped": len(mismatches),
        "mismatches_sample": mismatches,
    }


@app.function(
    image=image,
    timeout=21600,
    cpu=8.0,
    memory=65536,
    volumes={
        "/root/model": model_volume,
        "/root/hf_cache": hf_cache_volume,
    },
    secrets=_secrets,
)
def list_artifact_files(model_dir: str, max_files: int = 200):
    try:
        model_volume.reload()
        hf_cache_volume.reload()
    except Exception:
        pass
    model_path = Path(model_dir)
    if not model_path.exists():
        raise RuntimeError(f"model_dir not found: {model_dir}")
    rows = []
    for p in sorted(model_path.iterdir(), key=lambda x: x.name):
        if not p.is_file():
            continue
        try:
            size = int(p.stat().st_size)
        except Exception:
            size = -1
        rows.append({"name": p.name, "bytes": size})
        if len(rows) >= int(max_files):
            break
    return {"model_dir": str(model_path), "files": rows, "capped": len(rows) >= int(max_files)}


def _manual_gpt_oss_experts_forward(
    *,
    hidden_states: "torch.Tensor",
    router_indices: "torch.Tensor",
    routing_weights: "torch.Tensor",
    gate_up_proj: "torch.Tensor",
    gate_up_proj_bias: "torch.Tensor",
    down_proj: "torch.Tensor",
    down_proj_bias: "torch.Tensor",
    alpha: float = 1.702,
    limit: float = 7.0,
) -> "torch.Tensor":
    import torch

    if hidden_states.dim() != 3:
        raise ValueError(f"hidden_states must be [bs, seq, hidden], got {tuple(hidden_states.shape)}")
    bs, seq, hidden = hidden_states.shape
    flat = hidden_states.reshape(-1, hidden)
    num_tokens = int(flat.shape[0])
    num_experts = int(routing_weights.shape[1])
    if router_indices.shape[0] != num_tokens:
        raise ValueError("router_indices must be flattened [num_tokens, top_k]")
    if routing_weights.shape[0] != num_tokens:
        raise ValueError("routing_weights must be flattened [num_tokens, num_experts]")

    next_states = torch.zeros_like(flat)
    expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=num_experts + 1).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx_t in expert_hit[:]:
        expert_idx = int(expert_idx_t[0])
        if expert_idx == num_experts:
            continue
        _, token_idx = torch.where(expert_mask[expert_idx])
        current_state = flat[token_idx]
        gate_up = current_state @ gate_up_proj[expert_idx] + gate_up_proj_bias[expert_idx]
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(max=float(limit))
        up = up.clamp(min=-float(limit), max=float(limit))
        glu = gate * torch.sigmoid(gate * float(alpha))
        gated_output = (up + 1) * glu
        out = gated_output @ down_proj[expert_idx] + down_proj_bias[expert_idx]
        weighted_output = out * routing_weights[token_idx, expert_idx, None]
        next_states.index_add_(0, token_idx, weighted_output.to(flat.dtype))
    return next_states.view(bs, seq, hidden)


def _routing_weights_from_topk(
    *, router_indices: "torch.Tensor", topk_weights: "torch.Tensor", num_experts: int
) -> "torch.Tensor":
    import torch

    if router_indices.dim() != 2 or topk_weights.dim() != 2:
        raise ValueError("router_indices/topk_weights must be 2D [num_tokens, top_k]")
    if router_indices.shape != topk_weights.shape:
        raise ValueError("router_indices and topk_weights must have same shape")
    if int(num_experts) <= 0:
        raise ValueError("num_experts must be > 0")
    num_tokens = int(router_indices.shape[0])
    routing_weights = torch.zeros((num_tokens, int(num_experts)), device=topk_weights.device, dtype=topk_weights.dtype)
    routing_weights.scatter_add_(1, router_indices.to(torch.int64), topk_weights)
    return routing_weights


def _validate_gpt_oss_expert_math_toy(
    *,
    device: str,
    trials: int = 10,
    seed: int = 3407,
    hidden_size: int = 256,
    intermediate_size: int = 512,
    num_experts: int = 8,
    top_k: int = 2,
    batch_size: int = 2,
    seq_len: int = 4,
) -> dict[str, float]:
    import math
    from types import SimpleNamespace

    import torch
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

    torch.manual_seed(int(seed))
    device_t = torch.device(device)
    cfg = SimpleNamespace(
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        num_local_experts=int(num_experts),
    )
    experts = GptOssExperts(cfg).to(device_t)
    experts.eval()

    with torch.no_grad():
        for p in experts.parameters():
            p.normal_(mean=0.0, std=0.02)

    max_abs = 0.0
    max_rel = 0.0
    for i in range(int(trials)):
        hs = torch.randn((int(batch_size), int(seq_len), int(hidden_size)), device=device_t, dtype=torch.float32)
        num_tokens = int(batch_size * seq_len)
        router_indices = torch.randint(
            low=0, high=int(num_experts), size=(num_tokens, int(top_k)), device=device_t, dtype=torch.int64
        )
        w = torch.rand((num_tokens, int(top_k)), device=device_t, dtype=torch.float32)
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        routing_weights = _routing_weights_from_topk(router_indices=router_indices, topk_weights=w, num_experts=int(num_experts))

        with torch.no_grad():
            out_ref = experts(hs, router_indices=router_indices, routing_weights=routing_weights)
            out_manual = _manual_gpt_oss_experts_forward(
                hidden_states=hs,
                router_indices=router_indices,
                routing_weights=routing_weights,
                gate_up_proj=experts.gate_up_proj,
                gate_up_proj_bias=experts.gate_up_proj_bias,
                down_proj=experts.down_proj,
                down_proj_bias=experts.down_proj_bias,
                alpha=float(experts.alpha),
                limit=float(experts.limit),
            )
        diff = (out_ref - out_manual).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        denom = out_ref.abs().clamp_min(1e-8)
        max_rel = max(max_rel, float((diff / denom).max().item()))
        if math.isnan(max_abs) or math.isnan(max_rel):
            raise RuntimeError("NaN encountered in expert-math validation.")
    return {"max_abs": float(max_abs), "max_rel": float(max_rel)}


@app.local_entrypoint()
def main(
    task: str = "profile_20b",
    model_id_20b: str = DEFAULT_20B_MODEL_ID,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_split: str = DEFAULT_DATASET_SPLIT,
    text_column: str = DEFAULT_TEXT_COLUMN,
    domain: str = DEFAULT_DOMAIN,
    domain_column: str = DEFAULT_DOMAIN_COLUMN,
    math_dataset_id: str = DEFAULT_MATH_DATASET_ID,
    math_dataset_split: str = DEFAULT_MATH_DATASET_SPLIT,
    math_text_column: str = DEFAULT_MATH_TEXT_COLUMN,
    num_rows: int = 500,
    max_seq_length: int = 4096,
    batch_size: int = 1,
    domains_csv: str = "math,science,agentic,general",
    seed: int = 3407,
    max_scan_rows: int = 2_000_000,
    eaft_cc_quantile: float = 0.15,
    eaft_uncertain_quantile: float = 0.85,
    eaft_entropy_topk: int = 20,
    eaft_w_good: float = 1.0,
    eaft_w_uncertain: float = 0.25,
    eaft_w_conflict: float = -2.0,
    calib_packs_repo: str = DEFAULT_CALIB_PACKS_REPO,
    calib_pack_files_csv: str = ",".join(DEFAULT_CALIB_PACK_FILES),
    calib_pack_sample_strategy: str = "per_file",
    calib_pack_weights_csv: str = "",
    keep_fracs_csv: str = "0.75",
    keep_n_round: str = "ceil",
    keep_n_multiple_of: int = 4,
    keep_frac: float = 0.75,
    min_keep_per_layer: int = 16,
    max_keep_per_layer: int = 32,
    core_pos_top_m: int = 4,
    core_count_top_m: int = 0,
):
    """
    Pruning-track tasks:
    - profile_20b: produce `reports/20b_expert_usage_profile.md` + `data/20b_expert_usage.parquet`
    - reap_saliency_20b: produce `reports/20b_reap_saliency_profile.md` + `data/20b_reap_saliency.parquet`
    - eaftreap_saliency_20b: produce `reports/20b_eaftreap_saliency_profile.md` + `data/20b_eaftreap_saliency.parquet`
    - reap_saliency_by_domain_20b: produce `artifacts/reap_saliency_by_domain/*.parquet` + `reports/reap_saliency_by_domain.md`
    - soft_prune_20b: produce `reports/20b_soft_prune_eval.md`
    - build_pruned_20b: produce `reports/20b_structural_prune_build.md` (and pruned checkpoints in Modal volume)
    - build_pruned_20b_freq: produce `artifacts/20b_pruned_models_freq/manifest_freq.json` (frequency-based)
    - build_pruned_20b_reap: produce `artifacts/20b_pruned_models_reap/manifest_reap.json` (REAP-ranked)
    - build_pruned_20b_eaftreap: produce `artifacts/20b_pruned_models_eaftreap/manifest_eaftreap.json` (EAFT-REAP ranked)
    - build_pruned_20b_eaftreap_keepfrac: build EAFT-REAP structural prunes on calib packs at keep_fracs (fixed top_k)
    - build_pruned_20b_eaftreap_budgeted: build a keep_frac=0.75 prune with per-layer keep_n allocation (fixed top_k)
    - build_pruned_20b_noop_rewrite: rewrite model shards while keeping all experts (sanity: must be lossless)
    - scan_domain_values_20b: scan `DOMAIN_COLUMN` and report counts (debug data availability)
    - validate_expert_math_toy: validate our manual GPT-OSS expert math matches Transformers (toy config)
    - predownload_20b: CPU-only download base 20B model into volumes (new Modal profile safe)
    - predownload_calib_packs: CPU-only download calib pack parquet files into HF cache volume
    - sample_calib_packs: CPU-only deterministic JSONL sample from calib packs into data volume
    - inspect_eaftreap_budgeted: validate + list files for latest budgeted prune (debug artifacts/shards)
    """

    # Ensure local outputs (reports/, artifacts/, data/) land under the project
    # root (harmony/cuda-norm) instead of the caller's CWD.
    #
    # - Kaggle/VERSA: the synced checkout lives at `/kaggle/working/harmony/cuda-norm`.
    # - Modal workers: this file is copied to `/root/<script>.py` (no repo checkout).
    #
    # Only chdir when we can prove the repo layout exists next to this file.
    try:
        repo_root = Path(__file__).resolve().parents[1]
        if (repo_root / "modal" / "gpt_oss_pruning_track.py").exists():
            os.chdir(repo_root)
    except Exception:
        pass

    if task == "predownload_20b":
        out_report = Path("reports/predownload_20b.md")
        out_report.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        model_dir = _invoke(predownload_model, str(model_id_20b))
        out_report.write_text(
            "\n".join(
                [
                    "# Predownload 20B (CPU)",
                    "",
                    f"- Model: `{model_id_20b}`",
                    f"- Local dir: `{model_dir}`",
                    f"- dt_s: {time.time() - t0:.1f}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"[+] Wrote {out_report}")
        return

    if task == "predownload_calib_packs":
        pack_files = _parse_csv(str(calib_pack_files_csv or "")) or list(DEFAULT_CALIB_PACK_FILES)
        if not pack_files:
            raise SystemExit("No calib pack files specified.")

        out_report = Path("reports/predownload_calib_packs.md")
        out_report.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        res = _invoke(
            predownload_calib_packs_cpu,
            dataset_repo=str(calib_packs_repo),
            pack_files=list(pack_files),
        )
        paths = list(res.get("downloaded_paths") or [])

        out_report.write_text(
            "\n".join(
                [
                    "# Predownload calib packs (CPU)",
                    "",
                    f"- Dataset repo: `{calib_packs_repo}`",
                    f"- Files: {', '.join(pack_files)}",
                    f"- Downloaded: {len(paths)}",
                    f"- dt_s: {time.time() - t0:.1f}",
                    "",
                ]
                + [f"- `{p}`" for p in paths[:50]]
            ),
            encoding="utf-8",
        )
        print(f"[+] Wrote {out_report}")
        return

    if task == "sample_calib_packs":
        pack_files = _parse_csv(str(calib_pack_files_csv or "")) or list(DEFAULT_CALIB_PACK_FILES)
        if not pack_files:
            raise SystemExit("No calib pack files specified.")
        pack_weights: list[float] | None = None
        if str(calib_pack_weights_csv or "").strip():
            pack_weights = []
            for s in _parse_csv(str(calib_pack_weights_csv)):
                try:
                    pack_weights.append(float(s))
                except Exception:
                    pack_weights.append(0.0)
        sample = _invoke(
            sample_calib_packs_rows_cpu,
            dataset_repo=str(calib_packs_repo),
            pack_files=list(pack_files),
            text_column=str(text_column),
            num_rows=int(num_rows),
            seed=int(seed),
            strategy=str(calib_pack_sample_strategy),
            pack_weights=pack_weights,
        )
        out_report = Path("reports/sample_calib_packs.md")
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(
            "\n".join(
                [
                    "# Sample calib packs (CPU)",
                    "",
                    f"- Dataset repo: `{calib_packs_repo}`",
                    f"- Files: {', '.join(pack_files)}",
                    f"- num_rows: {int(num_rows)} seed={int(seed)} strategy=`{calib_pack_sample_strategy}`",
                    f"- rows_jsonl_path: `{sample.get('rows_jsonl_path')}`",
                    f"- prompt_hash: `{sample.get('prompt_hash')}`",
                    f"- scanned: {sample.get('scanned')}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"[+] Wrote {out_report}")
        return

    if task == "inspect_eaftreap_budgeted":
        manifest_path = Path("artifacts/20b_pruned_models_eaftreap_budgeted/manifest.json")
        if not manifest_path.exists():
            raise SystemExit(f"Missing manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        out_dir = str(manifest.get("out_dir") or "")
        if not out_dir:
            raise SystemExit(f"out_dir missing in manifest: {manifest_path}")
        files = _invoke(list_artifact_files, out_dir, max_files=500)
        shard_check = _invoke(validate_pruned_expert_shards, out_dir)
        rep = Path("reports/inspect_eaftreap_budgeted.md")
        rep.parent.mkdir(parents=True, exist_ok=True)
        rep.write_text(
            "\n".join(
                [
                    "# Inspect EAFT-REAP budgeted prune artifacts",
                    "",
                    f"- manifest: `{manifest_path}`",
                    f"- out_dir: `{out_dir}`",
                    "",
                    "## Shard mapping sanity",
                    "",
                    f"- checked: {shard_check.get('checked')}",
                    f"- ok: {shard_check.get('ok')}",
                    f"- mismatches_capped: {shard_check.get('mismatch_count_capped')}",
                    "",
                    "## Files (name, bytes)",
                    "",
                ]
                + [f"- `{r['name']}`: {r['bytes']}" for r in (files.get("files") or [])]
            ),
            encoding="utf-8",
        )
        print(f"[+] Wrote {rep}")
        return

    if task == "validate_expert_math_toy":
        out_report = Path("reports/validate_gpt_oss_expert_math_toy.md")
        out_report.parent.mkdir(parents=True, exist_ok=True)

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        res = _validate_gpt_oss_expert_math_toy(device=device, trials=20)
        lines = [
            "# GPT-OSS experts math equivalence (toy)",
            "",
            "This is a unit-style check to ensure our manual loop implementation matches",
            "`transformers.models.gpt_oss.modeling_gpt_oss.GptOssExperts.forward`.",
            "",
            f"- device: `{device}`",
            f"- max_abs_error: `{res['max_abs']:.3e}`",
            f"- max_rel_error: `{res['max_rel']:.3e}`",
            "",
            "## Interpretation",
            "",
            "- Expect ~0 (within FP32 noise). If this fails, our EAFT-REAP norm/saliency math is suspect.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python modal/gpt_oss_pruning_track.py --task validate_expert_math_toy",
            "```",
            "",
        ]
        out_report.write_text("\n".join(lines), encoding="utf-8")
        print(f"[+] Wrote {out_report} (max_abs={res['max_abs']:.3e}, max_rel={res['max_rel']:.3e})")
        return

    if task == "profile_20b":
        suffix = ""
        if str(domain or "").strip():
            safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(domain).strip())
            suffix = f"_domain_{safe}"
        out_parquet = Path(f"data/20b_expert_usage{suffix}.parquet")
        out_report = Path(f"reports/20b_expert_usage_profile{suffix}.md")
        out_ranking = Path(f"data/20b_expert_ranking_by_layer{suffix}.json")
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        out_report.parent.mkdir(parents=True, exist_ok=True)

        # CPU predownload to avoid spending GPU time on HF downloads.
        _ = predownload_model.remote(str(model_id_20b))

        res = profile_20b_expert_usage.remote(
            model_id=model_id_20b,
            dataset_id=dataset_id,
            dataset_split=dataset_split,
            text_column=text_column,
            domain=str(domain or ""),
            num_rows=int(num_rows),
            max_seq_length=int(max_seq_length),
        )

        out_parquet.write_bytes(res["parquet_bytes"])
        meta = res["meta"]
        ranking = res["expert_ranking_by_layer"]
        usage_counts_by_layer = res.get("usage_counts_by_layer")

        md_lines = [
            "# 20B expert usage profile (GPT-OSS MoE)",
            "",
            f"- Model: `{meta['model_id']}`",
            f"- Dataset: `{meta['dataset_id']}` split `{meta['dataset_split']}` col `{meta['text_column']}`",
            f"- Domain filter: `{meta.get('domain','')}` (empty = no filter)",
            f"- Rows: {meta['num_rows']}",
            f"- Max seq length: {meta['max_seq_length']}",
            f"- Layers: {meta['num_layers']} | Experts: {meta['num_experts']} | Top-k: {meta['top_k']}",
            f"- Total tokens processed: {meta['total_tokens']:,}",
            f"- Forward throughput: {meta['tokens_per_s']:.0f} tokens/s",
            f"- Routed selections recorded: {meta.get('hist_total', 'n/a')}",
            f"- Selected-prob samples recorded: {meta.get('prob_count_total', 'n/a')}",
            "",
            "## Top experts by layer (top 10, count)",
            "",
        ]
        for li, experts in enumerate(ranking):
            if isinstance(usage_counts_by_layer, list) and li < len(usage_counts_by_layer):
                counts = usage_counts_by_layer[li]
                top = [f"{e} ({counts[int(e)]})" for e in experts[:10]]
                md_lines.append(f"- layer_{li}: [{', '.join(top)}]")
            else:
                md_lines.append(f"- layer_{li}: {experts[:10]}")

        md_lines += [
            "",
            "## Artifacts",
            "",
            "- `data/20b_expert_usage.parquet` (usage + co-activation + confidence stats)",
            "- `data/20b_expert_ranking_by_layer.json` (for soft-prune + structural-prune followups)",
            "",
            "## Reproduce",
            "",
            "```bash",
            "modal run modal/gpt_oss_pruning_track.py "
            "--task profile_20b "
            f"--dataset-id {dataset_id} --dataset-split {dataset_split} --text-column {text_column} "
            f"--domain {str(domain or '')} "
            f"--num-rows {int(num_rows)} --max-seq-length {int(max_seq_length)}",
            "```",
            "",
        ]

        out_report.write_text("\n".join(md_lines), encoding="utf-8")
        out_ranking.write_text(json.dumps(ranking, indent=2), encoding="utf-8")
        print(f"[+] Wrote {out_parquet}")
        print(f"[+] Wrote {out_ranking}")
        print(f"[+] Wrote {out_report}")
        return

    if task == "reap_saliency_by_domain_20b":
        domains = _parse_csv(domains_csv)
        if not domains:
            raise SystemExit("No domains specified. Use --domains-csv 'math,science,agentic,general'")

        out_dir = Path("artifacts/reap_saliency_by_domain")
        out_dir.mkdir(parents=True, exist_ok=True)

        domain_results: dict[str, dict[str, Any]] = {}
        for dom in domains:
            dom = str(dom).strip()
            # Domain routing:
            # - `math` uses the dedicated math dataset by default.
            # - `general` maps to `chat_if` in the CPU artifacts dataset (based on scans).
            # - other names are treated as literal `meta_domain` values.
            dsid = str(dataset_id)
            dsplit = str(dataset_split)
            dcol = str(text_column)
            dom_filter = dom
            if dom == "math":
                dsid = str(math_dataset_id)
                dsplit = str(math_dataset_split)
                dcol = str(math_text_column)
                dom_filter = ""
            elif dom == "general":
                dom_filter = "chat_if"
            elif dom == "agentic/tools":
                dom_filter = "agentic"

            sample = sample_domain_rows_cpu.remote(
                dataset_id=str(dsid),
                dataset_split=str(dsplit),
                text_column=str(dcol),
                domain=str(dom_filter),
                domain_column=str(domain_column),
                num_rows=int(num_rows),
                seed=int(seed),
                max_scan_rows=int(max_scan_rows),
            )
            prof = profile_20b_reap_saliency.remote(
                model_id=model_id_20b,
                dataset_id=str(dsid),
                dataset_split=str(dsplit),
                text_column=str(dcol),
                domain=str(dom_filter),
                num_rows=int(num_rows),
                max_seq_length=int(max_seq_length),
                batch_size=int(batch_size),
                rows_jsonl_path=str(sample["rows_jsonl_path"]),
            )

            parquet_path = out_dir / f"{dom}.parquet"
            ranking_path = out_dir / f"{dom}_ranking_by_layer.json"
            parquet_path.write_bytes(prof["parquet_bytes"])
            ranking_path.write_text(json.dumps(prof["ranking_by_layer"], indent=2), encoding="utf-8")
            print(f"[+] Wrote {parquet_path}")
            print(f"[+] Wrote {ranking_path}")

            domain_results[dom] = {
                "sample": sample,
                "meta": prof.get("meta", {}),
                "concentration_by_layer": prof.get("concentration_by_layer", []),
                "paths": {"parquet": str(parquet_path), "ranking": str(ranking_path)},
                "source": {"dataset_id": dsid, "dataset_split": dsplit, "text_column": dcol, "domain_filter": dom_filter},
            }

        rep = Path("reports/reap_saliency_by_domain.md")
        rep.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = [
            "# 20B REAP-lite saliency by domain",
            "",
            f"- Base model: `{model_id_20b}`",
            f"- Dataset: `{dataset_id}` split `{dataset_split}` col `{text_column}`",
            f"- Domain column: `{domain_column}`",
            f"- Domains: {', '.join(domains)}",
            f"- Rows/domain: {int(num_rows)} | Seed: {int(seed)} | Max scan rows: {int(max_scan_rows)}",
            f"- Max seq length: {int(max_seq_length)} | Batch size: {int(batch_size)}",
            "",
        ]
        for dom in domains:
            d = domain_results.get(dom, {})
            s = d.get("sample", {})
            m = d.get("meta", {})
            src = d.get("source", {})
            conc = d.get("concentration_by_layer", [])
            lines += [
                f"## {dom}",
                "",
                f"- source: `{src.get('dataset_id','')}` split `{src.get('dataset_split','')}` col `{src.get('text_column','')}` domain_filter=`{src.get('domain_filter','')}`",
                f"- scanned={s.get('scanned')} matched={s.get('matched')} sample_path=`{s.get('rows_jsonl_path')}`",
                f"- kept_tokens={m.get('total_kept_tokens')} total_tokens={m.get('total_tokens')} tok_s_pred={float(m.get('tokens_per_s') or 0):.0f}",
                f"- parquet: `{d.get('paths',{}).get('parquet','')}`",
                "",
            ]
            if conc:
                for li in (0, 1, 10, 23):
                    if li >= len(conc):
                        continue
                    c = conc[li]
                    lines.append(
                        f"- layer_{li}: top4={float(c.get('top_4',0)):.3f} top8={float(c.get('top_8',0)):.3f} top16={float(c.get('top_16',0)):.3f}"
                    )
                lines.append("")

        rep.write_text("\n".join(lines), encoding="utf-8")
        print(f"[+] Wrote {rep}")
        return

    if task == "scan_domain_values_20b":
        res = scan_domain_counts_cpu.remote(
            dataset_id=str(dataset_id),
            dataset_split=str(dataset_split),
            domain_column=str(domain_column),
            max_scan_rows=int(max_scan_rows),
        )
        out = Path("reports/domain_value_scan.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        counts = res.get("counts") or {}
        top = sorted(counts.items(), key=lambda kv: int(kv[1]), reverse=True)[:50]
        lines = [
            "# Domain value scan",
            "",
            f"- Dataset: `{res.get('dataset_id')}` split `{res.get('dataset_split')}`",
            f"- Column: `{res.get('domain_column')}`",
            f"- Scanned: {res.get('scanned')} (max {res.get('max_scan_rows')}) | Missing: {res.get('missing')}",
            "",
            "## Top values",
            "",
        ]
        for k, v in top:
            lines.append(f"- {k}: {v}")
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"[+] Wrote {out}")
        return

    if task == "build_union_expert_sets_20b":
        # Build per-layer union expert sets from previously saved
        # artifacts/reap_saliency_by_domain/<domain>.parquet.
        domains = _parse_csv(domains_csv)
        if not domains:
            raise SystemExit("No domains specified. Use --domains-csv 'math,science,agentic,general'")

        sal_dir = Path("artifacts/reap_saliency_by_domain")
        if not sal_dir.exists():
            raise SystemExit(f"Missing `{sal_dir}`. Run --task reap_saliency_by_domain_20b first.")

        domain_masses: dict[str, dict[int, list[float]]] = {}
        for dom in domains:
            p = sal_dir / f"{dom}.parquet"
            if not p.exists():
                raise SystemExit(f"Missing `{p}`. Run --task reap_saliency_by_domain_20b for domain={dom}.")
            domain_masses[dom] = _read_reap_saliency_mass_parquet(p)

        # Default weighting reflects our priorities; can be adjusted later.
        weights = {d: 1.0 for d in domains}
        weights.setdefault("general", 1.0)
        weights.setdefault("science", 1.1)
        weights.setdefault("agentic", 1.2)
        weights.setdefault("math", 1.3)

        out_dir = Path("artifacts/union_expert_sets")
        out_dir.mkdir(parents=True, exist_ok=True)

        covs = [0.95, 0.97, 0.99]
        for cov in covs:
            keep = _union_keep_sets(domain_masses, cov=float(cov), weights=weights, core_n=4, cap=None)
            (out_dir / f"union_cov{int(cov*100)}.json").write_text(
                json.dumps(
                    {
                        "coverage": float(cov),
                        "domains": domains,
                        "weights": weights,
                        "core_n": 4,
                        "cap": None,
                        "keep_experts_by_layer": keep,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        # Derived variants for pruning.
        union50 = _union_keep_sets(domain_masses, cov=0.95, weights=weights, core_n=4, cap=16)
        union_agg = _union_keep_sets(domain_masses, cov=0.97, weights=weights, core_n=4, cap=12)
        (out_dir / "union50_cap16_cov95.json").write_text(
            json.dumps(
                {
                    "name": "union50",
                    "coverage": 0.95,
                    "cap": 16,
                    "domains": domains,
                    "weights": weights,
                    "core_n": 4,
                    "keep_experts_by_layer": union50,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (out_dir / "unionAgg_cap12_cov97.json").write_text(
            json.dumps(
                {
                    "name": "unionAgg",
                    "coverage": 0.97,
                    "cap": 12,
                    "domains": domains,
                    "weights": weights,
                    "core_n": 4,
                    "keep_experts_by_layer": union_agg,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # Report counts.
        def _counts(keep_by_layer: list[list[int]]) -> dict[str, Any]:
            counts = [len(x) for x in keep_by_layer]
            return {
                "min": int(min(counts)) if counts else 0,
                "max": int(max(counts)) if counts else 0,
                "avg": float(sum(counts) / max(1, len(counts))) if counts else 0.0,
                "counts": counts,
            }

        rep = Path("reports/union_policy.md")
        rep.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Union expert-set policy (REAP-lite)",
            "",
            f"- Domains: {', '.join(domains)} (column `{domain_column}`)",
            f"- Rows/domain used: {int(num_rows)} (see `artifacts/reap_saliency_by_domain/`)",
            f"- Weights: {json.dumps(weights)}",
            "",
            "## Outputs",
            "",
            "- `artifacts/union_expert_sets/union_cov95.json`",
            "- `artifacts/union_expert_sets/union_cov97.json`",
            "- `artifacts/union_expert_sets/union_cov99.json`",
            "- `artifacts/union_expert_sets/union50_cap16_cov95.json`",
            "- `artifacts/union_expert_sets/unionAgg_cap12_cov97.json`",
            "",
            "## Kept-expert counts (per layer)",
            "",
            f"- union50 (cap16@cov95): {json.dumps(_counts(union50))}",
            f"- unionAgg (cap12@cov97): {json.dumps(_counts(union_agg))}",
            "",
        ]
        rep.write_text("\n".join(lines), encoding="utf-8")
        print(f"[+] Wrote {rep}")
        return

    if task == "build_pruned_20b_union":
        # Structural-prune checkpoints using previously computed union sets.
        union_dir = Path("artifacts/union_expert_sets")
        u50_path = union_dir / "union50_cap16_cov95.json"
        uagg_path = union_dir / "unionAgg_cap12_cov97.json"
        if not u50_path.exists() or not uagg_path.exists():
            raise SystemExit(
                f"Missing union sets under `{union_dir}`. Run --task build_union_expert_sets_20b first."
            )
        u50 = json.loads(u50_path.read_text(encoding="utf-8"))
        uagg = json.loads(uagg_path.read_text(encoding="utf-8"))
        keep50 = u50.get("keep_experts_by_layer")
        keepagg = uagg.get("keep_experts_by_layer")
        if not isinstance(keep50, list) or not isinstance(keepagg, list):
            raise SystemExit("Invalid union json; missing keep_experts_by_layer.")

        union50_dir = structural_prune_20b_build.remote(
            model_id=model_id_20b,
            variant_name="union50",
            keep_experts_by_layer_json=json.dumps(keep50),
            out_subdir="20b_union_pruned",
        )
        unionagg_dir = structural_prune_20b_build.remote(
            model_id=model_id_20b,
            variant_name="unionAgg",
            keep_experts_by_layer_json=json.dumps(keepagg),
            out_subdir="20b_union_pruned",
        )
        ok50 = _invoke(sanity_infer_model_dir, model_dir=union50_dir)
        okagg = _invoke(sanity_infer_model_dir, model_dir=unionagg_dir)

        artifacts_dir = Path("artifacts/20b_union_pruned")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "base_model": str(model_id_20b),
            "dataset_id": str(dataset_id),
            "dataset_split": str(dataset_split),
            "text_column": str(text_column),
            "domain_column": str(domain_column),
            "domains": _parse_csv(domains_csv),
            "seed": int(seed),
            "num_rows_per_domain": int(num_rows),
            "max_scan_rows": int(max_scan_rows),
            "union50": {"dir": union50_dir, "ok": bool(ok50.get("ok"))},
            "unionAgg": {"dir": unionagg_dir, "ok": bool(okagg.get("ok"))},
            "union_set_files": {"union50": str(u50_path), "unionAgg": str(uagg_path)},
        }
        (artifacts_dir / "manifest_union.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        rep = Path("reports/20b_union_prune_build.md")
        rep.parent.mkdir(parents=True, exist_ok=True)
        rep.write_text(
            "\n".join(
                [
                    "# 20B structural prune build (union expert set)",
                    "",
                    f"- Base model: `{model_id_20b}`",
                    f"- union50: `{union50_dir}` ok={ok50.get('ok')}",
                    f"- unionAgg: `{unionagg_dir}` ok={okagg.get('ok')}",
                    "",
                    f"- Manifest: `{artifacts_dir/'manifest_union.json'}`",
                    "",
                    "## Reproduce",
                    "",
                    "```bash",
                    "modal run modal/gpt_oss_pruning_track.py --task build_pruned_20b_union",
                    "```",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"[+] Wrote {artifacts_dir/'manifest_union.json'}")
        print(f"[+] Wrote {rep}")
        return

    if task == "reap_saliency_20b":
        suffix = ""
        if str(domain or "").strip():
            safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(domain).strip())
            suffix = f"_domain_{safe}"
        out_parquet = Path(f"data/20b_reap_saliency{suffix}.parquet")
        out_report = Path(f"reports/20b_reap_saliency_profile{suffix}.md")
        out_ranking = Path(f"data/20b_reap_saliency_ranking_by_layer{suffix}.json")
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        out_report.parent.mkdir(parents=True, exist_ok=True)

        # CPU predownload to avoid spending GPU time on HF downloads.
        _ = _invoke(predownload_model, str(model_id_20b))

        res = _invoke(
            profile_20b_reap_saliency,
            model_id=model_id_20b,
            dataset_id=dataset_id,
            dataset_split=dataset_split,
            text_column=text_column,
            domain=str(domain or ""),
            num_rows=int(num_rows),
            max_seq_length=int(max_seq_length),
            batch_size=int(batch_size),
        )
        out_parquet.write_bytes(res["parquet_bytes"])
        meta = res["meta"]
        ranking = res["ranking_by_layer"]
        conc = res.get("concentration_by_layer") or []

        md_lines = [
            "# 20B REAP-lite saliency profile (GPT-OSS MoE)",
            "",
            f"- Model: `{meta['model_id']}`",
            f"- Dataset: `{meta['dataset_id']}` split `{meta['dataset_split']}` col `{meta['text_column']}`",
            f"- Domain filter: `{meta.get('domain','')}` (empty = no filter)",
            f"- Rows: {meta['num_rows']} | Batch size: {meta['batch_size']}",
            f"- Max seq length: {meta['max_seq_length']}",
            f"- Layers: {meta['num_layers']} | Experts: {meta['num_experts']} | Top-k: {meta['top_k']}",
            f"- Total tokens processed: {meta['total_tokens']:,} | Kept (assistant-span) tokens: {meta['total_kept_tokens']:,}",
            f"- Forward throughput: {meta['tokens_per_s']:.0f} tokens/s",
            f"- Prompts hash: `{meta['prompt_hash']}`",
            "",
            "## Top experts by layer (top 10, ranked by saliency_mean)",
            "",
        ]
        for li, experts in enumerate(ranking):
            md_lines.append(f"- layer_{li}: {experts[:10]}")

        if conc:
            md_lines += [
                "",
                "## Saliency concentration (by gate_norm_sum mass)",
                "",
                "| layer | top_4 | top_8 | top_16 |",
                "|---:|---:|---:|---:|",
            ]
            for li, row in enumerate(conc):
                md_lines.append(
                    f"| {li} | {float(row.get('top_4', 0.0)):.3f} | {float(row.get('top_8', 0.0)):.3f} | {float(row.get('top_16', 0.0)):.3f} |"
                )

        md_lines += [
            "",
            "## Artifacts",
            "",
            "- `data/20b_reap_saliency.parquet` (per-layer, per-expert count + gate/norm/saliency stats)",
            "- `data/20b_reap_saliency_ranking_by_layer.json` (sorted experts per layer)",
            "",
            "## Reproduce",
            "",
            "```bash",
            "modal run modal/gpt_oss_pruning_track.py "
            "--task reap_saliency_20b "
            f"--dataset-id {dataset_id} --dataset-split {dataset_split} --text-column {text_column} "
            f"--domain {str(domain or '')} "
            f"--num-rows {int(num_rows)} --max-seq-length {int(max_seq_length)} --batch-size {int(batch_size)}",
            "```",
            "",
        ]

        out_report.write_text("\n".join(md_lines), encoding="utf-8")
        out_ranking.write_text(json.dumps(ranking, indent=2), encoding="utf-8")
        print(f"[+] Wrote {out_parquet}")
        print(f"[+] Wrote {out_ranking}")
        print(f"[+] Wrote {out_report}")
        return

    if task == "eaftreap_saliency_20b":
        suffix = ""
        if str(domain or "").strip():
            safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(domain).strip())
            suffix = f"_domain_{safe}"
        out_parquet = Path(f"data/20b_eaftreap_saliency{suffix}.parquet")
        out_report = Path(f"reports/20b_eaftreap_saliency_profile{suffix}.md")
        out_ranking = Path(f"data/20b_eaftreap_saliency_ranking_by_layer{suffix}.json")
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        out_report.parent.mkdir(parents=True, exist_ok=True)

        # CPU predownload to avoid spending GPU time on HF downloads.
        _ = _invoke(predownload_model, str(model_id_20b))

        res = _invoke(
            profile_20b_eaftreap_saliency,
            model_id=model_id_20b,
            dataset_id=dataset_id,
            dataset_split=dataset_split,
            text_column=text_column,
            domain=str(domain or ""),
            num_rows=int(num_rows),
            max_seq_length=int(max_seq_length),
            batch_size=int(batch_size),
            cc_quantile=float(eaft_cc_quantile),
            uncertain_quantile=float(eaft_uncertain_quantile),
            entropy_topk=int(eaft_entropy_topk),
            w_good=float(eaft_w_good),
            w_uncertain=float(eaft_w_uncertain),
            w_conflict=float(eaft_w_conflict),
        )
        out_parquet.write_bytes(res["parquet_bytes"])
        meta = res["meta"]
        ranking = res["ranking_by_layer"]

        rc = meta.get("region_counts") or {}
        md_lines = [
            "# 20B EAFT-REAP saliency profile (correctness-aware)",
            "",
            f"- Model: `{meta['model_id']}`",
            f"- Dataset: `{meta['dataset_id']}` split `{meta['dataset_split']}` col `{meta['text_column']}`",
            f"- Domain filter: `{meta.get('domain','')}` (empty = no filter)",
            f"- Rows: {meta['num_rows']} | Batch size: {meta['batch_size']}",
            f"- Max seq length: {meta['max_seq_length']}",
            f"- Layers: {meta['num_layers']} | Experts: {meta['num_experts']} | Top-k: {meta['top_k']}",
            f"- EAFT entropy top-k: {meta.get('entropy_topk')}",
            f"- Thresholds: p_lo={meta.get('p_lo'):.6f} | h_lo={meta.get('h_lo'):.6f} | h_hi={meta.get('h_hi'):.6f}",
            f"- Weights: good={meta.get('w_good')} uncertain={meta.get('w_uncertain')} conflict={meta.get('w_conflict')}",
            f"- Region counts: good={rc.get('good', 0)} conflict={rc.get('conflict', 0)} uncertain={rc.get('uncertain', 0)}",
            f"- Forward throughput (pass2): {meta['tokens_per_s_pass2']:.0f} tokens/s",
            f"- Prompts hash: `{meta['prompt_hash']}`",
            "",
            "## Top experts by layer (top 10, ranked by eaft_saliency_mean)",
            "",
        ]
        for li, experts in enumerate(ranking):
            md_lines.append(f"- layer_{li}: {experts[:10]}")

        md_lines += [
            "",
            "## Artifacts",
            "",
            "- `data/20b_eaftreap_saliency.parquet` (per-layer, per-expert count + EAFT-weighted saliency stats)",
            "- `data/20b_eaftreap_saliency_ranking_by_layer.json` (sorted experts per layer)",
            "",
            "## Reproduce",
            "",
            "```bash",
            "modal run modal/gpt_oss_pruning_track.py "
            "--task eaftreap_saliency_20b "
            f"--dataset-id {dataset_id} --dataset-split {dataset_split} --text-column {text_column} "
            f"--domain {str(domain or '')} "
            f"--num-rows {int(num_rows)} --max-seq-length {int(max_seq_length)} --batch-size {int(batch_size)} "
            f"--eaft-cc-quantile {float(eaft_cc_quantile)} --eaft-uncertain-quantile {float(eaft_uncertain_quantile)} "
            f"--eaft-entropy-topk {int(eaft_entropy_topk)} --eaft-w-good {float(eaft_w_good)} "
            f"--eaft-w-uncertain {float(eaft_w_uncertain)} --eaft-w-conflict {float(eaft_w_conflict)}",
            "```",
            "",
        ]
        out_report.write_text("\n".join(md_lines), encoding="utf-8")
        out_ranking.write_text(json.dumps(ranking, indent=2), encoding="utf-8")
        print(f"[+] Wrote {out_parquet}")
        print(f"[+] Wrote {out_ranking}")
        print(f"[+] Wrote {out_report}")
        return

    if task == "soft_prune_20b":
        suffix = ""
        if str(domain or "").strip():
            safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(domain).strip())
            suffix = f"_domain_{safe}"
        ranking_path = Path(f"data/20b_expert_ranking_by_layer{suffix}.json")
        if not ranking_path.exists():
            raise SystemExit(
                f"Missing `{ranking_path}`. Run --task profile_20b first (with the same --domain)."
            )
        expert_ranking_by_layer_json = ranking_path.read_text(encoding="utf-8")
        res = soft_prune_20b_eval.remote(
            model_id=model_id_20b,
            dataset_id=dataset_id,
            dataset_split=dataset_split,
            text_column=text_column,
            expert_ranking_by_layer_json=expert_ranking_by_layer_json,
            keep_fracs_csv="1.0,0.5,0.25",
            top_ks_csv="4,2",
            eval_rows=256,
            max_seq_length=int(max_seq_length),
        )
        out_report = Path("reports/20b_soft_prune_eval.md")
        out_report.parent.mkdir(parents=True, exist_ok=True)

        results = res["results"]
        baseline = None
        for r in results:
            if r["keep_frac"] == 1.0 and r["top_k"] == 4:
                baseline = r
                break
        if baseline is None:
            baseline = results[0]

        lines = [
            "# 20B soft prune eval (inference-only)",
            "",
            f"- Model: `{res['meta']['model_id']}`",
            f"- Dataset: `{res['meta']['dataset_id']}` split `{res['meta']['dataset_split']}`",
            f"- Eval rows: {results[0]['eval_rows']} | Max seq length: {results[0]['max_seq_length']}",
            "",
            "| kept_experts | keep_frac | top_k | ppl | ppl_delta | tokens/s |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for r in sorted(results, key=lambda x: (x["keep_frac"], x["top_k"])):
            ppl_delta = float(r["ppl"]) - float(baseline["ppl"])
            lines.append(
                f"| {r['keep_n']} | {r['keep_frac']:.2f} | {r['top_k']} | {r['ppl']:.3f} | {ppl_delta:+.3f} | {r['tokens_per_s']:.0f} |"
            )
        lines += [
            "",
            "## Reproduce",
            "",
            "```bash",
            "modal run modal/gpt_oss_pruning_track.py --task soft_prune_20b",
            "```",
            "",
        ]
        out_report.write_text("\n".join(lines), encoding="utf-8")
        print(f"[+] Wrote {out_report}")
        return

    if task == "soft_prune_20b_eaftreap":
        # Soft-prune using EAFT-REAP ranking (correctness-aware), rather than
        # frequency ranking. This is the preferred A/B before structural prune.
        ranking_path = Path("data/20b_eaftreap_saliency_ranking_by_layer.json")
        if not ranking_path.exists():
            raise SystemExit(
                f"Missing `{ranking_path}`. Run --task eaftreap_saliency_20b first."
            )
        expert_ranking_by_layer_json = ranking_path.read_text(encoding="utf-8")
        res = soft_prune_20b_eval.remote(
            model_id=model_id_20b,
            dataset_id=dataset_id,
            dataset_split=dataset_split,
            text_column=text_column,
            expert_ranking_by_layer_json=expert_ranking_by_layer_json,
            keep_fracs_csv="1.0,0.5,0.25",
            top_ks_csv="4,2",
            eval_rows=256,
            max_seq_length=int(max_seq_length),
        )
        out_report = Path("reports/20b_soft_prune_eval_eaftreap.md")
        out_report.parent.mkdir(parents=True, exist_ok=True)

        results = res["results"]
        baseline = None
        for r in results:
            if r["keep_frac"] == 1.0 and r["top_k"] == 4:
                baseline = r
                break
        if baseline is None:
            baseline = results[0]

        lines = [
            "# 20B soft prune eval (EAFT-REAP ranking, inference-only)",
            "",
            f"- Model: `{res['meta']['model_id']}`",
            f"- Dataset: `{res['meta']['dataset_id']}` split `{res['meta']['dataset_split']}`",
            f"- Eval rows: {results[0]['eval_rows']} | Max seq length: {results[0]['max_seq_length']}",
            "",
            "| kept_experts | keep_frac | top_k | ppl | ppl_delta | tokens/s |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for r in sorted(results, key=lambda x: (x["keep_frac"], x["top_k"])):
            ppl_delta = float(r["ppl"]) - float(baseline["ppl"])
            lines.append(
                f"| {r['keep_n']} | {r['keep_frac']:.2f} | {r['top_k']} | {r['ppl']:.3f} | {ppl_delta:+.3f} | {r['tokens_per_s']:.0f} |"
            )
        lines += [
            "",
            "## Reproduce",
            "",
            "```bash",
            "modal run modal/gpt_oss_pruning_track.py --task soft_prune_20b_eaftreap",
            "```",
            "",
        ]
        out_report.write_text("\n".join(lines), encoding="utf-8")
        print(f"[+] Wrote {out_report}")
        return

    if task == "build_pruned_20b":
        def _load_topical(path: Path, *, num_layers: int) -> list[list[int]]:
            data = json.loads(path.read_text(encoding="utf-8"))
            out: list[list[int]] = []
            for li in range(num_layers):
                k = f"layer_{li}"
                if k not in data:
                    raise KeyError(f"Missing key {k} in {path}")
                out.append([int(x) for x in data[k]])
            return out

        all_path = Path("third_party/GPT-OSS-MoE-ExpertFingerprinting/topical_analytics/all.json")
        math_path = Path("third_party/GPT-OSS-MoE-ExpertFingerprinting/topical_analytics/math.json")
        if not all_path.exists() or not math_path.exists():
            raise SystemExit(
                "Missing topical analytics JSON. Ensure `third_party/GPT-OSS-MoE-ExpertFingerprinting` is present."
            )

        # GPT-OSS-20B has 24 layers.
        num_layers = 24
        all_rank = _load_topical(all_path, num_layers=num_layers)
        math_rank = _load_topical(math_path, num_layers=num_layers)

        general_keep = [layer[:16] for layer in all_rank]  # 50% of 32 experts
        math_keep = [layer[:8] for layer in math_rank]  # 25% of 32 experts

        general_dir = structural_prune_20b_build.remote(
            model_id=model_id_20b,
            variant_name="general_50pct_experts",
            keep_experts_by_layer_json=json.dumps(general_keep),
        )
        math_dir = structural_prune_20b_build.remote(
            model_id=model_id_20b,
            variant_name="math_25pct_experts",
            keep_experts_by_layer_json=json.dumps(math_keep),
        )

        general_ok = sanity_infer_model_dir.remote(model_dir=general_dir)
        math_ok = sanity_infer_model_dir.remote(model_dir=math_dir)

        out_report = Path("reports/20b_structural_prune_build.md")
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(
            "\n".join(
                [
                    "# 20B structural prune build",
                    "",
                    f"- Base model: `{model_id_20b}`",
                    "",
                    "## Variants",
                    "",
                    f"- general_50pct_experts: `{general_dir}`",
                    f"- math_25pct_experts: `{math_dir}`",
                    "",
                    "## Sanity inference",
                    "",
                    f"- general_50pct_experts ok={general_ok.get('ok')} preview:",
                    "```",
                    str(general_ok.get("generated", "")),
                    "```",
                    "",
                    f"- math_25pct_experts ok={math_ok.get('ok')} preview:",
                    "```",
                    str(math_ok.get("generated", "")),
                    "```",
                    "",
                    "## Reproduce",
                    "",
                    "```bash",
                    "modal run modal/gpt_oss_pruning_track.py --task build_pruned_20b",
                    "```",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        artifacts_dir = Path("artifacts/20b_pruned_models")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "base_model": model_id_20b,
                    "variants": {
                        "general_50pct_experts": general_dir,
                        "math_25pct_experts": math_dir,
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"[+] Wrote {out_report}")
        print(f"[+] Wrote {artifacts_dir/'manifest.json'}")
        return

    if task == "build_pruned_20b_freq":
        # Frequency baseline using `profile_20b_expert_usage` ranking (per-layer counts).
        # We produce 2 pruned variants under `/root/model/artifacts/20b_pruned_models_freq/...`.
        # - general: keep 16/32
        # - math: keep 8/32 (uses a math-only dataset by default)

        # CPU predownload to avoid spending GPU time on HF downloads.
        _ = predownload_model.remote(str(model_id_20b))

        # Build general ranking (domain="") and math ranking (from math dataset).
        general_prof = profile_20b_expert_usage.remote(
            model_id=model_id_20b,
            dataset_id=dataset_id,
            dataset_split=dataset_split,
            text_column=text_column,
            domain="",
            num_rows=int(num_rows),
            max_seq_length=int(max_seq_length),
        )
        math_prof = profile_20b_expert_usage.remote(
            model_id=model_id_20b,
            dataset_id=math_dataset_id,
            dataset_split=math_dataset_split,
            text_column=math_text_column,
            domain="",
            num_rows=int(num_rows),
            max_seq_length=int(max_seq_length),
        )

        general_rank = general_prof["expert_ranking_by_layer"]
        math_rank = math_prof["expert_ranking_by_layer"]
        num_layers = len(general_rank)

        general_keep = [layer[:16] for layer in general_rank]
        math_keep = [layer[:8] for layer in math_rank]

        general_dir = structural_prune_20b_build.remote(
            model_id=model_id_20b,
            variant_name="general_50pct_experts_freq",
            keep_experts_by_layer_json=json.dumps(general_keep),
            out_subdir="20b_pruned_models_freq",
        )
        math_dir = structural_prune_20b_build.remote(
            model_id=model_id_20b,
            variant_name="math_25pct_experts_freq",
            keep_experts_by_layer_json=json.dumps(math_keep),
            out_subdir="20b_pruned_models_freq",
        )

        general_ok = sanity_infer_model_dir.remote(model_dir=general_dir)
        math_ok = sanity_infer_model_dir.remote(model_dir=math_dir)

        artifacts_dir = Path("artifacts/20b_pruned_models_freq")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "manifest_freq.json").write_text(
            json.dumps(
                {
                    "base_model": model_id_20b,
                    "general_dataset": {
                        "dataset_id": dataset_id,
                        "dataset_split": dataset_split,
                        "text_column": text_column,
                    },
                    "math_dataset": {
                        "dataset_id": math_dataset_id,
                        "dataset_split": math_dataset_split,
                        "text_column": math_text_column,
                    },
                    "general_profile_domain": "",
                    "math_profile_domain": "",
                    "keep_frac_general": 0.50,
                    "keep_frac_math": 0.25,
                    "variants": {
                        "general_50pct_experts_freq": general_dir,
                        "math_25pct_experts_freq": math_dir,
                    },
                    "sanity": {
                        "general_ok": bool(general_ok.get("ok")),
                        "math_ok": bool(math_ok.get("ok")),
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        out_report = Path("reports/20b_structural_prune_build_freq.md")
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(
            "\n".join(
                [
                    "# 20B structural prune build (frequency baseline)",
                    "",
                    f"- Base model: `{model_id_20b}`",
                    f"- General dataset: `{dataset_id}` split `{dataset_split}` col `{text_column}`",
                    f"- Math dataset: `{math_dataset_id}` split `{math_dataset_split}` col `{math_text_column}`",
                    f"- Profile rows: {int(num_rows)} | Max seq length: {int(max_seq_length)}",
                    "",
                    "## Variants",
                    "",
                    f"- general_50pct_experts_freq: `{general_dir}`",
                    f"- math_25pct_experts_freq: `{math_dir}`",
                    "",
                    "## Sanity inference",
                    "",
                    f"- general ok={general_ok.get('ok')}",
                    f"- math ok={math_ok.get('ok')}",
                    "",
                    "## Artifacts",
                    "",
                    f"- `{artifacts_dir/'manifest_freq.json'}`",
                    "",
                    "## Reproduce",
                    "",
                    "```bash",
                    "modal run modal/gpt_oss_pruning_track.py --task build_pruned_20b_freq",
                    "```",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"[+] Wrote {artifacts_dir/'manifest_freq.json'}")
        print(f"[+] Wrote {out_report}")
        return

    if task == "build_pruned_20b_reap":
        # REAP-lite saliency ranked structural prunes.
        # We produce 2 pruned variants under `/root/model/artifacts/20b_pruned_models_reap/...`.
        # - general: keep 16/32 (domain="")
        # - math: keep 8/32 (uses a math-only dataset by default)

        # CPU predownload to avoid spending GPU time on HF downloads.
        _ = predownload_model.remote(str(model_id_20b))

        general_prof = profile_20b_reap_saliency.remote(
            model_id=model_id_20b,
            dataset_id=dataset_id,
            dataset_split=dataset_split,
            text_column=text_column,
            domain="",
            num_rows=int(num_rows),
            max_seq_length=int(max_seq_length),
            batch_size=int(batch_size),
        )
        math_prof = profile_20b_reap_saliency.remote(
            model_id=model_id_20b,
            dataset_id=math_dataset_id,
            dataset_split=math_dataset_split,
            text_column=math_text_column,
            domain="",
            num_rows=int(num_rows),
            max_seq_length=int(max_seq_length),
            batch_size=int(batch_size),
        )

        general_rank = general_prof["ranking_by_layer"]
        math_rank = math_prof["ranking_by_layer"]

        general_keep = [layer[:16] for layer in general_rank]
        math_keep = [layer[:8] for layer in math_rank]

        general_dir = structural_prune_20b_build.remote(
            model_id=model_id_20b,
            variant_name="general_50pct_experts_reap",
            keep_experts_by_layer_json=json.dumps(general_keep),
            out_subdir="20b_pruned_models_reap",
        )
        math_dir = structural_prune_20b_build.remote(
            model_id=model_id_20b,
            variant_name="math_25pct_experts_reap",
            keep_experts_by_layer_json=json.dumps(math_keep),
            out_subdir="20b_pruned_models_reap",
        )

        general_ok = sanity_infer_model_dir.remote(model_dir=general_dir)
        math_ok = sanity_infer_model_dir.remote(model_dir=math_dir)

        artifacts_dir = Path("artifacts/20b_pruned_models_reap")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "manifest_reap.json").write_text(
            json.dumps(
                {
                    "base_model": model_id_20b,
                    "general_dataset": {
                        "dataset_id": dataset_id,
                        "dataset_split": dataset_split,
                        "text_column": text_column,
                    },
                    "math_dataset": {
                        "dataset_id": math_dataset_id,
                        "dataset_split": math_dataset_split,
                        "text_column": math_text_column,
                    },
                    "keep_frac_general": 0.50,
                    "keep_frac_math": 0.25,
                    "profile_top_k": int(general_prof["meta"]["top_k"]),
                    "general_profile": {
                        "domain": "",
                        "prompt_hash": general_prof["meta"]["prompt_hash"],
                    },
                    "math_profile": {
                        "domain": "",
                        "prompt_hash": math_prof["meta"]["prompt_hash"],
                    },
                    "variants": {
                        "general_50pct_experts_reap": general_dir,
                        "math_25pct_experts_reap": math_dir,
                    },
                    "sanity": {
                        "general_ok": bool(general_ok.get("ok")),
                        "math_ok": bool(math_ok.get("ok")),
                    },
                    "keep_experts_by_layer": {
                        "general": general_keep,
                        "math": math_keep,
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        out_report = Path("reports/20b_structural_prune_build_reap.md")
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(
            "\n".join(
                [
                    "# 20B structural prune build (REAP-lite ranking)",
                    "",
                    f"- Base model: `{model_id_20b}`",
                    f"- General dataset: `{dataset_id}` split `{dataset_split}` col `{text_column}`",
                    f"- Math dataset: `{math_dataset_id}` split `{math_dataset_split}` col `{math_text_column}`",
                    f"- Profile rows: {int(num_rows)} | Max seq length: {int(max_seq_length)} | Batch size: {int(batch_size)}",
                    "",
                    "## Variants",
                    "",
                    f"- general_50pct_experts_reap: `{general_dir}`",
                    f"- math_25pct_experts_reap: `{math_dir}`",
                    "",
                    "## Sanity inference",
                    "",
                    f"- general ok={general_ok.get('ok')}",
                    f"- math ok={math_ok.get('ok')}",
                    "",
                    "## Artifacts",
                    "",
                    f"- `{artifacts_dir/'manifest_reap.json'}`",
                    "",
                    "## Reproduce",
                    "",
                    "```bash",
                    "modal run modal/gpt_oss_pruning_track.py --task build_pruned_20b_reap",
                    "```",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"[+] Wrote {artifacts_dir/'manifest_reap.json'}")
        print(f"[+] Wrote {out_report}")
        return

    if task == "build_pruned_20b_eaftreap":
        # EAFT-REAP ranked structural prunes (correctness-aware).
        # We produce 2 pruned variants under `/root/model/artifacts/20b_pruned_models_eaftreap/...`.
        # - general: keep 16/32 (domain="")
        # - math: keep 8/32 (uses a math-only dataset by default)

        # CPU predownload to avoid spending GPU time on HF downloads.
        _ = _invoke(predownload_model, str(model_id_20b))

        general_prof = _invoke(
            profile_20b_eaftreap_saliency,
            model_id=model_id_20b,
            dataset_id=dataset_id,
            dataset_split=dataset_split,
            text_column=text_column,
            domain="",
            num_rows=int(num_rows),
            max_seq_length=int(max_seq_length),
            batch_size=int(batch_size),
            cc_quantile=float(eaft_cc_quantile),
            uncertain_quantile=float(eaft_uncertain_quantile),
            entropy_topk=int(eaft_entropy_topk),
            w_good=float(eaft_w_good),
            w_uncertain=float(eaft_w_uncertain),
            w_conflict=float(eaft_w_conflict),
        )
        math_prof = _invoke(
            profile_20b_eaftreap_saliency,
            model_id=model_id_20b,
            dataset_id=math_dataset_id,
            dataset_split=math_dataset_split,
            text_column=math_text_column,
            domain="",
            num_rows=int(num_rows),
            max_seq_length=int(max_seq_length),
            batch_size=int(batch_size),
            cc_quantile=float(eaft_cc_quantile),
            uncertain_quantile=float(eaft_uncertain_quantile),
            entropy_topk=int(eaft_entropy_topk),
            w_good=float(eaft_w_good),
            w_uncertain=float(eaft_w_uncertain),
            w_conflict=float(eaft_w_conflict),
        )

        general_rank = general_prof["ranking_by_layer"]
        math_rank = math_prof["ranking_by_layer"]

        general_keep = [layer[:16] for layer in general_rank]
        math_keep = [layer[:8] for layer in math_rank]

        general_dir = _invoke(
            structural_prune_20b_build,
            model_id=model_id_20b,
            variant_name="general_50pct_experts_eaftreap",
            keep_experts_by_layer_json=json.dumps(general_keep),
            out_subdir="20b_pruned_models_eaftreap",
        )
        math_dir = _invoke(
            structural_prune_20b_build,
            model_id=model_id_20b,
            variant_name="math_25pct_experts_eaftreap",
            keep_experts_by_layer_json=json.dumps(math_keep),
            out_subdir="20b_pruned_models_eaftreap",
        )

        general_ok = _invoke(sanity_infer_model_dir, model_dir=general_dir)
        math_ok = _invoke(sanity_infer_model_dir, model_dir=math_dir)

        artifacts_dir = Path("artifacts/20b_pruned_models_eaftreap")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "manifest_eaftreap.json").write_text(
            json.dumps(
                {
                    "base_model": model_id_20b,
                    "general_dataset": {
                        "dataset_id": dataset_id,
                        "dataset_split": dataset_split,
                        "text_column": text_column,
                    },
                    "math_dataset": {
                        "dataset_id": math_dataset_id,
                        "dataset_split": math_dataset_split,
                        "text_column": math_text_column,
                    },
                    "keep_frac_general": 0.50,
                    "keep_frac_math": 0.25,
                    "eaftreap": {
                        "profile_top_k": int(general_prof["meta"]["top_k"]),
                        "entropy_topk": int(general_prof["meta"].get("entropy_topk") or 0),
                        "cc_quantile": float(eaft_cc_quantile),
                        "uncertain_quantile": float(eaft_uncertain_quantile),
                        "weights": {
                            "good": float(eaft_w_good),
                            "uncertain": float(eaft_w_uncertain),
                            "conflict": float(eaft_w_conflict),
                        },
                    },
                    "general_profile": {
                        "domain": "",
                        "prompt_hash": general_prof["meta"]["prompt_hash"],
                        "p_lo": float(general_prof["meta"]["p_lo"]),
                        "h_lo": float(general_prof["meta"]["h_lo"]),
                        "h_hi": float(general_prof["meta"]["h_hi"]),
                        "region_counts": general_prof["meta"].get("region_counts") or {},
                    },
                    "math_profile": {
                        "domain": "",
                        "prompt_hash": math_prof["meta"]["prompt_hash"],
                        "p_lo": float(math_prof["meta"]["p_lo"]),
                        "h_lo": float(math_prof["meta"]["h_lo"]),
                        "h_hi": float(math_prof["meta"]["h_hi"]),
                        "region_counts": math_prof["meta"].get("region_counts") or {},
                    },
                    "variants": {
                        "general_50pct_experts_eaftreap": general_dir,
                        "math_25pct_experts_eaftreap": math_dir,
                    },
                    "sanity": {
                        "general_ok": bool(general_ok.get("ok")),
                        "math_ok": bool(math_ok.get("ok")),
                    },
                    "keep_experts_by_layer": {
                        "general": general_keep,
                        "math": math_keep,
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        out_report = Path("reports/20b_structural_prune_build_eaftreap.md")
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(
            "\n".join(
                [
                    "# 20B structural prune build (EAFT-REAP ranking)",
                    "",
                    f"- Base model: `{model_id_20b}`",
                    f"- General dataset: `{dataset_id}` split `{dataset_split}` col `{text_column}`",
                    f"- Math dataset: `{math_dataset_id}` split `{math_dataset_split}` col `{math_text_column}`",
                    f"- Profile rows: {int(num_rows)} | Max seq length: {int(max_seq_length)} | Batch size: {int(batch_size)}",
                    f"- EAFT: cc_q={float(eaft_cc_quantile)} uncertain_q={float(eaft_uncertain_quantile)} "
                    f"entropy_topk={int(eaft_entropy_topk)} weights(good/uncertain/conflict)="
                    f"{float(eaft_w_good)}/{float(eaft_w_uncertain)}/{float(eaft_w_conflict)}",
                    "",
                    "## Variants",
                    "",
                    f"- general_50pct_experts_eaftreap: `{general_dir}`",
                    f"- math_25pct_experts_eaftreap: `{math_dir}`",
                    "",
                    "## Sanity inference",
                    "",
                    f"- general ok={general_ok.get('ok')}",
                    f"- math ok={math_ok.get('ok')}",
                    "",
                    "## Artifacts",
                    "",
                    f"- `{artifacts_dir/'manifest_eaftreap.json'}`",
                    "",
                    "## Reproduce",
                    "",
                    "```bash",
                    "modal run modal/gpt_oss_pruning_track.py --task build_pruned_20b_eaftreap",
                    "```",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"[+] Wrote {artifacts_dir/'manifest_eaftreap.json'}")
        print(f"[+] Wrote {out_report}")
        return

    if task == "build_pruned_20b_eaftreap_keepfrac":
        # Calib-packs EAFT-REAP pruning sweep (keep_frac only; top_k unchanged).
        # This is the recommended regime to match our evaluation harness.
        keep_fracs: list[float] = []
        for s in _parse_csv(str(keep_fracs_csv or "")):
            try:
                keep_fracs.append(float(s))
            except Exception:
                continue
        if not keep_fracs:
            raise SystemExit("--keep-fracs-csv must contain at least one float like 0.75,0.60")

        pack_files = _parse_csv(str(calib_pack_files_csv or "")) or list(DEFAULT_CALIB_PACK_FILES)
        if not pack_files:
            raise SystemExit("No calib pack files specified.")

        # CPU predownload the base model + CPU sample the pack rows into JSONL.
        _ = _invoke(predownload_model, str(model_id_20b))
        sample = _invoke(
            sample_calib_packs_rows_cpu,
            dataset_repo=str(calib_packs_repo),
            pack_files=list(pack_files),
            text_column=str(text_column),
            num_rows=int(num_rows),
            seed=int(seed),
            strategy=str(calib_pack_sample_strategy),
        )

        prof = _invoke(
            profile_20b_eaftreap_saliency,
            model_id=str(model_id_20b),
            dataset_id=str(calib_packs_repo),
            dataset_split="__calib_packs__",  # informational only
            text_column=str(text_column),
            domain="",
            num_rows=int(num_rows),
            max_seq_length=int(max_seq_length),
            batch_size=int(batch_size),
            cc_quantile=float(eaft_cc_quantile),
            uncertain_quantile=float(eaft_uncertain_quantile),
            entropy_topk=int(eaft_entropy_topk),
            w_good=float(eaft_w_good),
            w_uncertain=float(eaft_w_uncertain),
            w_conflict=float(eaft_w_conflict),
            rows_jsonl_path=str(sample["rows_jsonl_path"]),
        )

        ranking = prof["ranking_by_layer"]
        meta = prof.get("meta") or {}
        num_experts = int(meta.get("num_experts") or 0)
        num_layers = int(meta.get("num_layers") or 0)
        if num_experts <= 0 or num_layers <= 0:
            raise SystemExit(f"Invalid EAFT-REAP meta: num_layers={num_layers} num_experts={num_experts}")

        # Save profiling artifacts for audit/debug (no finetune).
        artifacts_dir = Path("artifacts/20b_pruned_models_eaftreap_keepfrac")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "eaftreap_saliency.parquet").write_bytes(prof["parquet_bytes"])
        (artifacts_dir / "eaftreap_saliency_ranking_by_layer.json").write_text(
            json.dumps(ranking, indent=2), encoding="utf-8"
        )

        import math as _math

        round_mode = str(keep_n_round or "ceil").strip().lower()
        if round_mode not in ("ceil", "floor", "round"):
            raise SystemExit("--keep-n-round must be one of: ceil,floor,round")
        mult = int(keep_n_multiple_of)
        if mult < 1:
            raise SystemExit("--keep-n-multiple-of must be >= 1")

        variants: dict[str, str] = {}
        keep_by_variant: dict[str, Any] = {}
        for keep_frac in keep_fracs:
            if not (0.0 < float(keep_frac) <= 1.0):
                raise SystemExit(f"Invalid keep_frac={keep_frac} (must be 0<k<=1)")
            raw = float(keep_frac) * float(num_experts)
            if round_mode == "ceil":
                keep_n = int(_math.ceil(raw))
            elif round_mode == "floor":
                keep_n = int(_math.floor(raw))
            else:
                keep_n = int(round(raw))
            keep_n = max(1, min(int(keep_n), int(num_experts)))
            # SGLang FlashInfer MXFP4 fused MoE routing expects num_experts % 4 == 0.
            # Enforce a multiple-of constraint so pruned models can be evaluated/served
            # with those kernels.
            if mult > 1:
                keep_n = int(_math.ceil(float(keep_n) / float(mult)) * int(mult))
                if keep_n > int(num_experts):
                    keep_n = int(_math.floor(float(num_experts) / float(mult)) * int(mult))
                keep_n = max(int(mult), keep_n)
                keep_n = min(int(num_experts), keep_n)
            keep_by_layer = []
            for li in range(num_layers):
                layer_rank = ranking[li]
                keep_by_layer.append([int(x) for x in layer_rank[:keep_n]])

            pct = int(round(float(keep_n) * 100.0 / float(num_experts)))
            variant_name = f"calib_union_keep{keep_n}of{num_experts}_k{pct}_eaftreap"
            out_dir = _invoke(
                structural_prune_20b_build,
                model_id=str(model_id_20b),
                variant_name=str(variant_name),
                keep_experts_by_layer_json=json.dumps(keep_by_layer),
                out_subdir="20b_pruned_models_eaftreap",
            )
            variants[str(variant_name)] = str(out_dir)
            keep_by_variant[str(variant_name)] = {
                "keep_frac_requested": float(keep_frac),
                "keep_frac_actual": float(keep_n) / float(num_experts),
                "keep_n": int(keep_n),
                "keep_n_multiple_of": int(mult),
            }

        manifest = {
            "base_model": str(model_id_20b),
            "calib_packs": {
                "dataset_repo": str(calib_packs_repo),
                "pack_files": pack_files,
                "sample": sample,
            },
            "eaft": {
                "cc_q": float(eaft_cc_quantile),
                "uncertain_q": float(eaft_uncertain_quantile),
                "entropy_topk": int(eaft_entropy_topk),
                "w_good": float(eaft_w_good),
                "w_uncertain": float(eaft_w_uncertain),
                "w_conflict": float(eaft_w_conflict),
            },
            "profile": {
                "meta": meta,
            },
            "variants": variants,
            "keep": keep_by_variant,
            "keep_fracs_csv": str(keep_fracs_csv),
            "keep_n_round": str(round_mode),
            "keep_n_multiple_of": int(mult),
        }
        (artifacts_dir / "manifest_eaftreap_keepfrac.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        rep = Path("reports/20b_structural_prune_build_eaftreap_keepfrac.md")
        rep.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# 20B structural prune build (EAFT-REAP, calib packs, keep_frac sweep)",
            "",
            f"- Base model: `{model_id_20b}`",
            f"- Calib repo: `{calib_packs_repo}`",
            f"- Packs: {', '.join(pack_files)}",
            f"- Sample rows: {int(num_rows)} seed={int(seed)} sample_jsonl=`{sample.get('rows_jsonl_path','')}`",
            f"- Max seq length: {int(max_seq_length)} | Batch size: {int(batch_size)}",
            f"- EAFT weights: good={float(eaft_w_good)} uncertain={float(eaft_w_uncertain)} conflict={float(eaft_w_conflict)}",
            "",
            "## Variants",
            "",
        ]
        for name, out_dir in variants.items():
            k = keep_by_variant.get(name, {})
            lines.append(f"- {name}: keep_frac={k.get('keep_frac')} keep_n={k.get('keep_n')} dir=`{out_dir}`")
        lines += [
            "",
            "## Artifacts",
            "",
            f"- `{artifacts_dir/'manifest_eaftreap_keepfrac.json'}`",
            "",
            "## Reproduce (Kaggle/VERSA)",
            "",
            "```bash",
            "bash harmony/cuda-norm/scripts/versa_run_pruning_track_kaggle.sh \\",
            "  --task build_pruned_20b_eaftreap_keepfrac \\",
            f"  --model-id-20b {model_id_20b} \\",
            f"  --num-rows {int(num_rows)} --max-seq-length {int(max_seq_length)} --batch-size {int(batch_size)} \\",
            f"  --keep-fracs-csv {keep_fracs_csv!s}",
            "```",
            "",
        ]
        rep.write_text("\n".join(lines), encoding="utf-8")
        print(f"[+] Wrote {artifacts_dir/'manifest_eaftreap_keepfrac.json'}")
        print(f"[+] Wrote {rep}")
        return

    if task == "build_pruned_20b_noop_rewrite":
        # Structural rewrite while keeping all experts. This must be near-identical
        # to the base model; otherwise our rewrite/index/mapping path is corrupt.
        meta = _invoke(read_model_cfg_meta, str(model_id_20b))
        num_layers = int(meta.get("num_layers") or 0)
        num_experts = int(meta.get("num_experts") or 0)
        if num_layers <= 0 or num_experts <= 0:
            raise SystemExit(f"Invalid config: num_layers={num_layers} num_experts={num_experts}")
        keep_by_layer = [list(range(num_experts)) for _ in range(num_layers)]
        out_dir = _invoke(
            structural_prune_20b_build,
            model_id=str(model_id_20b),
            variant_name="noop_rewrite_keepall_experts",
            keep_experts_by_layer_json=json.dumps(keep_by_layer),
            out_subdir="20b_pruned_models_noop",
        )
        artifacts_dir = Path("artifacts/20b_pruned_models_noop")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "base_model": str(model_id_20b),
                    "variant_name": "noop_rewrite_keepall_experts",
                    "keep_n": int(num_experts),
                    "num_layers": int(num_layers),
                    "num_experts": int(num_experts),
                    "out_dir": str(out_dir),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        rep = Path("reports/20b_noop_rewrite_build.md")
        rep.parent.mkdir(parents=True, exist_ok=True)
        rep.write_text(
            "\n".join(
                [
                    "# 20B noop rewrite (keep all experts)",
                    "",
                    f"- Base model: `{model_id_20b}`",
                    f"- Output dir: `{out_dir}`",
                    f"- Manifest: `{artifacts_dir/'manifest.json'}`",
                    "",
                    "Next: run EAFT/PPL parity base vs this dir; deltas should be ~0.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"[+] Wrote {rep}")
        return

    if task == "build_pruned_20b_eaftreap_budgeted":
        # Build a single keep_frac prune (one-shot, no finetune).
        #
        # IMPORTANT: GPT-OSS requires a single `num_local_experts` across all layers.
        # So even if we compute per-layer sensitivity metrics, the pruned model must
        # keep the SAME number of experts in every layer.
        pack_files = _parse_csv(str(calib_pack_files_csv or "")) or list(DEFAULT_CALIB_PACK_FILES)
        if not pack_files:
            raise SystemExit("No calib pack files specified.")

        _ = _invoke(predownload_model, str(model_id_20b))
        pack_weights: list[float] | None = None
        if str(calib_pack_weights_csv or "").strip():
            pack_weights = []
            for s in _parse_csv(str(calib_pack_weights_csv)):
                try:
                    pack_weights.append(float(s))
                except Exception:
                    pack_weights.append(0.0)
        sample = _invoke(
            sample_calib_packs_rows_cpu,
            dataset_repo=str(calib_packs_repo),
            pack_files=list(pack_files),
            text_column=str(text_column),
            num_rows=int(num_rows),
            seed=int(seed),
            strategy=str(calib_pack_sample_strategy),
            pack_weights=pack_weights,
        )

        prof = _invoke(
            profile_20b_eaftreap_saliency,
            model_id=str(model_id_20b),
            dataset_id=str(calib_packs_repo),
            dataset_split="__calib_packs__",
            text_column=str(text_column),
            domain="",
            num_rows=int(num_rows),
            max_seq_length=int(max_seq_length),
            batch_size=int(batch_size),
            cc_quantile=float(eaft_cc_quantile),
            uncertain_quantile=float(eaft_uncertain_quantile),
            entropy_topk=int(eaft_entropy_topk),
            w_good=float(eaft_w_good),
            w_uncertain=float(eaft_w_uncertain),
            w_conflict=float(eaft_w_conflict),
            rows_jsonl_path=str(sample["rows_jsonl_path"]),
        )

        ranking = prof["ranking_by_layer"]
        meta = prof.get("meta") or {}
        num_experts = int(meta.get("num_experts") or 0)
        num_layers = int(meta.get("num_layers") or 0)
        if num_experts <= 0 or num_layers <= 0:
            raise SystemExit(f"Invalid EAFT-REAP meta: num_layers={num_layers} num_experts={num_experts}")

        import math as _math

        keep_n = int(_math.ceil(float(keep_frac) * float(num_experts)))
        keep_n = max(1, min(int(keep_n), int(num_experts)))

        # Checkpoint profiling outputs early so a long run is not "lost" if the
        # structural rewrite step fails later.
        artifacts_dir = Path("artifacts/20b_pruned_models_eaftreap_budgeted")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "eaftreap_saliency.parquet").write_bytes(prof["parquet_bytes"])
        (artifacts_dir / "eaftreap_ranking_by_layer.json").write_text(
            json.dumps(ranking, indent=2), encoding="utf-8"
        )
        (artifacts_dir / "profile_meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
        )
        (artifacts_dir / "sample.json").write_text(
            json.dumps(sample, indent=2, sort_keys=True), encoding="utf-8"
        )
        (artifacts_dir / "keep_n.json").write_text(
            json.dumps({"keep_n": keep_n, "num_experts": num_experts}, indent=2),
            encoding="utf-8",
        )

        # Read per-layer mass from parquet we just produced.
        parquet_path = Path("/tmp") / f"eaftreap_saliency_{int(time.time())}.parquet"
        parquet_path.write_bytes(prof["parquet_bytes"])
        by_layer_mass = _read_reap_saliency_mass_parquet(parquet_path, mass_column="eaft_gate_norm_sum")
        mass_by_layer = []
        for li in range(num_layers):
            mass_by_layer.append(by_layer_mass.get(li, [0.0] * num_experts))

        core_by_layer = _core_experts_from_eaft_parquet(
            parquet_path,
            num_layers=num_layers,
            num_experts=num_experts,
            pos_top_m=int(core_pos_top_m),
            count_top_m=int(core_count_top_m),
        )
        neff_by_layer = [float(_compute_layer_neff(m)) for m in mass_by_layer]
        keep_by_layer = []
        for li in range(num_layers):
            core = [int(x) for x in (core_by_layer[li] if li < len(core_by_layer) else [])]
            chosen: list[int] = []
            seen: set[int] = set()

            # Safety core MUST be included first (otherwise it is a no-op).
            for e in core:
                ei = int(e)
                if 0 <= ei < num_experts and ei not in seen:
                    chosen.append(ei)
                    seen.add(ei)
                if len(chosen) >= keep_n:
                    break

            # Fill remaining slots by EAFT-REAP ranking.
            if len(chosen) < keep_n:
                for e in ranking[li]:
                    ei = int(e)
                    if 0 <= ei < num_experts and ei not in seen:
                        chosen.append(ei)
                        seen.add(ei)
                    if len(chosen) >= keep_n:
                        break

            if len(chosen) != keep_n:
                raise RuntimeError(
                    f"Failed to choose keep_n={keep_n} experts for layer {li}: got {len(chosen)}"
                )
            keep_by_layer.append(chosen)

        variant_name = (
            f"calib_budget_keepfrac{float(keep_frac):.2f}_eaftreap"
            f"_corep{int(core_pos_top_m)}c{int(core_count_top_m)}"
        )
        out_dir = _invoke(
            structural_prune_20b_build,
            model_id=str(model_id_20b),
            variant_name=str(variant_name),
            keep_experts_by_layer_json=json.dumps(keep_by_layer),
            out_subdir="20b_pruned_models_eaftreap",
        )

        (artifacts_dir / "neff_by_layer.json").write_text(
            json.dumps(neff_by_layer, indent=2), encoding="utf-8"
        )
        (artifacts_dir / "core_experts_by_layer.json").write_text(
            json.dumps(core_by_layer, indent=2), encoding="utf-8"
        )
        (artifacts_dir / "keep_experts_by_layer.json").write_text(
            json.dumps(keep_by_layer, indent=2), encoding="utf-8"
        )
        (artifacts_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "base_model": str(model_id_20b),
                    "variant_name": str(variant_name),
                    "out_dir": str(out_dir),
                    "keep_frac": float(keep_frac),
                    "keep_n": int(keep_n),
                    "neff_by_layer": neff_by_layer,
                    "core_pos_top_m": int(core_pos_top_m),
                    "core_count_top_m": int(core_count_top_m),
                    "calib": {"repo": str(calib_packs_repo), "pack_files": pack_files, "sample": sample},
                    "eaft": {
                        "cc_q": float(eaft_cc_quantile),
                        "uncertain_q": float(eaft_uncertain_quantile),
                        "entropy_topk": int(eaft_entropy_topk),
                        "w_good": float(eaft_w_good),
                        "w_uncertain": float(eaft_w_uncertain),
                        "w_conflict": float(eaft_w_conflict),
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        rep = Path("reports/20b_structural_prune_build_eaftreap_budgeted.md")
        rep.parent.mkdir(parents=True, exist_ok=True)
        rep.write_text(
            "\n".join(
                [
                    "# 20B structural prune build (EAFT-REAP budgeted keep counts)",
                    "",
                    f"- Base model: `{model_id_20b}`",
                    f"- Output dir: `{out_dir}`",
                    f"- keep_frac: {float(keep_frac):.2f}",
                    f"- safety core: pos_top_m={int(core_pos_top_m)} count_top_m={int(core_count_top_m)}",
                    f"- keep_n (uniform across layers): {int(keep_n)}/{int(num_experts)}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"[+] Wrote {rep}")
        return

    if task == "inspect_pruned_20b":
        # Debug helper: inspect a key's mapped file + tensor shape.
        model_dir = str(_ARTIFACTS_DIR / "20b_pruned_models" / "general_50pct_experts")
        infos = []
        for li in (0, 1, 10, 23):
            infos.append(
                inspect_pruned_checkpoint.remote(
                    model_dir=model_dir,
                    layer_idx=int(li),
                )
            )
        infos.append(
            validate_pruned_expert_shards.remote(
                model_dir=model_dir
            )
        )
        print(json.dumps(infos, indent=2))
        return

    if task == "build_pruned_120b_first64":
        # CPU-only structural prune build for GPT-OSS-120B with a deterministic baseline:
        # keep experts [0..63] for every layer. This validates the full rewrite/indexing path
        # without needing any 120B saliency ranking yet.
        #
        # NOTE: We do not run sanity_infer_model_dir here because 120B cannot be loaded on H100:1.
        model_id_120b = DEFAULT_120B_MODEL_ID

        # Read num_layers/num_experts remotely (local entrypoint should never touch /root/*).
        meta = read_model_cfg_meta.remote(str(model_id_120b))
        num_layers = int(meta.get("num_layers") or 0)
        num_experts = int(meta.get("num_experts") or 0)
        if num_layers <= 0 or num_experts <= 0:
            raise SystemExit(f"Invalid config for {model_id_120b}: num_layers={num_layers} num_experts={num_experts}")

        keep_n = 64
        if keep_n > num_experts:
            raise SystemExit(f"keep_n={keep_n} exceeds num_experts={num_experts} for {model_id_120b}")
        keep = list(range(keep_n))
        keep_by_layer = [keep for _ in range(num_layers)]

        variant_name = f"first{keep_n}_experts_keepfrac50"
        out_dir = structural_prune_20b_build.remote(
            model_id=model_id_120b,
            variant_name=variant_name,
            keep_experts_by_layer_json=json.dumps(keep_by_layer),
            out_subdir="120b_pruned_models",
        )

        # Validate a few keys for correct mapping + tensor shape.
        infos = []
        for li in (0, 1, 9, 18, 27, 35):
            infos.append(inspect_pruned_checkpoint.remote(model_dir=out_dir, layer_idx=int(li)))
        infos.append(validate_pruned_expert_shards.remote(model_dir=out_dir))

        artifacts_dir = Path("artifacts/120b_pruned_models_first64")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "base_model": model_id_120b,
                    "variant_name": variant_name,
                    "keep_n": keep_n,
                    "num_layers": num_layers,
                    "num_experts": num_experts,
                    "keep_experts_by_layer": keep_by_layer,
                    "model_dir": out_dir,
                    "inspect": infos,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        out_report = Path("reports/120b_structural_prune_build_first64.md")
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(
            "\n".join(
                [
                    "# 120B structural prune build (baseline: first64)",
                    "",
                    f"- Base model: `{model_id_120b}`",
                    f"- keep_n: {keep_n}/{num_experts} (all {num_layers} layers)",
                    "",
                    "## Output",
                    "",
                    f"- model_dir: `{out_dir}`",
                    "",
                    "## Validation (CPU)",
                    "",
                    f"- wrote: `{artifacts_dir/'manifest.json'}`",
                    "",
                    "## Reproduce",
                    "",
                    "```bash",
                    "modal run modal/gpt_oss_pruning_track.py --task build_pruned_120b_first64",
                    "```",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"[+] Wrote {artifacts_dir/'manifest.json'}")
        print(f"[+] Wrote {out_report}")
        print(json.dumps(infos, indent=2))
        return

    raise SystemExit(f"Unknown --task {task!r}")
