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


def _ensure_hf_env() -> None:
    os.environ.setdefault("HF_HOME", "/root/hf_cache")
    os.environ.setdefault("XDG_CACHE_HOME", "/root/hf_cache/.cache")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for p in (
        "/root/hf_cache",
        "/root/hf_cache/.cache",
        "/root/data",
        "/root/model",
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

    cache_dir = Path("/root/model/.hf_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Avoid re-downloading in GPU containers: if the model was already
    # predownloaded into the persistent volume, use it directly.
    local_dir = Path("/root/model") / str(model_id)
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
    local_dir = Path("/root/model") / str(model_id)
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


def _read_reap_saliency_mass_parquet(path: Path) -> dict[int, list[float]]:
    import pyarrow.parquet as pq

    table = pq.read_table(str(path), columns=["layer", "expert", "gate_norm_sum"])
    layers = table.column("layer").to_pylist()
    experts = table.column("expert").to_pylist()
    masses = table.column("gate_norm_sum").to_pylist()
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
    gpu="H100:1",
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

    out_dir = Path("/root/data/reap_domain_samples") / str(dataset_id).replace("/", "__") / str(dataset_split)
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
    gpu="H100:1",
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
    gpu="H100:1",
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
        c = count_cpu[li].to(torch.float32).clamp_min(1.0)
        mean = gate_norm_sum_weighted_cpu[li] / c
        order = torch.argsort(mean, descending=True).tolist()
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
    gpu="H100:1",
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
):
    from safetensors.torch import safe_open
    from safetensors.torch import save_file
    from huggingface_hub import hf_hub_download

    _ensure_hf_env()

    keep_experts_by_layer = json.loads(keep_experts_by_layer_json)
    if not isinstance(keep_experts_by_layer, list) or not keep_experts_by_layer:
        raise ValueError("keep_experts_by_layer_json must be a non-empty JSON list.")

    t0 = time.time()

    # NOTE: We previously symlinked snapshot files into the pruned output dir and
    # then overwrote `config.json` / `model.safetensors.index.json`, which can
    # mutate the underlying HF snapshot via symlink-following writes.
    # Force-refresh these small metadata files before we proceed.
    token = _get_hf_token()
    cache_dir = Path("/root/model/.hf_cache")
    for fname in ("config.json", "model.safetensors.index.json"):
        hf_hub_download(
            repo_id=str(model_id),
            repo_type="model",
            filename=fname,
            cache_dir=str(cache_dir),
            token=token,
            force_download=True,
        )

    snapshot_dir = _snapshot_download_model(model_id)
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

    out_dir = Path("/root/model/artifacts") / str(out_subdir) / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove previously-generated shards if re-running in the same volume path.
    for p in out_dir.glob("*.safetensors"):
        if p.name.startswith(("base_", "pruned_layer_", "model-")):
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

    # Build base shards (one per original shard file) containing only the
    # remaining (non-pruned) keys.
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

    for src_file, keys in remaining_by_file.items():
        src_path = snapshot_dir / src_file
        if not src_path.exists():
            raise RuntimeError(f"Missing shard file {src_path}")
        dst_name = f"base_{src_file}"
        dst_path = out_dir / dst_name

        tensors: dict[str, Any] = {}
        with safe_open(str(src_path), framework="pt", device="cpu") as f:
            for key in keys:
                tensors[key] = f.get_tensor(key)
        save_file(tensors, str(dst_path), metadata={"format": "pt", "harmony": "base"})
        for key in keys:
            new_weight_map[key] = dst_name
        print(f"[*] wrote {dst_name} keys={len(keys)}", flush=True)

    for li in range(num_layers):
        keep = [int(x) for x in keep_experts_by_layer[li]]
        mapping_json[str(li)] = {str(old): int(new) for new, old in enumerate(keep)}

        shard_name = f"pruned_layer_{li}.safetensors"
        shard_path = out_dir / shard_name

        tensors: dict[str, Any] = {}
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
            tensors[key] = t[keep].contiguous()
            new_weight_map[key] = shard_name

        save_file(tensors, str(shard_path), metadata={"format": "pt", "harmony": "pruned"})
        print(f"[*] wrote {shard_name} keys={len(tensors)}", flush=True)

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
    gpu="H100:1",
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
    for li in range(num_layers):
        expected = f"pruned_layer_{li}.safetensors"
        prefix = f"model.layers.{li}.mlp."
        for k, v in wm.items():
            if not k.startswith(prefix):
                continue
            if ".router." not in k and ".experts." not in k:
                continue
            if v != expected:
                mismatches.append({"layer": li, "key": k, "file": v, "expected": expected})
                if len(mismatches) >= 50:
                    break
        if len(mismatches) >= 50:
            break

    return {
        "model_dir": str(model_path),
        "config_num_local_experts": int(cfg.get("num_local_experts") or 0),
        "config_num_layers": int(num_layers),
        "mismatch_count_capped": len(mismatches),
        "mismatches_sample": mismatches,
    }


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
    - scan_domain_values_20b: scan `DOMAIN_COLUMN` and report counts (debug data availability)
    """

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
        ok50 = sanity_infer_model_dir.remote(model_dir=union50_dir)
        okagg = sanity_infer_model_dir.remote(model_dir=unionagg_dir)

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
        _ = predownload_model.remote(str(model_id_20b))

        res = profile_20b_reap_saliency.remote(
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
        _ = predownload_model.remote(str(model_id_20b))

        res = profile_20b_eaftreap_saliency.remote(
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
        _ = predownload_model.remote(str(model_id_20b))

        general_prof = profile_20b_eaftreap_saliency.remote(
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
        math_prof = profile_20b_eaftreap_saliency.remote(
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

        general_dir = structural_prune_20b_build.remote(
            model_id=model_id_20b,
            variant_name="general_50pct_experts_eaftreap",
            keep_experts_by_layer_json=json.dumps(general_keep),
            out_subdir="20b_pruned_models_eaftreap",
        )
        math_dir = structural_prune_20b_build.remote(
            model_id=model_id_20b,
            variant_name="math_25pct_experts_eaftreap",
            keep_experts_by_layer_json=json.dumps(math_keep),
            out_subdir="20b_pruned_models_eaftreap",
        )

        general_ok = sanity_infer_model_dir.remote(model_dir=general_dir)
        math_ok = sanity_infer_model_dir.remote(model_dir=math_dir)

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

    if task == "inspect_pruned_20b":
        # Debug helper: inspect a key's mapped file + tensor shape.
        infos = []
        for li in (0, 1, 10, 23):
            infos.append(
                inspect_pruned_checkpoint.remote(
                    model_dir="/root/model/artifacts/20b_pruned_models/general_50pct_experts",
                    layer_idx=int(li),
                )
            )
        infos.append(
            validate_pruned_expert_shards.remote(
                model_dir="/root/model/artifacts/20b_pruned_models/general_50pct_experts"
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
