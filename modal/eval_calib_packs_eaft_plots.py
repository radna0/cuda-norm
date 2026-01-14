"""
EAFT-style visual diagnostics (plots) for pruning degradation on curated calib packs.

This generates paper-style "entropy–probability landscapes" for:
  - base model (e.g., openai/gpt-oss-20b)
  - pruned model (e.g., sandeshrajx/gpt-oss-20b-reap-0.5-mxfp4)

For each pack (and UNION mix) and each seq_len (1024, 2048), it computes on
completion-only tokens (Harmony packed blocks):
  - PPL (completion-only)
  - p_t = P(reference token)
  - H_t = normalized Top-K entropy (K=entropy_topk, default 20)
  - Confident Conflict rate (CC): bottom q in BOTH p_t and H_t (default q=0.15)
  - 2D histogram of (p_t, H_t) or (log10(p_t), H_t) for base vs pruned
  - 1D histograms for p_t (or log10(p_t)) and H_t

Outputs (local files):
  - reports/20b_calib_packs_eaft_plots.html   (self-contained dashboard)
  - artifacts/eaft_plots/<run_id>/eaft_data.json
  - artifacts/eaft_plots/<run_id>/20b_calib_packs_eaft_plots.html

Run (always log to unsloth_logs/):
  mkdir -p unsloth_logs
  ts=$(date +%Y%m%d_%H%M%S)
  nohup env MODAL_PROFILE=phamtrinhkien1203 modal run modal/eval_calib_packs_eaft_plots.py \
    --base-model-id openai/gpt-oss-20b \
    --pruned-model-id sandeshrajx/gpt-oss-20b-reap-0.5-mxfp4 \
    --top-k 4 --entropy-topk 20 --cc-quantile 0.15 \
    --num-blocks 32 --batch-size 1 \
    --prob-scale linear --hist-xbins 160 --hist-ybins 120 --logp-min -12 \
    > "unsloth_logs/calib_packs_eaft_plots_${ts}.log" 2>&1 &
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

APP_NAME = "eval-calib-packs-eaft-plots"


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

DEFAULT_DATASET_REPO = os.environ.get("CALIB_PACKS_DATASET", "radna0/harmony-qwen3-calib-packs-v2-20260113")
DEFAULT_PACK_FILES = [
    "packs/reasoning_style_10k_v2/reasoning_style_10k_v2.parquet",
    "tool_agentic_10k_v6.parquet",
    "packs/calib_prompt_10000_v2/calib_prompt_10000_v2.parquet",
]

DEFAULT_BASE_MODEL_ID = os.environ.get("MODEL_ID_20B", "openai/gpt-oss-20b")
DEFAULT_PRUNED_MODEL_ID = os.environ.get("PRUNED_MODEL_ID", "sandeshrajx/gpt-oss-20b-reap-0.5-mxfp4")

_secrets: list[modal.Secret] = []
if os.environ.get("HF_TOKEN"):
    _secrets.append(modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}))

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
        "numpy==2.2.0 pyarrow==22.0.0 "
        "accelerate==1.10.1 "
        "transformers==4.56.2 tokenizers safetensors "
        "kernels==0.11.7 "
        "hf_transfer huggingface-hub==0.34.0"
    )
)

app = modal.App(APP_NAME)


def _ensure_hf_env() -> None:
    os.environ.setdefault("HF_HOME", "/root/hf_cache")
    os.environ.setdefault("XDG_CACHE_HOME", "/root/hf_cache/.cache")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for p in ("/root/hf_cache", "/root/hf_cache/.cache", "/root/data", "/root/model"):
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
    token = _get_hf_token()
    cache_dir = Path("/root/model/.hf_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
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


def _download_dataset_file(dataset_repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    _ensure_hf_env()
    token = _get_hf_token()
    return Path(
        hf_hub_download(
            repo_id=str(dataset_repo),
            repo_type="dataset",
            filename=str(filename),
            token=token,
        )
    )


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
        while len(buf_ids) >= (seq_len + 1) and len(blocks_ids) < num_blocks:
            block_i = buf_ids[: seq_len + 1]
            block_k = buf_keep[: seq_len + 1]
            block_k[0] = False
            blocks_ids.append(block_i)
            blocks_keep.append(block_k)
            del buf_ids[: seq_len + 1]
            del buf_keep[: seq_len + 1]
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
    model,
    *,
    blocks: PackedBlocks,
    batch_size: int,
    entropy_topk: int,
    cc_quantile: float,
    base_thresholds: tuple[float, float] | None,
    hist_xbins: int,
    hist_ybins: int,
    prob_scale: str,
    logp_min: float,
    logp_max: float,
) -> dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn.functional as F

    entropy_topk = int(entropy_topk)
    cc_quantile = float(cc_quantile)
    if not (0.0 < cc_quantile < 1.0):
        raise ValueError("cc_quantile must be in (0,1)")

    probs_cpu: list[torch.Tensor] = []
    ent_cpu: list[torch.Tensor] = []

    total_nll = 0.0
    total_nll_sumsq = 0.0
    kept_tokens = 0
    total_pred_tokens = 0
    t0 = time.time()
    for start in range(0, len(blocks.blocks_ids), batch_size):
        batch_ids = blocks.blocks_ids[start : start + batch_size]
        batch_keep = blocks.blocks_keep[start : start + batch_size]
        input_ids = torch.tensor(batch_ids, device="cuda", dtype=torch.long)
        keep_mask = torch.tensor(batch_keep, device="cuda", dtype=torch.bool)
        with torch.inference_mode():
            logits = model(input_ids[:, :-1], use_cache=False).logits

        targets = input_ids[:, 1:]
        keep_t = keep_mask[:, 1:]
        bsz, seql = targets.shape
        total_pred_tokens += int(bsz * seql)
        vocab = int(logits.shape[-1])

        nll = F.cross_entropy(
            logits.reshape(-1, vocab),
            targets.reshape(-1),
            reduction="none",
        ).view(bsz, seql)
        prob = torch.exp(-nll)

        k = min(max(1, int(entropy_topk)), vocab)
        topk_logits = torch.topk(logits, k=k, dim=-1).values  # (bsz, seql, k)
        topk_p = torch.softmax(topk_logits.float(), dim=-1)
        ent = -(topk_p * torch.log(topk_p.clamp_min(1e-12))).sum(dim=-1)
        ent_norm = ent / float(max(1e-12, math.log(float(k))))

        kept_nll = nll[keep_t]
        total_nll += float(kept_nll.sum().item())
        total_nll_sumsq += float((kept_nll * kept_nll).sum().item())
        kept_tokens += int(keep_t.sum().item())

        probs_cpu.append(prob[keep_t].detach().float().cpu())
        ent_cpu.append(ent_norm[keep_t].detach().float().cpu())

    torch.cuda.synchronize()
    dt = max(1e-9, time.time() - t0)

    p_all_t = torch.cat(probs_cpu) if probs_cpu else torch.empty((0,), dtype=torch.float32)
    h_all_t = torch.cat(ent_cpu) if ent_cpu else torch.empty((0,), dtype=torch.float32)
    if p_all_t.numel() != h_all_t.numel():
        raise RuntimeError("prob/entropy arrays length mismatch")
    if p_all_t.numel() == 0:
        raise RuntimeError("No kept tokens; cannot compute EAFT diagnostics.")

    p_thr = float(torch.quantile(p_all_t, q=float(cc_quantile)).item())
    h_thr = float(torch.quantile(h_all_t, q=float(cc_quantile)).item())
    cc_mask = (p_all_t <= p_thr) & (h_all_t <= h_thr)
    cc_rate = float(cc_mask.float().mean().item())
    cc_count = int(cc_mask.sum().item())

    cc_rate_base = None
    cc_count_base = None
    if base_thresholds is not None:
        bp, bh = base_thresholds
        base_mask = (p_all_t <= float(bp)) & (h_all_t <= float(bh))
        cc_rate_base = float(base_mask.float().mean().item())
        cc_count_base = int(base_mask.sum().item())

    mean_nll = total_nll / max(1, kept_tokens)
    var_nll = max(0.0, (total_nll_sumsq / max(1, kept_tokens)) - (mean_nll * mean_nll))
    ppl = math.exp(mean_nll)

    p_all = p_all_t.numpy().astype(np.float32, copy=False)
    h_all = h_all_t.numpy().astype(np.float32, copy=False)
    h_all = np.clip(h_all, 0.0, 1.0)
    logp = np.log10(np.clip(p_all, 1e-12, 1.0)).astype(np.float32, copy=False)

    prob_scale = str(prob_scale).lower().strip()
    if prob_scale not in ("linear", "log"):
        raise ValueError("prob_scale must be 'linear' or 'log'")
    if prob_scale == "linear":
        x_vals = p_all
        x_min = 0.0
        x_max = 1.0
        x_thr = float(p_thr)
    else:
        x_vals = logp
        x_min = float(logp_min)
        x_max = float(logp_max)
        x_thr = float(math.log10(max(1e-12, float(p_thr))))

    hist2d = _hist2d(
        x_vals,
        h_all,
        xbins=int(hist_xbins),
        ybins=int(hist_ybins),
        xmin=float(x_min),
        xmax=float(x_max),
        ymin=0.0,
        ymax=1.0,
    )
    hist_x = _hist1d(x_vals, bins=int(hist_xbins), vmin=float(x_min), vmax=float(x_max))
    hist_h = _hist1d(h_all, bins=int(hist_ybins), vmin=0.0, vmax=1.0)

    # Quadrant masses under this model's thresholds (low/high p, low/high H).
    p_le = p_all_t <= float(p_thr)
    h_le = h_all_t <= float(h_thr)
    ll = float((p_le & h_le).float().mean().item())
    lh = float((p_le & (~h_le)).float().mean().item())
    hl = float(((~p_le) & h_le).float().mean().item())
    hh = max(0.0, 1.0 - ll - lh - hl)

    q05 = float(torch.quantile(p_all_t, q=0.05).item())
    q50 = float(torch.quantile(p_all_t, q=0.50).item())
    q95 = float(torch.quantile(p_all_t, q=0.95).item())
    h05 = float(torch.quantile(h_all_t, q=0.05).item())
    h50 = float(torch.quantile(h_all_t, q=0.50).item())
    h95 = float(torch.quantile(h_all_t, q=0.95).item())

    return {
        "ppl": float(ppl),
        "kept_tokens": int(kept_tokens),
        "tok_s_pred": float(total_pred_tokens / dt),
        "mean_prob": float(p_all_t.mean().item()),
        "mean_entropy": float(h_all_t.mean().item()),
        "mean_nll": float(mean_nll),
        "var_nll": float(var_nll),
        "p_thr": float(p_thr),
        "h_thr": float(h_thr),
        "logp_thr": float(math.log10(max(1e-12, float(p_thr)))),
        "x_thr": float(x_thr),
        "p_q05": q05,
        "p_q50": q50,
        "p_q95": q95,
        "h_q05": h05,
        "h_q50": h50,
        "h_q95": h95,
        "cc_rate": float(cc_rate),
        "cc_rate_base_thr": cc_rate_base,
        "cc_count": int(cc_count),
        "cc_count_base_thr": cc_count_base,
        "quadrants": {"LL": ll, "LH": lh, "HL": hl, "HH": hh},
        "hist2d_x_H": hist2d,
        "hist1d_x": hist_x,
        "hist1d_H": hist_h,
        "x_scale": prob_scale,
        "x_min": float(x_min),
        "x_max": float(x_max),
    }


def _js_divergence(counts_a: list[int], counts_b: list[int]) -> float:
    import numpy as np

    a = np.asarray(counts_a, dtype=np.float64)
    b = np.asarray(counts_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("JS divergence requires same shape")
    a_sum = float(a.sum())
    b_sum = float(b.sum())
    if a_sum <= 0.0 or b_sum <= 0.0:
        return 0.0
    p = a / a_sum
    q = b / b_sum
    m = 0.5 * (p + q)
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = np.clip(m, eps, 1.0)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return 0.5 * (kl_pm + kl_qm)


def _z_for_prop_delta(p1: float, n1: int, p2: float, n2: int) -> float:
    if n1 <= 0 or n2 <= 0:
        return 0.0
    p1 = float(p1)
    p2 = float(p2)
    pooled = (p1 * n1 + p2 * n2) / float(n1 + n2)
    var = pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2)
    if var <= 0.0:
        return 0.0
    return float((p2 - p1) / math.sqrt(var))


@app.function(
    image=image,
    gpu="H100:1",
    timeout=21600,
    cpu=16.0,
    memory=262144,
    volumes={"/root/data": data_volume, "/root/model": model_volume, "/root/hf_cache": hf_cache_volume},
    secrets=_secrets,
)
def eval_calib_packs_eaft_plots(
    *,
    dataset_repo: str,
    pack_files: list[str],
    base_model_id: str,
    pruned_model_id: str,
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
) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _ensure_hf_env()
    try:
        model_volume.reload()
        hf_cache_volume.reload()
        data_volume.reload()
    except Exception:
        pass

    base_dir = _snapshot_download_model(str(base_model_id))
    pruned_dir = _snapshot_download_model(str(pruned_model_id))

    tok = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
    if not getattr(tok, "is_fast", False):
        raise RuntimeError("Need a fast tokenizer for return_offsets_mapping=True")
    eos = tok.eos_token_id
    if eos is None:
        raise RuntimeError("Tokenizer missing eos_token_id")

    pack_paths: dict[str, Path] = {}
    for f in pack_files:
        pack_paths[str(f)] = _download_dataset_file(str(dataset_repo), str(f))

    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        torch_dtype="auto",
        device_map={"": 0},
        trust_remote_code=True,
    )
    pruned_model = AutoModelForCausalLM.from_pretrained(
        str(pruned_dir),
        torch_dtype="auto",
        device_map={"": 0},
        trust_remote_code=True,
    )
    base_model.eval()
    pruned_model.eval()
    _apply_top_k(base_model, int(top_k))
    _apply_top_k(pruned_model, int(top_k))

    def _eval_pack(name: str, text_it: Callable[[], Iterable[str]]) -> dict[str, Any]:
        out: dict[str, Any] = {"pack": name, "seq": {}}
        for seq_len in (1024, 2048):
            blocks = _pack_blocks(
                text_iter=text_it,
                tok=tok,
                eos_id=int(eos),
                seq_len=int(seq_len),
                num_blocks=int(num_blocks),
            )
            base_metrics = _eaft_collect_for_plots(
                base_model,
                blocks=blocks,
                batch_size=int(batch_size),
                entropy_topk=int(entropy_topk),
                cc_quantile=float(cc_quantile),
                base_thresholds=None,
                hist_xbins=int(hist_xbins),
                hist_ybins=int(hist_ybins),
                prob_scale=str(prob_scale),
                logp_min=float(logp_min),
                logp_max=float(logp_max),
            )
            base_thresholds = (float(base_metrics["p_thr"]), float(base_metrics["h_thr"]))
            pruned_metrics = _eaft_collect_for_plots(
                pruned_model,
                blocks=blocks,
                batch_size=int(batch_size),
                entropy_topk=int(entropy_topk),
                cc_quantile=float(cc_quantile),
                base_thresholds=base_thresholds,
                hist_xbins=int(hist_xbins),
                hist_ybins=int(hist_ybins),
                prob_scale=str(prob_scale),
                logp_min=float(logp_min),
                logp_max=float(logp_max),
            )
            # Distribution shift diagnostics (JS divergence on binned densities).
            js2d = _js_divergence(
                base_metrics["hist2d_x_H"]["counts"],
                pruned_metrics["hist2d_x_H"]["counts"],
            )
            jsx = _js_divergence(
                base_metrics["hist1d_x"]["counts"],
                pruned_metrics["hist1d_x"]["counts"],
            )
            jsh = _js_divergence(
                base_metrics["hist1d_H"]["counts"],
                pruned_metrics["hist1d_H"]["counts"],
            )

            # CC_rate delta significance (base thresholds vs pruned).
            n_base = int(base_metrics["kept_tokens"])
            n_pruned = int(pruned_metrics["kept_tokens"])
            base_cc = float(base_metrics["cc_rate"])
            pruned_cc_base = float(pruned_metrics["cc_rate_base_thr"] or 0.0)
            cc_delta = float(pruned_cc_base - base_cc)
            cc_z = _z_for_prop_delta(base_cc, n_base, pruned_cc_base, n_pruned)
            out["seq"][str(int(seq_len))] = {
                "rows_seen": int(blocks.rows_seen),
                "pack_wall_s": float(blocks.wall_s),
                "base": base_metrics,
                "pruned": pruned_metrics,
                "base_thresholds": {"p_thr": float(base_thresholds[0]), "h_thr": float(base_thresholds[1])},
                "js_divergence_2d": float(js2d),
                "js_divergence_x": float(jsx),
                "js_divergence_h": float(jsh),
                "cc_delta_base_thr": float(cc_delta),
                "cc_z_base_thr": float(cc_z),
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

    return {
        "meta": {
            "dataset_repo": str(dataset_repo),
            "pack_files": list(pack_files),
            "base_model_id": str(base_model_id),
            "pruned_model_id": str(pruned_model_id),
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
            "axes": {"x": "p_t" if str(prob_scale) == "linear" else "log10(p_t)", "y": "H_topK/ln(K)"},
            "x_min": 0.0 if str(prob_scale) == "linear" else float(logp_min),
            "x_max": 1.0 if str(prob_scale) == "linear" else float(logp_max),
        },
        "packs": results,
    }


def _render_html_dashboard(data: dict[str, Any]) -> str:
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EAFT diagnostics — probability/entropy landscapes</title>
  <style>
    :root {{
      --bg: #0b0f14;
      --panel: #0f172a;
      --panel-2: #111827;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --grid: #1f2937;
      --accent: #f97316;
      --good: #34d399;
      --bad: #f87171;
      --plot-bg: #0b0f14;
      --plot-axis: #cbd5e1;
    }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: "IBM Plex Sans", "Space Grotesk", ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; color: var(--text); background: var(--bg); }}
    h1 {{ margin: 0 0 8px 0; font-size: 22px; letter-spacing: 0.4px; }}
    h2 {{ margin: 0 0 12px 0; font-size: 16px; }}
    .meta {{ color: var(--muted); font-size: 13px; line-height: 1.4; }}
    .row {{ display: flex; gap: 16px; align-items: stretch; flex-wrap: nowrap; overflow-x: auto; padding-bottom: 6px; }}
    .panel {{ border: 1px solid var(--grid); border-radius: 12px; padding: 14px; background: var(--panel); min-width: 660px; flex: 0 0 auto; }}
    .panel h3 {{ margin: 0 0 10px 0; font-size: 14px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }}
    .panel.hero {{ min-width: 100%; background: linear-gradient(135deg, #0f172a, #0b1220); }}
    .controls {{ display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin: 12px 0 16px 0; min-width: 100%; }}
    select {{ padding: 6px 10px; background: #0b1220; color: var(--text); border: 1px solid var(--grid); border-radius: 8px; }}
    canvas {{ border: 1px solid var(--grid); border-radius: 8px; background: var(--plot-bg); width: 100%; height: auto; }}
    canvas.heatmap {{ height: clamp(320px, 45vh, 560px); }}
    canvas.hist {{ height: clamp(140px, 18vh, 240px); }}
    .small {{ font-size: 12px; color: var(--muted); }}
    table {{ border-collapse: collapse; font-size: 12px; width: 100%; }}
    td, th {{ border: 1px solid var(--grid); padding: 8px 10px; text-align: right; }}
    th {{ background: #0b1220; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; font-size: 11px; }}
    .left {{ text-align: left; }}
    .legend {{ display: flex; gap: 12px; align-items: center; }}
    .swatch {{ width: 240px; height: 12px; border-radius: 999px; background: linear-gradient(90deg, #1b1f3a, #234f9a, #25a18e, #e9c46a, #f4a261, #e76f51); border: 1px solid var(--grid); }}
    .hint {{ color: var(--muted); font-size: 12px; }}
    .hero-grid {{ display: grid; grid-template-columns: repeat(5, minmax(160px, 1fr)); gap: 12px; }}
    .stat {{ background: var(--panel-2); border: 1px solid var(--grid); border-radius: 12px; padding: 12px; }}
    .stat .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .stat .value {{ font-size: 26px; font-weight: 700; margin-top: 6px; }}
    .stat .sub {{ font-size: 11px; color: var(--muted); margin-top: 4px; }}
    .badge {{ display: inline-block; padding: 2px 6px; border-radius: 999px; font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .bad {{ color: var(--bad); }}
    .good {{ color: var(--good); }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }}
    .ranking-table tr {{ cursor: pointer; }}
    .ranking-table tr:hover {{ background: #0b1220; }}
  </style>
</head>
<body>
  <h1>EAFT diagnostics: probability–entropy landscapes</h1>
  <div class="meta" id="meta"></div>

  <div class="panel hero">
    <h2>Degradation Summary (selected pack)</h2>
    <div class="hero-grid" id="heroStats"></div>
    <div class="small" id="heroNote"></div>
  </div>

  <div class="panel" style="min-width:100%;">
    <h2>Pack Ranking (Degradation Score)</h2>
    <div class="small">Score uses z-normalized metrics across packs for the selected seq_len. Higher = worse.</div>
    <div id="rankingTable"></div>
  </div>

  <div class="controls panel">
    <div><span class="small">Pack</span><br/><select id="packSel"></select></div>
    <div><span class="small">seq_len</span><br/><select id="seqSel"></select></div>
    <div class="legend">
      <div class="swatch"></div>
      <div class="hint">heatmap shows log(count+1) density. Lines are CC thresholds.</div>
    </div>
  </div>

  <div class="row">
    <div class="panel">
      <h3>Base</h3>
      <canvas id="cBase" class="heatmap" width="700" height="520"></canvas>
      <div class="small" id="baseStats"></div>
    </div>
    <div class="panel">
      <h3>Pruned</h3>
      <canvas id="cPruned" class="heatmap" width="700" height="520"></canvas>
      <div class="small" id="prunedStats"></div>
    </div>
    <div class="panel">
      <h3>Δ Density (pruned - base)</h3>
      <canvas id="cDelta" class="heatmap" width="700" height="520"></canvas>
      <div class="small" id="deltaStats"></div>
    </div>
  </div>

  <div class="row" style="margin-top: 16px;">
    <div class="panel">
      <h3>Histograms (Base)</h3>
      <canvas id="hBaseLogp" class="hist" width="700" height="180"></canvas>
      <canvas id="hBaseH" class="hist" width="700" height="180" style="margin-top:10px;"></canvas>
    </div>
    <div class="panel">
      <h3>Histograms (Pruned)</h3>
      <canvas id="hPrunedLogp" class="hist" width="700" height="180"></canvas>
      <canvas id="hPrunedH" class="hist" width="700" height="180" style="margin-top:10px;"></canvas>
    </div>
  </div>

  <div class="panel" style="margin-top: 16px; min-width:100%;">
    <h2>Base vs Pruned Metrics</h2>
    <div id="compareTable"></div>
  </div>

  <script id="DATA" type="application/json">{payload}</script>
  <script>
    const DATA = JSON.parse(document.getElementById("DATA").textContent);

    function fmt(x, d=4) {{
      if (x === null || x === undefined || Number.isNaN(x)) return "—";
      return Number(x).toFixed(d);
    }}

    function fmtSigned(x, d=4) {{
      if (x === null || x === undefined || Number.isNaN(x)) return "—";
      const v = Number(x);
      const sign = v >= 0 ? "+" : "";
      return sign + v.toFixed(d);
    }}

    function pFromZ(z) {{
      // Two-sided p-value from z using an erf approximation.
      const x = Math.abs(z) / Math.SQRT2;
      // Abramowitz & Stegun 7.1.26
      const t = 1 / (1 + 0.3275911 * x);
      const a1 = 0.254829592;
      const a2 = -0.284496736;
      const a3 = 1.421413741;
      const a4 = -1.453152027;
      const a5 = 1.061405429;
      const erf = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
      const erfc = 1 - erf;
      const p = erfc; // two-sided
      return p;
    }}

    function deltaClass(v) {{
      if (v === null || v === undefined || Number.isNaN(v)) return "";
      if (v > 0) return "bad";
      if (v < 0) return "good";
      return "";
    }}

    function nllCI(mean, variance, n) {{
      const nn = Math.max(1, n || 1);
      const se = Math.sqrt(Math.max(0, variance) / nn);
      const lo = mean - 1.96 * se;
      const hi = mean + 1.96 * se;
      return {{ lo, hi, se }};
    }}

    function pplCI(mean, variance, n) {{
      const ci = nllCI(mean, variance, n);
      return {{
        lo: Math.exp(ci.lo),
        hi: Math.exp(ci.hi),
      }};
    }}

    function deltaNllStats(base, pruned) {{
      const n1 = base.kept_tokens || 1;
      const n2 = pruned.kept_tokens || 1;
      const se = Math.sqrt((base.var_nll / n1) + (pruned.var_nll / n2));
      const delta = pruned.mean_nll - base.mean_nll;
      const z = se > 0 ? (delta / se) : 0;
      const p = pFromZ(z);
      const ci = {{
        lo: delta - 1.96 * se,
        hi: delta + 1.96 * se,
      }};
      return {{ delta, z, p, ci }};
    }}

    function ccDeltaCI(p1, n1, p2, n2) {{
      const nn1 = Math.max(1, n1 || 1);
      const nn2 = Math.max(1, n2 || 1);
      const se = Math.sqrt((p1 * (1 - p1) / nn1) + (p2 * (1 - p2) / nn2));
      return {{
        lo: (p2 - p1) - 1.96 * se,
        hi: (p2 - p1) + 1.96 * se,
      }};
    }}

    function cssVar(name, fallback) {{
      const val = getComputedStyle(document.body).getPropertyValue(name).trim();
      return val || fallback;
    }}

    const THEME = {{
      bg: cssVar("--plot-bg", "#0b0f14"),
      axis: cssVar("--plot-axis", "#cbd5e1"),
      text: cssVar("--text", "#e5e7eb"),
      grid: cssVar("--grid", "#1f2937"),
    }};

    function prepareCanvas(canvas) {{
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(1, rect.width);
      const h = Math.max(1, rect.height);
      const needResize = canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr);
      if (needResize) {{
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }}
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return {{ ctx, W: w, H: h }};
    }}

    function colorRamp(t) {{
      // High-contrast gradient for dark mode.
      const stops = [
        [0.00, [12, 18, 36]],
        [0.25, [25, 72, 160]],
        [0.45, [37, 168, 154]],
        [0.65, [234, 204, 120]],
        [0.85, [245, 153, 94]],
        [1.00, [239, 83, 80]],
      ];
      t = Math.min(1, Math.max(0, t));
      for (let i = 0; i < stops.length - 1; i++) {{
        const a = stops[i], b = stops[i+1];
        if (t >= a[0] && t <= b[0]) {{
          const u = (t - a[0]) / (b[0] - a[0] + 1e-12);
          const rgb = [
            Math.round(a[1][0] + u*(b[1][0]-a[1][0])),
            Math.round(a[1][1] + u*(b[1][1]-a[1][1])),
            Math.round(a[1][2] + u*(b[1][2]-a[1][2])),
          ];
          return `rgb(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}})`;
        }}
      }}
      return "rgb(0,0,0)";
    }}

    function drawHeatmap(canvas, hist, thresholds, title, extraThresholds=null) {{
      const {{ ctx, W, H }} = prepareCanvas(canvas);
      const padL = 44, padR = 10, padT = 16, padB = 36;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = THEME.bg;
      ctx.fillRect(0, 0, W, H);
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;

      const xbins = hist.xbins, ybins = hist.ybins;
      const counts = hist.counts;
      // counts are flattened with x-major, y-minor (histogram2d default): idx = ix*ybins + iy
      let maxLog = 0;
      for (let i = 0; i < counts.length; i++) {{
        const v = Math.log1p(counts[i]);
        if (v > maxLog) maxLog = v;
      }}
      maxLog = Math.max(1e-12, maxLog);

      // heatmap pixels
      for (let ix = 0; ix < xbins; ix++) {{
        for (let iy = 0; iy < ybins; iy++) {{
          const idx = ix*ybins + iy;
          const v = Math.log1p(counts[idx]) / maxLog;
          ctx.fillStyle = colorRamp(v);
          const x0 = padL + (ix / xbins) * plotW;
          const y0 = padT + ((ybins - 1 - iy) / ybins) * plotH;
          const x1 = padL + ((ix + 1) / xbins) * plotW;
          const y1 = padT + ((ybins - iy) / ybins) * plotH;
          ctx.fillRect(x0, y0, x1 - x0 + 0.5, y1 - y0 + 0.5);
        }}
      }}

      // axes
      ctx.strokeStyle = THEME.axis;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, padT);
      ctx.lineTo(padL, padT + plotH);
      ctx.lineTo(padL + plotW, padT + plotH);
      ctx.stroke();

      // labels
      ctx.fillStyle = THEME.text;
      ctx.font = "12px sans-serif";
      const xLabel = DATA.meta.axes.x || "p_t";
      ctx.fillText(xLabel, padL + plotW/2 - 22, H - 10);
      ctx.save();
      ctx.translate(14, padT + plotH/2 + 28);
      ctx.rotate(-Math.PI/2);
      ctx.fillText("H_topK/ln(K)", 0, 0);
      ctx.restore();

      // threshold lines (CC region)
      if (thresholds) {{
        const xMin = DATA.meta.x_min, xMax = DATA.meta.x_max;
        const yMin = 0.0, yMax = 1.0;
        const xThr = Math.min(xMax, Math.max(xMin, thresholds.x_thr));
        const yThr = Math.min(yMax, Math.max(yMin, thresholds.h_thr));
        const x = padL + ((xThr - xMin) / (xMax - xMin)) * plotW;
        const y = padT + (1 - (yThr - yMin) / (yMax - yMin)) * plotH;
        ctx.strokeStyle = "rgba(226,232,240,0.9)";
        ctx.setLineDash([6,4]);
        ctx.beginPath();
        ctx.moveTo(x, padT);
        ctx.lineTo(x, padT + plotH);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(padL + plotW, y);
        ctx.stroke();
        ctx.setLineDash([]);

        // shade CC rectangle (low p, low H)
        ctx.fillStyle = "rgba(255,255,255,0.10)";
        ctx.fillRect(padL, y, x - padL, (padT + plotH) - y);
        ctx.strokeStyle = "rgba(255,255,255,0.35)";
        ctx.strokeRect(padL, y, x - padL, (padT + plotH) - y);
      }}

      // optional second set of thresholds (for comparison, no shading)
      if (extraThresholds) {{
        const xMin = DATA.meta.x_min, xMax = DATA.meta.x_max;
        const yMin = 0.0, yMax = 1.0;
        const xThr = Math.min(xMax, Math.max(xMin, extraThresholds.x_thr));
        const yThr = Math.min(yMax, Math.max(yMin, extraThresholds.h_thr));
        const x = padL + ((xThr - xMin) / (xMax - xMin)) * plotW;
        const y = padT + (1 - (yThr - yMin) / (yMax - yMin)) * plotH;
        ctx.strokeStyle = "rgba(255,255,255,0.7)";
        ctx.setLineDash([2,4]);
        ctx.beginPath();
        ctx.moveTo(x, padT);
        ctx.lineTo(x, padT + plotH);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(padL + plotW, y);
        ctx.stroke();
        ctx.setLineDash([]);
      }}

      if (title) {{
        ctx.fillStyle = THEME.text;
        ctx.font = "bold 12px sans-serif";
        ctx.fillText(title, padL, 12);
      }}
    }}

    function drawDeltaHeatmap(canvas, histA, histB, thresholds, title) {{
      const {{ ctx, W, H }} = prepareCanvas(canvas);
      const padL = 44, padR = 10, padT = 16, padB = 36;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = THEME.bg;
      ctx.fillRect(0, 0, W, H);
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;

      const xbins = histA.xbins, ybins = histA.ybins;
      const a = histA.counts;
      const b = histB.counts;
      let maxAbs = 0;
      const diff = new Array(a.length);
      for (let i = 0; i < a.length; i++) {{
        const v = Math.log1p(b[i]) - Math.log1p(a[i]);
        diff[i] = v;
        maxAbs = Math.max(maxAbs, Math.abs(v));
      }}
      maxAbs = Math.max(1e-9, maxAbs);

      function diverging(t) {{
        // t in [-1, 1], blue -> white -> red
        const u = (t + 1) / 2;
        const r = Math.round(220 * u + 20);
        const b = Math.round(220 * (1 - u) + 20);
        const g = Math.round(220 * (1 - Math.abs(t)) + 20);
        return `rgb(${{r}},${{g}},${{b}})`;
      }}

      for (let ix = 0; ix < xbins; ix++) {{
        for (let iy = 0; iy < ybins; iy++) {{
          const idx = ix*ybins + iy;
          const v = diff[idx] / maxAbs;
          ctx.fillStyle = diverging(v);
          const x0 = padL + (ix / xbins) * plotW;
          const y0 = padT + ((ybins - 1 - iy) / ybins) * plotH;
          const x1 = padL + ((ix + 1) / xbins) * plotW;
          const y1 = padT + ((ybins - iy) / ybins) * plotH;
          ctx.fillRect(x0, y0, x1 - x0 + 0.5, y1 - y0 + 0.5);
        }}
      }}

      // axes
      ctx.strokeStyle = THEME.axis;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, padT);
      ctx.lineTo(padL, padT + plotH);
      ctx.lineTo(padL + plotW, padT + plotH);
      ctx.stroke();

      // labels
      ctx.fillStyle = THEME.text;
      ctx.font = "12px sans-serif";
      const xLabel = DATA.meta.axes.x || "p_t";
      ctx.fillText(xLabel, padL + plotW/2 - 22, H - 10);
      ctx.save();
      ctx.translate(14, padT + plotH/2 + 28);
      ctx.rotate(-Math.PI/2);
      ctx.fillText("H_topK/ln(K)", 0, 0);
      ctx.restore();

      // threshold lines (base thresholds)
      if (thresholds) {{
        const xMin = DATA.meta.x_min, xMax = DATA.meta.x_max;
        const yMin = 0.0, yMax = 1.0;
        const xThr = Math.min(xMax, Math.max(xMin, thresholds.x_thr));
        const yThr = Math.min(yMax, Math.max(yMin, thresholds.h_thr));
        const x = padL + ((xThr - xMin) / (xMax - xMin)) * plotW;
        const y = padT + (1 - (yThr - yMin) / (yMax - yMin)) * plotH;
        ctx.strokeStyle = "rgba(226,232,240,0.9)";
        ctx.setLineDash([6,4]);
        ctx.beginPath();
        ctx.moveTo(x, padT);
        ctx.lineTo(x, padT + plotH);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(padL + plotW, y);
        ctx.stroke();
        ctx.setLineDash([]);
      }}

      if (title) {{
        ctx.fillStyle = THEME.text;
        ctx.font = "bold 12px sans-serif";
        ctx.fillText(title, padL, 12);
      }}
    }}

    function drawHist(canvas, hist, thresholds, label) {{
      const {{ ctx, W, H }} = prepareCanvas(canvas);
      const padL = 44, padR = 10, padT = 10, padB = 20;
      ctx.clearRect(0,0,W,H);
      ctx.fillStyle = THEME.bg; ctx.fillRect(0,0,W,H);
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;
      const counts = hist.counts;
      let maxC = 1;
      for (const c of counts) maxC = Math.max(maxC, c);
      // bars
      const n = counts.length;
      for (let i = 0; i < n; i++) {{
        const x0 = padL + (i / n) * plotW;
        const x1 = padL + ((i + 1) / n) * plotW;
        const h = (counts[i] / maxC) * plotH;
      ctx.fillStyle = "rgba(76, 146, 219, 0.85)";
        ctx.fillRect(x0, padT + (plotH - h), Math.max(1, x1 - x0 - 0.5), h);
      }}
      // axes
      ctx.strokeStyle=THEME.axis; ctx.beginPath();
      ctx.moveTo(padL, padT); ctx.lineTo(padL, padT+plotH); ctx.lineTo(padL+plotW, padT+plotH); ctx.stroke();
      // label
      ctx.fillStyle=THEME.text; ctx.font="12px sans-serif";
      ctx.fillText(label, padL, H-4);
      // threshold line
      if (thresholds && thresholds.value !== null && thresholds.value !== undefined) {{
        const vmin = hist.edges[0], vmax = hist.edges[hist.edges.length-1];
        const x = padL + ((thresholds.value - vmin) / (vmax - vmin)) * plotW;
        ctx.strokeStyle="rgba(226,232,240,0.9)";
        ctx.setLineDash([6,4]);
        ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT+plotH); ctx.stroke();
        ctx.setLineDash([]);
      }}
    }}

    function populate() {{
      const packSel = document.getElementById("packSel");
      const seqSel = document.getElementById("seqSel");
      packSel.innerHTML = "";
      seqSel.innerHTML = "";
      for (const p of DATA.packs) {{
        const opt = document.createElement("option");
        opt.value = p.pack;
        opt.textContent = p.pack;
        packSel.appendChild(opt);
      }}
      ["1024","2048"].forEach(s => {{
        const opt = document.createElement("option");
        opt.value = s;
        opt.textContent = s;
        seqSel.appendChild(opt);
      }});
    }}

    function getPack(name) {{
      return DATA.packs.find(p => p.pack === name);
    }}

    function metricForPack(p, seq) {{
      const s = p.seq[seq];
      const base = s.base;
      const pruned = s.pruned;
      const deltaPpl = pruned.ppl - base.ppl;
      const ccDelta = (s.cc_delta_base_thr !== undefined) ? s.cc_delta_base_thr : (pruned.cc_rate - base.cc_rate);
      const ccDeltaPP = ccDelta * 100.0;
      const meanPDropPP = (base.mean_prob - pruned.mean_prob) * 100.0;
      const js2d = s.js_divergence_2d;
      return {{
        pack: p.pack,
        seq,
        deltaPpl,
        ccDeltaPP,
        js2d,
        meanPDropPP,
        base,
        pruned,
        s,
      }};
    }}

    function zScores(values) {{
      const n = values.length;
      if (!n) return values.map(() => 0);
      const mean = values.reduce((a, b) => a + b, 0) / n;
      const variance = values.reduce((a, b) => a + (b - mean) * (b - mean), 0) / n;
      const std = Math.sqrt(variance) || 1e-9;
      return values.map(v => (v - mean) / std);
    }}

    function buildRanking(seq) {{
      const rows = DATA.packs.map(p => metricForPack(p, seq));
      const zDelta = zScores(rows.map(r => r.deltaPpl));
      const zCC = zScores(rows.map(r => r.ccDeltaPP));
      const zJS = zScores(rows.map(r => r.js2d));
      const zMeanP = zScores(rows.map(r => r.meanPDropPP));
      rows.forEach((r, i) => {{
        r.z_delta = zDelta[i];
        r.z_cc = zCC[i];
        r.z_js = zJS[i];
        r.z_meanp = zMeanP[i];
        r.score = r.z_delta + r.z_cc + r.z_js + r.z_meanp;
      }});
      const zScore = zScores(rows.map(r => r.score));
      rows.forEach((r, i) => {{
        r.score_z = zScore[i];
      }});
      rows.sort((a, b) => b.score - a.score);
      return rows;
    }}

    function computeGlobal(rows) {{
      const n = rows.length || 1;
      const avg = (getter) => rows.reduce((a, r) => a + getter(r), 0) / n;
      return {{
        deltaPpl: avg(r => r.deltaPpl),
        ccDeltaPP: avg(r => r.ccDeltaPP),
        js2d: avg(r => r.js2d),
        meanPDropPP: avg(r => r.meanPDropPP),
        cc_z: avg(r => r.s.cc_z_base_thr || 0.0),
      }};
    }}

    function renderRanking(rows, selectedPack) {{
      let html = "<table class='ranking-table'><tr><th>#</th><th class='left'>pack</th><th>score_z</th><th>score</th><th>ΔPPL</th><th>CC Δ (pp)</th><th>JS2D</th><th>Δ mean p (pp)</th><th>CC z</th></tr>";
      rows.forEach((r, idx) => {{
        const sel = (r.pack === selectedPack) ? " style='background:#0b1220;'" : "";
        html += `<tr data-pack='${{r.pack}}'${{sel}}>` +
          `<td>${{idx + 1}}</td>` +
          `<td class='left'>${{r.pack}}</td>` +
          `<td class='mono'>${{fmt(r.score_z, 2)}}</td>` +
          `<td class='mono'>${{fmt(r.score, 2)}}</td>` +
          `<td class='mono'>${{fmtSigned(r.deltaPpl, 3)}}</td>` +
          `<td class='mono'>${{fmtSigned(r.ccDeltaPP, 2)}}</td>` +
          `<td class='mono'>${{fmt(r.js2d, 4)}}</td>` +
          `<td class='mono'>${{fmtSigned(r.meanPDropPP, 2)}}</td>` +
          `<td class='mono'>${{fmt(r.s.cc_z_base_thr, 2)}}</td>` +
          "</tr>";
      }});
      html += "</table>";
      const node = document.getElementById("rankingTable");
      node.innerHTML = html;
      node.querySelectorAll("tr[data-pack]").forEach(tr => {{
        tr.addEventListener("click", () => {{
          document.getElementById("packSel").value = tr.getAttribute("data-pack");
          render();
        }});
      }});
    }}

    function renderHero(row, globalRow) {{
      const score = row.score_z;
      const severity = score >= 2.0 ? "bad" : (score <= -2.0 ? "good" : "");
      const ccP = pFromZ(row.s.cc_z_base_thr);
      const scoreP = pFromZ(score);
      const tail = scoreP / 2.0;
      const percentile = (1.0 - tail) * 100.0;
      const verdict = (row.deltaPpl > 0 && row.ccDeltaPP > 0 && row.js2d > 0.01 && score >= 2.0 && scoreP < 0.05)
        ? "DEFINITIVELY WORSE"
        : "MIXED / INCONCLUSIVE";
      const verdictCls = verdict === "DEFINITIVELY WORSE" ? "bad" : "";
      const hero = [
        {{
          label: "degradation score (z)",
          value: fmt(score, 2),
          sub: `rank z-score | raw=${{fmt(row.score, 2)}}`,
          cls: severity,
        }},
        {{
          label: "ΔPPL",
          value: fmtSigned(row.deltaPpl, 3),
          sub: "completion-only PPL",
          cls: row.deltaPpl > 0 ? "bad" : "good",
        }},
        {{
          label: "CC Δ (pp)",
          value: fmtSigned(row.ccDeltaPP, 2),
          sub: `z=${{fmt(row.s.cc_z_base_thr, 2)}} | p=${{ccP.toExponential(2)}}`,
          cls: row.ccDeltaPP > 0 ? "bad" : "good",
        }},
        {{
          label: "JS2D",
          value: fmt(row.js2d, 4),
          sub: "distribution shift",
          cls: row.js2d > 0.02 ? "bad" : "",
        }},
        {{
          label: "Δ mean p (pp)",
          value: fmtSigned(row.meanPDropPP, 2),
          sub: "base - pruned",
          cls: row.meanPDropPP > 0 ? "bad" : "good",
        }},
        {{
          label: "VERDICT",
          value: verdict,
          sub: "based on ΔPPL + CC Δ + JS2D + score_z",
          cls: verdictCls,
        }},
        {{
          label: "GLOBAL ΔPPL",
          value: fmtSigned(globalRow.deltaPpl, 3),
          sub: "avg across packs",
          cls: globalRow.deltaPpl > 0 ? "bad" : "good",
        }},
        {{
          label: "GLOBAL CC Δ (pp)",
          value: fmtSigned(globalRow.ccDeltaPP, 2),
          sub: "avg across packs",
          cls: globalRow.ccDeltaPP > 0 ? "bad" : "good",
        }},
        {{
          label: "GLOBAL JS2D",
          value: fmt(globalRow.js2d, 4),
          sub: "avg across packs",
          cls: globalRow.js2d > 0.02 ? "bad" : "",
        }},
        {{
          label: "GLOBAL Δ mean p (pp)",
          value: fmtSigned(globalRow.meanPDropPP, 2),
          sub: "avg across packs",
          cls: globalRow.meanPDropPP > 0 ? "bad" : "good",
        }},
      ];
      let html = "";
      hero.forEach(h => {{
        const extra = h.cls ? (" " + h.cls) : "";
        html += `<div class='stat'><div class='label'>${{h.label}}</div><div class='value${{extra}}'>${{h.value}}</div><div class='sub'>${{h.sub}}</div></div>`;
      }});
      document.getElementById("heroStats").innerHTML = html;
      document.getElementById("heroNote").textContent =
        "Score = z-score of combined degradation across packs. " +
        `Interpretation: score_z≈${{fmt(score,2)}} (~${{percentile.toFixed(1)}}th percentile, p=${{scoreP.toExponential(2)}}).`;
    }}

    function renderCompare(base, pruned, s) {{
      const ccP = pFromZ(s.cc_z_base_thr || 0.0);
      const baseNllCI = nllCI(base.mean_nll, base.var_nll, base.kept_tokens);
      const prunedNllCI = nllCI(pruned.mean_nll, pruned.var_nll, pruned.kept_tokens);
      const basePplCI = pplCI(base.mean_nll, base.var_nll, base.kept_tokens);
      const prunedPplCI = pplCI(pruned.mean_nll, pruned.var_nll, pruned.kept_tokens);
      const deltaNll = deltaNllStats(base, pruned);
      const ccDelta = ccDeltaCI(base.cc_rate, base.kept_tokens, pruned.cc_rate_base_thr, pruned.kept_tokens);
      const rows = [
        ["PPL", fmt(base.ppl,3), fmt(pruned.ppl,3), fmtSigned(pruned.ppl - base.ppl,3)],
        ["PPL 95% CI", `${{fmt(basePplCI.lo,3)}}–${{fmt(basePplCI.hi,3)}}`, `${{fmt(prunedPplCI.lo,3)}}–${{fmt(prunedPplCI.hi,3)}}`, "—"],
        ["mean NLL", fmt(base.mean_nll,6), fmt(pruned.mean_nll,6), fmtSigned(pruned.mean_nll - base.mean_nll,6)],
        ["mean NLL 95% CI", `${{fmt(baseNllCI.lo,6)}}–${{fmt(baseNllCI.hi,6)}}`, `${{fmt(prunedNllCI.lo,6)}}–${{fmt(prunedNllCI.hi,6)}}`, "—"],
        ["Δ NLL (mean)", "—", "—", fmtSigned(deltaNll.delta,6)],
        ["Δ NLL 95% CI", "—", "—", `${{fmt(deltaNll.ci.lo,6)}}–${{fmt(deltaNll.ci.hi,6)}}`],
        ["Δ NLL z / p", "—", "—", `${{fmt(deltaNll.z,2)}} | p=${{deltaNll.p.toExponential(2)}}`],
        ["CC_rate (self)", fmt(base.cc_rate,4), fmt(pruned.cc_rate,4), fmtSigned(pruned.cc_rate - base.cc_rate,4)],
        ["CC_rate (baseThr)", fmt(base.cc_rate,4), fmt(pruned.cc_rate_base_thr,4), fmtSigned(s.cc_delta_base_thr,5)],
        ["CC Δ 95% CI (baseThr)", "—", "—", `${{fmtSigned(ccDelta.lo,5)}}–${{fmtSigned(ccDelta.hi,5)}}`],
        ["CC z / p (baseThr)", fmt(s.cc_z_base_thr,2), ccP.toExponential(2), "—"],
        ["mean_prob", fmt(base.mean_prob,6), fmt(pruned.mean_prob,6), fmtSigned(pruned.mean_prob - base.mean_prob,6)],
        ["mean_entropy", fmt(base.mean_entropy,4), fmt(pruned.mean_entropy,4), fmtSigned(pruned.mean_entropy - base.mean_entropy,4)],
        ["p quantiles", `p05=${{fmt(base.p_q05,6)}} p50=${{fmt(base.p_q50,6)}} p95=${{fmt(base.p_q95,6)}}`, `p05=${{fmt(pruned.p_q05,6)}} p50=${{fmt(pruned.p_q50,6)}} p95=${{fmt(pruned.p_q95,6)}}`, "—"],
        ["H quantiles", `H05=${{fmt(base.h_q05,4)}} H50=${{fmt(base.h_q50,4)}} H95=${{fmt(base.h_q95,4)}}`, `H05=${{fmt(pruned.h_q05,4)}} H50=${{fmt(pruned.h_q50,4)}} H95=${{fmt(pruned.h_q95,4)}}`, "—"],
        ["JS divergence (2D)", fmt(0.0,4), fmt(s.js_divergence_2d,4), fmtSigned(s.js_divergence_2d,4)],
        ["JS divergence (x)", fmt(0.0,4), fmt(s.js_divergence_x,4), fmtSigned(s.js_divergence_x,4)],
        ["JS divergence (H)", fmt(0.0,4), fmt(s.js_divergence_h,4), fmtSigned(s.js_divergence_h,4)],
      ];
      let html = "<table><tr><th class='left'>metric</th><th>base</th><th>pruned</th><th>Δ</th></tr>";
      rows.forEach(r => {{
        const d = (typeof r[3] === "string" && r[3] !== "—") ? Number(r[3]) : null;
        const cls = deltaClass(d);
        html += `<tr><td class='left'>${{r[0]}}</td><td>${{r[1]}}</td><td>${{r[2]}}</td><td class='mono ${{cls}}'>${{r[3]}}</td></tr>`;
      }});
      html += "</table>";
      document.getElementById("compareTable").innerHTML = html;
    }}

    function render() {{
      const packName = document.getElementById("packSel").value;
      const seq = document.getElementById("seqSel").value;
      const p = getPack(packName);
      const s = p.seq[seq];
      const base = s.base;
      const pruned = s.pruned;
      document.getElementById("meta").textContent =
        `dataset=${{DATA.meta.dataset_repo}} | base=${{DATA.meta.base_model_id}} | pruned=${{DATA.meta.pruned_model_id}} | top_k=${{DATA.meta.top_k}} | entropy_topk=${{DATA.meta.entropy_topk}} | cc_q=${{DATA.meta.cc_quantile}} | blocks=${{DATA.meta.num_blocks}} | x=${{DATA.meta.axes.x}}`;

      const baseThr = {{x_thr: base.x_thr, h_thr: base.h_thr}};
      const prunedThr = {{x_thr: pruned.x_thr, h_thr: pruned.h_thr}};
      const baseXThr = (DATA.meta.prob_scale === "linear")
        ? s.base_thresholds.p_thr
        : Math.log10(Math.max(1e-12, s.base_thresholds.p_thr));
      const baseThrForPruned = {{x_thr: baseXThr, h_thr: s.base_thresholds.h_thr}};

      drawHeatmap(
        document.getElementById("cBase"),
        base.hist2d_x_H,
        baseThr,
        `ppl=${{fmt(base.ppl,3)}} CC=${{fmt(base.cc_rate,4)}}`
      );
      // Pruned heatmap uses BASE thresholds as the primary CC region (shading),
      // and draws pruned's own thresholds as a secondary overlay (dotted).
      drawHeatmap(
        document.getElementById("cPruned"),
        pruned.hist2d_x_H,
        baseThrForPruned,
        `ppl=${{fmt(pruned.ppl,3)}} CC=${{fmt(pruned.cc_rate,4)}} (CC@baseThr=${{fmt(pruned.cc_rate_base_thr,4)}})`,
        prunedThr
      );
      drawDeltaHeatmap(
        document.getElementById("cDelta"),
        base.hist2d_x_H,
        pruned.hist2d_x_H,
        baseThrForPruned,
        `Δ log1p(density) | JS2D=${{fmt(s.js_divergence_2d,4)}}`
      );

      const xLabel = (DATA.meta.axes && DATA.meta.axes.x) ? DATA.meta.axes.x : "p_t";
      drawHist(document.getElementById("hBaseLogp"), base.hist1d_x, {{value: base.x_thr}}, `base ${{xLabel}}`);
      drawHist(document.getElementById("hBaseH"), base.hist1d_H, {{value: base.h_thr}}, "base H");
      drawHist(document.getElementById("hPrunedLogp"), pruned.hist1d_x, {{value: baseThrForPruned.x_thr}}, `pruned ${{xLabel}} (threshold=base)`);
      drawHist(document.getElementById("hPrunedH"), pruned.hist1d_H, {{value: baseThrForPruned.h_thr}}, "pruned H (threshold=base)");

      document.getElementById("baseStats").innerHTML =
        `rows_seen=${{s.rows_seen}} | kept_tokens=${{base.kept_tokens}} | tok/s(pred)=${{fmt(base.tok_s_pred,1)}} | mean_prob=${{fmt(base.mean_prob,5)}} | mean_entropy=${{fmt(base.mean_entropy,4)}} | p_thr=${{fmt(base.p_thr,6)}} | H_thr=${{fmt(base.h_thr,4)}} | quadrants LL=${{fmt(base.quadrants.LL,4)}} LH=${{fmt(base.quadrants.LH,4)}} HL=${{fmt(base.quadrants.HL,4)}}`;
      document.getElementById("prunedStats").innerHTML =
        `rows_seen=${{s.rows_seen}} | kept_tokens=${{pruned.kept_tokens}} | tok/s(pred)=${{fmt(pruned.tok_s_pred,1)}} | mean_prob=${{fmt(pruned.mean_prob,5)}} | mean_entropy=${{fmt(pruned.mean_entropy,4)}} | p_thr=${{fmt(pruned.p_thr,6)}} | H_thr=${{fmt(pruned.h_thr,4)}} | quadrants LL=${{fmt(pruned.quadrants.LL,4)}} LH=${{fmt(pruned.quadrants.LH,4)}} HL=${{fmt(pruned.quadrants.HL,4)}}`;
      document.getElementById("deltaStats").innerHTML =
        `CC Δ (baseThr)=${{fmtSigned(s.cc_delta_base_thr,5)}} | z=${{fmt(s.cc_z_base_thr,2)}} | JS2D=${{fmt(s.js_divergence_2d,4)}} | JSx=${{fmt(s.js_divergence_x,4)}} | JSh=${{fmt(s.js_divergence_h,4)}}`;

      const ranking = buildRanking(seq);
      renderRanking(ranking, packName);
      const selectedRow = ranking.find(r => r.pack === packName);
      const globalRow = computeGlobal(ranking);
      renderHero(selectedRow, globalRow);
      renderCompare(base, pruned, s);
    }}

    populate();
    document.getElementById("packSel").addEventListener("change", render);
    document.getElementById("seqSel").addEventListener("change", render);
    window.addEventListener("resize", render);
    render();
  </script>
</body>
</html>
"""


def _zscore(vals: list[float]) -> list[float]:
    if not vals:
        return []
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(var) if var > 0 else 1e-9
    return [(v - mean) / std for v in vals]


def _write_summary_md(res: dict[str, Any], out_path: Path) -> None:
    packs = res.get("packs", [])
    lines: list[str] = [
        "# EAFT degradation summary (base vs pruned)",
        "",
        f"- dataset: `{res['meta'].get('dataset_repo')}`",
        f"- base: `{res['meta'].get('base_model_id')}`",
        f"- pruned: `{res['meta'].get('pruned_model_id')}`",
        f"- top_k: {res['meta'].get('top_k')}",
        f"- entropy_topk: {res['meta'].get('entropy_topk')}",
        "",
    ]
    csv_lines = [
        "seq_len,pack,score_z,score,delta_ppl,delta_nll,delta_nll_p,cc_delta_pp,cc_delta_ci_lo,cc_delta_ci_hi,js2d,mean_p_drop_pp,cc_z,cc_p"
    ]

    for seq in ("1024", "2048"):
        rows = []
        for p in packs:
            s = p["seq"][seq]
            base = s["base"]
            pruned = s["pruned"]
            delta_ppl = float(pruned["ppl"] - base["ppl"])
            cc_delta_pp = float(s.get("cc_delta_base_thr", pruned["cc_rate"] - base["cc_rate"])) * 100.0
            mean_p_drop_pp = float(base["mean_prob"] - pruned["mean_prob"]) * 100.0
            js2d = float(s["js_divergence_2d"])
            cc_z = float(s.get("cc_z_base_thr", 0.0))
            cc_p = float(math.erfc(abs(cc_z) / math.sqrt(2.0)))
            # Delta NLL z + p
            n1 = max(1, int(base.get("kept_tokens", 1)))
            n2 = max(1, int(pruned.get("kept_tokens", 1)))
            var1 = float(base.get("var_nll", 0.0))
            var2 = float(pruned.get("var_nll", 0.0))
            mean1 = float(base.get("mean_nll", 0.0))
            mean2 = float(pruned.get("mean_nll", 0.0))
            delta_nll = mean2 - mean1
            se = math.sqrt((var1 / n1) + (var2 / n2)) if n1 and n2 else 0.0
            z_nll = (delta_nll / se) if se > 0 else 0.0
            p_nll = float(math.erfc(abs(z_nll) / math.sqrt(2.0)))
            # CC delta CI
            p1 = float(base.get("cc_rate", 0.0))
            p2 = float(pruned.get("cc_rate_base_thr", 0.0))
            cc_se = math.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))
            cc_ci_lo = (p2 - p1) - 1.96 * cc_se
            cc_ci_hi = (p2 - p1) + 1.96 * cc_se
            rows.append(
                {
                    "pack": p["pack"],
                    "delta_ppl": delta_ppl,
                    "delta_nll": delta_nll,
                    "delta_nll_p": p_nll,
                    "cc_delta_pp": cc_delta_pp,
                    "cc_ci_lo": cc_ci_lo,
                    "cc_ci_hi": cc_ci_hi,
                    "mean_p_drop_pp": mean_p_drop_pp,
                    "js2d": js2d,
                    "cc_z": cc_z,
                    "cc_p": cc_p,
                }
            )

        z_delta = _zscore([r["delta_ppl"] for r in rows])
        z_cc = _zscore([r["cc_delta_pp"] for r in rows])
        z_js = _zscore([r["js2d"] for r in rows])
        z_mp = _zscore([r["mean_p_drop_pp"] for r in rows])
        for i, r in enumerate(rows):
            r["score"] = z_delta[i] + z_cc[i] + z_js[i] + z_mp[i]
        z_score = _zscore([r["score"] for r in rows])
        for i, r in enumerate(rows):
            r["score_z"] = z_score[i]
        rows.sort(key=lambda r: r["score"], reverse=True)

        for r in rows:
            csv_lines.append(
                f"{seq},{r['pack']},{r['score_z']:.3f},{r['score']:.3f},"
                f"{r['delta_ppl']:+.6f},{r['delta_nll']:+.6f},{r['delta_nll_p']:.6e},"
                f"{r['cc_delta_pp']:+.4f},{r['cc_ci_lo']:+.6f},{r['cc_ci_hi']:+.6f},{r['js2d']:.6f},"
                f"{r['mean_p_drop_pp']:+.4f},{r['cc_z']:.4f},{r['cc_p']:.6e}"
            )

        # Global averages across packs.
        if rows:
            avg_delta = sum(r["delta_ppl"] for r in rows) / len(rows)
            avg_cc = sum(r["cc_delta_pp"] for r in rows) / len(rows)
            avg_js = sum(r["js2d"] for r in rows) / len(rows)
            avg_mp = sum(r["mean_p_drop_pp"] for r in rows) / len(rows)
            avg_cc_z = sum(r["cc_z"] for r in rows) / len(rows)
            avg_cc_p = float(math.erfc(abs(avg_cc_z) / math.sqrt(2.0)))
        else:
            avg_delta = avg_cc = avg_js = avg_mp = avg_cc_z = avg_cc_p = 0.0

        lines += [
            f"## seq_len={seq}",
            "",
            "| rank | pack | score_z | score | ΔPPL | ΔNLL | ΔNLL p | CC Δ (pp) | CC Δ CI | JS2D | Δ mean p (pp) | CC z | CC p |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| — | **GLOBAL_AVG** | — | — | {avg_delta:+.3f} | — | — | {avg_cc:+.2f} | — | {avg_js:.4f} | {avg_mp:+.2f} | {avg_cc_z:.2f} | {avg_cc_p:.2e} |",
        ]
        for i, r in enumerate(rows, 1):
            lines.append(
                f"| {i} | {r['pack']} | {r['score_z']:+.2f} | {r['score']:+.2f} | {r['delta_ppl']:+.3f} | "
                f"{r['delta_nll']:+.4f} | {r['delta_nll_p']:.2e} | {r['cc_delta_pp']:+.2f} | "
                f"{r['cc_ci_lo']:+.4f}–{r['cc_ci_hi']:+.4f} | {r['js2d']:.4f} | {r['mean_p_drop_pp']:+.2f} | "
                f"{r['cc_z']:.2f} | {r['cc_p']:.2e} |"
            )
        if rows:
            worst = rows[0]
            lines += [
                "",
                f"Worst pack (by score): `{worst['pack']}` | score_z={worst['score_z']:+.2f} | score={worst['score']:+.2f} | "
                f"ΔPPL={worst['delta_ppl']:+.3f} | ΔNLL={worst['delta_nll']:+.4f} | CC Δ={worst['cc_delta_pp']:+.2f} pp",
                "",
            ]

    lines += [
        "## Interpretation",
        "",
        "- `score_z` is the z-score of the combined degradation (ΔPPL + CC Δ + JS2D + Δ mean p) across packs.",
        "- `CC p` is a two-sided p-value from the CC z-score (smaller means more significant).",
        "- For quality risk: look for **positive ΔPPL**, **positive CC Δ**, **positive JS2D**, and **mean p drop** together.",
        "",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    csv_path = out_path.parent / "20b_calib_packs_eaft_degradation_summary.csv"
    csv_path.write_text("\n".join(csv_lines), encoding="utf-8")


@app.local_entrypoint()
def main(
    dataset_repo: str = DEFAULT_DATASET_REPO,
    pack_files_csv: str = ",".join(DEFAULT_PACK_FILES),
    base_model_id: str = DEFAULT_BASE_MODEL_ID,
    pruned_model_id: str = DEFAULT_PRUNED_MODEL_ID,
    top_k: int = 4,
    entropy_topk: int = 20,
    cc_quantile: float = 0.15,
    num_blocks: int = 32,
    batch_size: int = 1,
    hist_xbins: int = 160,
    hist_ybins: int = 120,
    prob_scale: str = "linear",
    logp_min: float = -12.0,
    logp_max: float = 0.0,
    input_json: str = "",
):
    pack_files = [x.strip() for x in (pack_files_csv or "").split(",") if x.strip()]
    if not pack_files:
        raise SystemExit("Empty --pack-files-csv")

    if input_json:
        res = json.loads(Path(input_json).read_text(encoding="utf-8"))
    else:
        res = eval_calib_packs_eaft_plots.remote(
            dataset_repo=str(dataset_repo),
            pack_files=pack_files,
            base_model_id=str(base_model_id),
            pruned_model_id=str(pruned_model_id),
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
        )

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("artifacts/eaft_plots") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "eaft_data.json"
    data_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    html = _render_html_dashboard(res)
    html_run_path = out_dir / "20b_calib_packs_eaft_plots.html"
    html_run_path.write_text(html, encoding="utf-8")

    latest_path = Path("reports/20b_calib_packs_eaft_plots.html")
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(html, encoding="utf-8")

    summary_path = Path("reports/20b_calib_packs_eaft_degradation_summary.md")
    _write_summary_md(res, summary_path)

    print(f"[+] Wrote {data_path}")
    print(f"[+] Wrote {html_run_path}")
    print(f"[+] Wrote {latest_path}")
    print(f"[+] Wrote {summary_path}")
