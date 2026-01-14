"""
EAFT-style pruning diagnostics on curated calib packs (completion-only, Harmony-packed).

Paper: Entropy-Adaptive Fine-Tuning (EAFT) introduces the "Confident Conflict" failure mode:
  - low probability assigned to the reference token p_t
  - low predictive entropy H_t (model is confident in a different token)

This script computes:
  - parity PPL (completion-only, packed blocks)
  - Top-K entropy approximation (default K=20) on the predicted distribution
  - Confident Conflict rate (CC_rate): % tokens in bottom q for BOTH p_t and H_t

Dataset repo:
  radna0/harmony-qwen3-calib-packs-v2-20260113

Requested pack set:
  - packs/reasoning_style_10k_v2/reasoning_style_10k_v2.parquet
  - tool_agentic_10k_v6.parquet
  - packs/calib_prompt_10000_v2/calib_prompt_10000_v2.parquet

Output:
  - reports/20b_calib_packs_eaft_diagnostics.md

Run (always log to unsloth_logs/):
  mkdir -p unsloth_logs
  ts=$(date +%Y%m%d_%H%M%S)
  nohup env MODAL_PROFILE=phamtrinhkien1203 modal run modal/eval_calib_packs_eaft_diagnostics.py \
    --base-model-id openai/gpt-oss-20b \
    --pruned-model-id sandeshrajx/gpt-oss-20b-reap-0.5-mxfp4 \
    --top-k 4 --entropy-topk 20 --cc-quantile 0.15 \
    --num-blocks 32 --batch-size 1 \
    > "unsloth_logs/calib_packs_eaft_${ts}.log" 2>&1 &
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable, Iterable

import modal

APP_NAME = "eval-calib-packs-eaft-diagnostics"


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


def _eaft_metrics_on_blocks(
    model,
    *,
    blocks: PackedBlocks,
    batch_size: int,
    entropy_topk: int,
    cc_quantile: float,
    base_thresholds: tuple[float, float] | None,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    entropy_topk = int(entropy_topk)
    cc_quantile = float(cc_quantile)
    if not (0.0 < cc_quantile < 1.0):
        raise ValueError("cc_quantile must be in (0,1)")

    probs_cpu: list[torch.Tensor] = []
    ent_cpu: list[torch.Tensor] = []

    total_nll = 0.0
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

        total_nll += float(nll[keep_t].sum().item())
        kept_tokens += int(keep_t.sum().item())

        probs_cpu.append(prob[keep_t].detach().float().cpu())
        ent_cpu.append(ent_norm[keep_t].detach().float().cpu())

    torch.cuda.synchronize()
    dt = max(1e-9, time.time() - t0)

    p_all = torch.cat(probs_cpu) if probs_cpu else torch.empty((0,), dtype=torch.float32)
    h_all = torch.cat(ent_cpu) if ent_cpu else torch.empty((0,), dtype=torch.float32)
    if p_all.numel() != h_all.numel():
        raise RuntimeError("prob/entropy arrays length mismatch")
    if p_all.numel() == 0:
        raise RuntimeError("No kept tokens; cannot compute EAFT diagnostics.")

    p_thr = float(torch.quantile(p_all, q=float(cc_quantile)).item())
    h_thr = float(torch.quantile(h_all, q=float(cc_quantile)).item())
    cc_mask = (p_all <= p_thr) & (h_all <= h_thr)
    cc_rate = float(cc_mask.float().mean().item())

    cc_rate_base = None
    if base_thresholds is not None:
        bp, bh = base_thresholds
        cc_rate_base = float(((p_all <= float(bp)) & (h_all <= float(bh))).float().mean().item())

    ppl = math.exp(total_nll / max(1, kept_tokens))

    return {
        "ppl": float(ppl),
        "kept_tokens": int(kept_tokens),
        "tok_s_pred": float(total_pred_tokens / dt),
        "mean_prob": float(p_all.mean().item()),
        "mean_entropy": float(h_all.mean().item()),
        "p_thr": float(p_thr),
        "h_thr": float(h_thr),
        "cc_rate": float(cc_rate),
        "cc_rate_base_thr": cc_rate_base,
    }


@app.function(
    image=image,
    gpu="H100:1",
    timeout=21600,
    cpu=16.0,
    memory=262144,
    volumes={"/root/data": data_volume, "/root/model": model_volume, "/root/hf_cache": hf_cache_volume},
    secrets=_secrets,
)
def eval_calib_packs_eaft(
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
        out: dict[str, Any] = {"pack": name}
        for seq_len in (1024, 2048):
            blocks = _pack_blocks(text_iter=text_it, tok=tok, eos_id=int(eos), seq_len=int(seq_len), num_blocks=int(num_blocks))
            base_metrics = _eaft_metrics_on_blocks(
                base_model,
                blocks=blocks,
                batch_size=int(batch_size),
                entropy_topk=int(entropy_topk),
                cc_quantile=float(cc_quantile),
                base_thresholds=None,
            )
            base_thresholds = (float(base_metrics["p_thr"]), float(base_metrics["h_thr"]))
            pruned_metrics = _eaft_metrics_on_blocks(
                pruned_model,
                blocks=blocks,
                batch_size=int(batch_size),
                entropy_topk=int(entropy_topk),
                cc_quantile=float(cc_quantile),
                base_thresholds=base_thresholds,
            )
            out[f"rows_seen_{seq_len}"] = int(blocks.rows_seen)
            out[f"base_ppl_{seq_len}"] = float(base_metrics["ppl"])
            out[f"pruned_ppl_{seq_len}"] = float(pruned_metrics["ppl"])
            out[f"delta_ppl_{seq_len}"] = float(pruned_metrics["ppl"] - base_metrics["ppl"])
            out[f"base_cc_{seq_len}"] = float(base_metrics["cc_rate"])
            out[f"pruned_cc_{seq_len}"] = float(pruned_metrics["cc_rate"])
            out[f"pruned_cc_baseThr_{seq_len}"] = float(pruned_metrics["cc_rate_base_thr"] or 0.0)
            out[f"base_meanH_{seq_len}"] = float(base_metrics["mean_entropy"])
            out[f"pruned_meanH_{seq_len}"] = float(pruned_metrics["mean_entropy"])
            out[f"base_meanP_{seq_len}"] = float(base_metrics["mean_prob"])
            out[f"pruned_meanP_{seq_len}"] = float(pruned_metrics["mean_prob"])
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
        },
        "results": results,
    }


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
):
    pack_files = [x.strip() for x in (pack_files_csv or "").split(",") if x.strip()]
    if not pack_files:
        raise SystemExit("Empty --pack-files-csv")

    out_path = Path("reports/20b_calib_packs_eaft_diagnostics.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    res = eval_calib_packs_eaft.remote(
        dataset_repo=str(dataset_repo),
        pack_files=pack_files,
        base_model_id=str(base_model_id),
        pruned_model_id=str(pruned_model_id),
        top_k=int(top_k),
        entropy_topk=int(entropy_topk),
        cc_quantile=float(cc_quantile),
        num_blocks=int(num_blocks),
        batch_size=int(batch_size),
    )

    q = float(res["meta"]["cc_quantile"])
    lines: list[str] = [
        "# 20B pruning diagnostics: EAFT-style Confident Conflicts",
        "",
        f"- Dataset repo: `{res['meta']['dataset_repo']}`",
        f"- Packs: {', '.join('`'+p+'`' for p in res['meta']['pack_files'])}",
        f"- Base: `{res['meta']['base_model_id']}`",
        f"- Pruned: `{res['meta']['pruned_model_id']}`",
        f"- top_k (forced for both): {int(res['meta']['top_k'])}",
        f"- entropy_topk: {int(res['meta']['entropy_topk'])} (entropy normalized by ln(K))",
        f"- CC definition: bottom {q:.2f} quantile in BOTH p_t and H_t (completion-only kept tokens)",
        f"- blocks per pack per seq_len: {int(res['meta']['num_blocks'])} | batch_size: {int(res['meta']['batch_size'])}",
        "",
        "## Summary (PPL + CC_rate)",
        "",
        "| pack | ppl1024 base | ppl1024 pruned | Δ | CC1024 base | CC1024 pruned | pruned CC@baseThr | ppl2048 base | ppl2048 pruned | Δ | CC2048 base | CC2048 pruned | pruned CC@baseThr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in res["results"]:
        lines.append(
            f"| {r['pack']} | {r['base_ppl_1024']:.3f} | {r['pruned_ppl_1024']:.3f} | {r['delta_ppl_1024']:+.3f} | "
            f"{r['base_cc_1024']:.4f} | {r['pruned_cc_1024']:.4f} | {r['pruned_cc_baseThr_1024']:.4f} | "
            f"{r['base_ppl_2048']:.3f} | {r['pruned_ppl_2048']:.3f} | {r['delta_ppl_2048']:+.3f} | "
            f"{r['base_cc_2048']:.4f} | {r['pruned_cc_2048']:.4f} | {r['pruned_cc_baseThr_2048']:.4f} |"
        )

    lines += [
        "",
        "## Mean Probability / Entropy",
        "",
        "These summarize the EAFT landscape axes on completion-only tokens:",
        "- `mean_prob` = mean reference-token probability `p_t`",
        "- `mean_entropy` = mean normalized Top-K entropy `H_topK / ln(K)`",
        "",
        "### seq_len=1024",
        "",
        "| pack | base mean_prob | pruned mean_prob | base mean_entropy | pruned mean_entropy |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in res["results"]:
        lines.append(
            f"| {r['pack']} | {r['base_meanP_1024']:.4f} | {r['pruned_meanP_1024']:.4f} | "
            f"{r['base_meanH_1024']:.4f} | {r['pruned_meanH_1024']:.4f} |"
        )

    lines += [
        "",
        "### seq_len=2048",
        "",
        "| pack | base mean_prob | pruned mean_prob | base mean_entropy | pruned mean_entropy |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in res["results"]:
        lines.append(
            f"| {r['pack']} | {r['base_meanP_2048']:.4f} | {r['pruned_meanP_2048']:.4f} | "
            f"{r['base_meanH_2048']:.4f} | {r['pruned_meanH_2048']:.4f} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- `CC_rate` is computed per-model using its own p/H quantile thresholds; `pruned CC@baseThr` applies base thresholds to pruned tokens.",
        "- `H_t` is Top-K entropy (K=entropy_topk) computed on the model's Top-K distribution and normalized by ln(K).",
        "- UNION is round-robin interleaving across packs (not dominated by the first pack).",
        "",
        "## Reproduce",
        "",
        "```bash",
        "modal run modal/eval_calib_packs_eaft_diagnostics.py \\",
        f"  --dataset-repo {dataset_repo} \\",
        f"  --pack-files-csv {pack_files_csv} \\",
        f"  --base-model-id {base_model_id} \\",
        f"  --pruned-model-id {pruned_model_id} \\",
        f"  --top-k {int(top_k)} --entropy-topk {int(entropy_topk)} --cc-quantile {float(cc_quantile)} \\",
        f"  --num-blocks {int(num_blocks)} --batch-size {int(batch_size)}",
        "```",
        "",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[+] Wrote {out_path}")
