"""
Parity PPL eval on curated calib packs (completion-only, Harmony-packed).

Dataset repo (HF datasets repo with parquet + jsonl):
  radna0/harmony-qwen3-calib-packs-v2-20260113

Requested pack set (30k total):
  - packs/reasoning_style_10k_v2/reasoning_style_10k_v2.parquet
  - tool_agentic_10k_v6.parquet
  - packs/calib_prompt_10000_v2/calib_prompt_10000_v2.parquet

Outputs:
  - reports/20b_calib_packs_ppl_parity.md

Run (always log to unsloth_logs/):
  mkdir -p unsloth_logs
  ts=$(date +%Y%m%d_%H%M%S)
  nohup env MODAL_PROFILE=phamtrinhkien1203 modal run modal/eval_calib_packs_ppl_parity.py \
    --base-model-id openai/gpt-oss-20b \
    --pruned-model-id sandeshrajx/gpt-oss-20b-reap-0.5-mxfp4 \
    --top-k 4 --num-blocks 64 --batch-size 1 \
    > "unsloth_logs/calib_packs_ppl_parity_${ts}.log" 2>&1 &
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

APP_NAME = "eval-calib-packs-ppl-parity"


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


def _apply_top_k(model, top_k: int) -> dict[str, Any]:
    applied = 0
    for layer in _iter_gpt_oss_layers(model):
        router = getattr(getattr(layer, "mlp", None), "router", None)
        if router is None:
            continue
        try:
            router.top_k = int(top_k)
            applied += 1
        except Exception:
            pass
    try:
        model.config.num_experts_per_tok = int(top_k)
        model.config.experts_per_token = int(top_k)
    except Exception:
        pass
    return {"layers_patched": int(applied)}


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


def _ppl_on_blocks(model, *, blocks: PackedBlocks, batch_size: int) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

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
        total_nll += float(nll[keep_t].sum().item())
        kept_tokens += int(keep_t.sum().item())
    torch.cuda.synchronize()
    dt = max(1e-9, time.time() - t0)
    ppl = math.exp(total_nll / max(1, kept_tokens))
    return {
        "ppl": float(ppl),
        "kept_tokens": int(kept_tokens),
        "pred_tokens": int(total_pred_tokens),
        "tok_s_pred": float(total_pred_tokens / dt),
        "wall_s": float(dt),
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
def eval_calib_packs(
    *,
    dataset_repo: str,
    pack_files: list[str],
    base_model_id: str,
    pruned_model_id: str,
    top_k: int,
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

    # Download parquet paths (once).
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
    _ = _apply_top_k(base_model, int(top_k))
    _ = _apply_top_k(pruned_model, int(top_k))

    def _eval_pack(name: str, text_it: Callable[[], Iterable[str]]) -> dict[str, Any]:
        blocks1024 = _pack_blocks(text_iter=text_it, tok=tok, eos_id=int(eos), seq_len=1024, num_blocks=int(num_blocks))
        blocks2048 = _pack_blocks(text_iter=text_it, tok=tok, eos_id=int(eos), seq_len=2048, num_blocks=int(num_blocks))

        base1024 = _ppl_on_blocks(base_model, blocks=blocks1024, batch_size=int(batch_size))
        base2048 = _ppl_on_blocks(base_model, blocks=blocks2048, batch_size=int(batch_size))
        pruned1024 = _ppl_on_blocks(pruned_model, blocks=blocks1024, batch_size=int(batch_size))
        pruned2048 = _ppl_on_blocks(pruned_model, blocks=blocks2048, batch_size=int(batch_size))
        return {
            "pack": name,
            "rows_seen_1024": int(blocks1024.rows_seen),
            "rows_seen_2048": int(blocks2048.rows_seen),
            "pack_wall_s_1024": float(blocks1024.wall_s),
            "pack_wall_s_2048": float(blocks2048.wall_s),
            "base_ppl_1024": float(base1024["ppl"]),
            "base_ppl_2048": float(base2048["ppl"]),
            "pruned_ppl_1024": float(pruned1024["ppl"]),
            "pruned_ppl_2048": float(pruned2048["ppl"]),
            "delta_1024": float(pruned1024["ppl"] - base1024["ppl"]),
            "delta_2048": float(pruned2048["ppl"] - base2048["ppl"]),
        }

    results: list[dict[str, Any]] = []
    # Individual packs
    for f, p in pack_paths.items():
        name = Path(f).stem
        results.append(_eval_pack(name, lambda p=p: _iter_parquet_texts(p, text_column="text")))

    # Union pack: concatenate streams in a deterministic order.
    def union_iter() -> Iterable[str]:
        iters = [_iter_parquet_texts(pack_paths[str(f)], text_column="text") for f in pack_files]
        # Round-robin interleave so UNION isn't dominated by the first pack.
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
    num_blocks: int = 64,
    batch_size: int = 1,
):
    pack_files = [x.strip() for x in (pack_files_csv or "").split(",") if x.strip()]
    if not pack_files:
        raise SystemExit("Empty --pack-files-csv")

    out_path = Path("reports/20b_calib_packs_ppl_parity.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    res = eval_calib_packs.remote(
        dataset_repo=str(dataset_repo),
        pack_files=pack_files,
        base_model_id=str(base_model_id),
        pruned_model_id=str(pruned_model_id),
        top_k=int(top_k),
        num_blocks=int(num_blocks),
        batch_size=int(batch_size),
    )

    lines: list[str] = [
        "# 20B pruning eval: calib packs parity PPL",
        "",
        f"- Dataset repo: `{res['meta']['dataset_repo']}`",
        f"- Packs: {', '.join('`'+p+'`' for p in res['meta']['pack_files'])}",
        f"- Base: `{res['meta']['base_model_id']}`",
        f"- Pruned: `{res['meta']['pruned_model_id']}`",
        f"- top_k (forced for both): {int(res['meta']['top_k'])}",
        f"- blocks per pack: {int(res['meta']['num_blocks'])} | batch_size: {int(res['meta']['batch_size'])}",
        "",
        "| pack | base ppl1024 | pruned ppl1024 | delta | base ppl2048 | pruned ppl2048 | delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in res["results"]:
        lines.append(
            f"| {r['pack']} | {r['base_ppl_1024']:.3f} | {r['pruned_ppl_1024']:.3f} | {r['delta_1024']:+.3f} | "
            f"{r['base_ppl_2048']:.3f} | {r['pruned_ppl_2048']:.3f} | {r['delta_2048']:+.3f} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- Completion-only PPL on Harmony assistant spans, packed into blocks of seq_len+1.",
        "- Each pack is evaluated independently; UNION is a round-robin interleaving across packs (not dominated by the first pack).",
        "- top_k is forced to keep routing consistent across models.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "modal run modal/eval_calib_packs_ppl_parity.py \\",
        f"  --dataset-repo {dataset_repo} \\",
        f"  --pack-files-csv {pack_files_csv} \\",
        f"  --base-model-id {base_model_id} \\",
        f"  --pruned-model-id {pruned_model_id} \\",
        f"  --top-k {int(top_k)} --num-blocks {int(num_blocks)} --batch-size {int(batch_size)}",
        "```",
        "",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[+] Wrote {out_path}")
