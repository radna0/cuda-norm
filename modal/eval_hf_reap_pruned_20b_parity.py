"""
Parity PPL eval: base GPT-OSS-20B vs HF REAP-pruned checkpoint.

This uses the same packing + completion-only masking rules as:
- `modal/eval_prune_quality_reap_vs_freq.py`
- `modal/verify_sglang_gptoss_transmla.py`

Target use-case:
- Compare `openai/gpt-oss-20b` vs `sandeshrajx/gpt-oss-20b-reap-0.5-mxfp4`
  under top_k=4 vs top_k=2 to see how PPL changes.

Run (always log to unsloth_logs/):
  mkdir -p unsloth_logs
  ts=$(date +%Y%m%d_%H%M%S)
  nohup env MODAL_PROFILE=locthaokien1201 modal run modal/eval_hf_reap_pruned_20b_parity.py \
    --pruned-model-id sandeshrajx/gpt-oss-20b-reap-0.5-mxfp4 \
    --num-blocks 64 --batch-size 1 \
    > "unsloth_logs/hf_reap_pruned_20b_parity_${ts}.log" 2>&1 &
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

import modal

APP_NAME = "eval-hf-reap-pruned-20b-parity"


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

DEFAULT_DATASET_ID = os.environ.get("DATASET_ID", "radna0/nemotron-math-v2-harmony-tools")
DEFAULT_DATASET_SPLIT = os.environ.get("DATASET_SPLIT", "high_part00")
DEFAULT_TEXT_COLUMN = os.environ.get("TEXT_COLUMN", "text")

BASE_20B_MODEL_ID = os.environ.get("MODEL_ID_20B", "openai/gpt-oss-20b")
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
        "numpy==2.2.0 datasets==3.2.0 accelerate==1.10.1 "
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
    if layers is None:
        return []
    return list(layers)


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


def _cfg_top_k(model) -> int | None:
    for k in ("num_experts_per_tok", "experts_per_token"):
        try:
            v = getattr(model.config, k, None)
            if v is not None:
                return int(v)
        except Exception:
            pass
    return None


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
def eval_parity_base_vs_pruned(
    *,
    dataset_id: str,
    dataset_split: str,
    text_column: str,
    pruned_model_id: str,
    seq_len: int,
    num_blocks: int,
    batch_size: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _ensure_hf_env()
    try:
        model_volume.reload()
        hf_cache_volume.reload()
        data_volume.reload()
    except Exception:
        pass

    seq_len = int(seq_len)
    num_blocks = int(num_blocks)
    batch_size = int(max(1, batch_size))

    base_dir = _snapshot_download_model(BASE_20B_MODEL_ID)
    pruned_dir = _snapshot_download_model(str(pruned_model_id))

    tok = AutoTokenizer.from_pretrained(str(base_dir), trust_remote_code=True)
    if not getattr(tok, "is_fast", False):
        raise RuntimeError("Need a fast tokenizer for return_offsets_mapping=True")
    eos = tok.eos_token_id
    if eos is None:
        raise RuntimeError("Tokenizer missing eos_token_id")

    def text_iter() -> Iterable[str]:
        ds = load_dataset(str(dataset_id), split=str(dataset_split), streaming=True)
        for ex in ds:
            t = ex.get(text_column)
            if not isinstance(t, str) or not t.strip():
                continue
            yield t

    # ---- Pack once (token ids + completion-only keep mask).
    buf_ids: list[int] = []
    buf_keep: list[bool] = []
    blocks_ids: list[list[int]] = []
    blocks_keep: list[list[bool]] = []
    rows_seen = 0
    t_pack0 = time.time()
    for text in text_iter():
        rows_seen += 1
        spans = _assistant_content_spans(text)
        enc = tok(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
        ids: list[int] = enc["input_ids"]
        offs: list[tuple[int, int]] = enc["offset_mapping"]
        keep = _token_keep_mask(offs, spans)
        buf_ids.extend(ids)
        buf_keep.extend(keep)
        buf_ids.append(int(eos))
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
    pack_s = time.time() - t_pack0
    if len(blocks_ids) < num_blocks:
        raise RuntimeError(f"Only built {len(blocks_ids)}/{num_blocks} blocks (rows_seen={rows_seen}).")

    def ppl_for_loaded(model, *, force_top_k: int | None) -> dict[str, Any]:
        if force_top_k is not None:
            _ = _apply_top_k(model, top_k=int(force_top_k))
        model.eval()
        total_nll = 0.0
        kept_tokens = 0
        total_pred_tokens = 0
        t0 = time.time()
        for start in range(0, len(blocks_ids), batch_size):
            batch_ids = blocks_ids[start : start + batch_size]
            batch_keep = blocks_keep[start : start + batch_size]
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

    base_cfg_k = _cfg_top_k(base_model)
    pruned_cfg_k = _cfg_top_k(pruned_model)
    base_k4 = ppl_for_loaded(base_model, force_top_k=4)
    base_k2 = ppl_for_loaded(base_model, force_top_k=2)
    pruned_as_is = ppl_for_loaded(pruned_model, force_top_k=None)
    pruned_k4 = ppl_for_loaded(pruned_model, force_top_k=4)
    pruned_k2 = ppl_for_loaded(pruned_model, force_top_k=2)

    return {
        "meta": {
            "dataset_id": str(dataset_id),
            "dataset_split": str(dataset_split),
            "text_column": str(text_column),
            "seq_len": int(seq_len),
            "num_blocks": int(num_blocks),
            "batch_size": int(batch_size),
            "rows_seen": int(rows_seen),
            "pack_wall_s": float(pack_s),
            "base_model_id": str(BASE_20B_MODEL_ID),
            "pruned_model_id": str(pruned_model_id),
            "base_cfg_top_k": base_cfg_k,
            "pruned_cfg_top_k": pruned_cfg_k,
        },
        "models": {
            "base_topk4": {"path": str(base_dir), "top_k": 4, **base_k4},
            "base_topk2": {"path": str(base_dir), "top_k": 2, **base_k2},
            "pruned_as_is": {"path": str(pruned_dir), "top_k": pruned_cfg_k, **pruned_as_is},
            "pruned_topk4": {"path": str(pruned_dir), "top_k": 4, **pruned_k4},
            "pruned_topk2": {"path": str(pruned_dir), "top_k": 2, **pruned_k2},
        },
    }


@app.local_entrypoint()
def main(
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_split: str = DEFAULT_DATASET_SPLIT,
    text_column: str = DEFAULT_TEXT_COLUMN,
    pruned_model_id: str = DEFAULT_PRUNED_MODEL_ID,
    num_blocks: int = 64,
    batch_size: int = 1,
):
    out_path = Path("reports/20b_hf_reap_pruned_ppl_parity.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict[int, Any] = {}
    for seq_len in (1024, 2048):
        results[seq_len] = eval_parity_base_vs_pruned.remote(
            dataset_id=str(dataset_id),
            dataset_split=str(dataset_split),
            text_column=str(text_column),
            pruned_model_id=str(pruned_model_id),
            seq_len=int(seq_len),
            num_blocks=int(num_blocks),
            batch_size=int(batch_size),
        )

    lines: list[str] = [
        "# 20B HF REAP-pruned model: parity PPL",
        "",
        f"- Dataset: `{dataset_id}` split `{dataset_split}` col `{text_column}`",
        f"- Blocks: {int(num_blocks)} | Batch size: {int(batch_size)}",
        f"- Base: `{results[1024]['meta']['base_model_id']}` (cfg_top_k={results[1024]['meta']['base_cfg_top_k']})",
        f"- Pruned: `{results[1024]['meta']['pruned_model_id']}` (cfg_top_k={results[1024]['meta']['pruned_cfg_top_k']})",
        "",
        "| model | top_k | ppl1024 | ppl2048 | delta vs base_topk4 (1024/2048) |",
        "|---|---:|---:|---:|---:|",
    ]

    base1024 = float(results[1024]["models"]["base_topk4"]["ppl"])
    base2048 = float(results[2048]["models"]["base_topk4"]["ppl"])

    order = ["base_topk4", "base_topk2", "pruned_as_is", "pruned_topk2", "pruned_topk4"]
    for name in order:
        m1024 = results[1024]["models"][name]
        m2048 = results[2048]["models"][name]
        ppl1024 = float(m1024["ppl"])
        ppl2048 = float(m2048["ppl"])
        topk = m1024.get("top_k")
        d1 = ppl1024 - base1024
        d2 = ppl2048 - base2048
        lines.append(f"| {name} | {topk if topk is not None else ''} | {ppl1024:.3f} | {ppl2048:.3f} | {d1:+.3f} / {d2:+.3f} |")

    lines += [
        "",
        "## Notes",
        "",
        "- Completion-only PPL on Harmony assistant spans, packed into blocks of seq_len+1.",
        "- `pruned_as_is` uses the pruned model's config routing; `pruned_topk2/4` forces router.top_k at runtime.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "modal run modal/eval_hf_reap_pruned_20b_parity.py --pruned-model-id sandeshrajx/gpt-oss-20b-reap-0.5-mxfp4 --num-blocks 64 --batch-size 1",
        "```",
        "",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[+] Wrote {out_path}")

