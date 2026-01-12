"""
Parity PPL eval for structurally-pruned GPT-OSS-20B checkpoints (Transformers).

This uses the same packing + completion-only masking rules as:
- `reports/pruning_eval_parity.md`
- `modal/verify_sglang_gptoss_transmla.py` (shape: packed blocks of seq_len+1)

It evaluates 3 models:
- base: `openai/gpt-oss-20b`
- structural prunes (built in the pruning model volume):
  - /root/model/artifacts/20b_pruned_models/general_50pct_experts
  - /root/model/artifacts/20b_pruned_models/math_25pct_experts

Output:
- `reports/20b_structural_prune_ppl_parity.md`

Run (always log to unsloth_logs/):
  mkdir -p unsloth_logs
  ts=$(date +%Y%m%d_%H%M%S)
  nohup modal run modal/eval_structural_prune_20b_parity.py --seq-len 1024 --num-blocks 64 \
    > "unsloth_logs/20b_structural_prune_ppl_parity_${ts}.log" 2>&1 &
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from typing import Iterable

import modal

APP_NAME = "eval-structural-prune-20b-parity"


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
GENERAL_PRUNED_DIR = "/root/model/artifacts/20b_pruned_models/general_50pct_experts"
MATH_PRUNED_DIR = "/root/model/artifacts/20b_pruned_models/math_25pct_experts"

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
def eval_structural_prune_parity(
    *,
    dataset_id: str,
    dataset_split: str,
    text_column: str,
    seq_len: int,
    num_blocks: int,
    batch_size: int,
) -> dict[str, Any]:
    import math

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

    # Build packed blocks once (tokenizer offsets depend on tokenizer, but GPT-OSS
    # variants share tokenizer; we use base tokenizer to build the blocks).
    base_dir = _snapshot_download_model(BASE_20B_MODEL_ID)
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

    def ppl_for(model_path: str) -> dict[str, Any]:
        m = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map={"": 0},
            trust_remote_code=True,
        )
        m.eval()
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
                logits = m(input_ids[:, :-1], use_cache=False).logits
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

    base = ppl_for(str(base_dir))
    general = ppl_for(GENERAL_PRUNED_DIR)
    mathp = ppl_for(MATH_PRUNED_DIR)

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
        },
        "models": {
            "base": {"path": str(base_dir), **base},
            "general_50pct_experts": {"path": GENERAL_PRUNED_DIR, **general},
            "math_25pct_experts": {"path": MATH_PRUNED_DIR, **mathp},
        },
    }


@app.local_entrypoint()
def main(
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_split: str = DEFAULT_DATASET_SPLIT,
    text_column: str = DEFAULT_TEXT_COLUMN,
    seq_len: int = 1024,
    num_blocks: int = 64,
    batch_size: int = 1,
):
    res = eval_structural_prune_parity.remote(
        dataset_id=str(dataset_id),
        dataset_split=str(dataset_split),
        text_column=str(text_column),
        seq_len=int(seq_len),
        num_blocks=int(num_blocks),
        batch_size=int(batch_size),
    )

    meta = res["meta"]
    m = res["models"]
    base = m["base"]["ppl"]

    out = Path("reports/20b_structural_prune_ppl_parity.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(
            [
                "# 20B structural prune PPL (parity harness)",
                "",
                f"- Dataset: `{meta['dataset_id']}` split `{meta['dataset_split']}` col `{meta['text_column']}`",
                f"- seq_len: {meta['seq_len']} | blocks: {meta['num_blocks']} | batch_size: {meta['batch_size']}",
                f"- rows_seen: {meta['rows_seen']} | pack_wall_s: {meta['pack_wall_s']:.1f}s",
                "",
                "| model | ppl | ppl_delta | tok/s(pred) | path |",
                "|---|---:|---:|---:|---|",
                f"| base | {m['base']['ppl']:.6f} | {0.0:+.4f} | {m['base']['tok_s_pred']:.0f} | `{m['base']['path']}` |",
                f"| general_50pct_experts | {m['general_50pct_experts']['ppl']:.6f} | {(m['general_50pct_experts']['ppl']-base):+.4f} | {m['general_50pct_experts']['tok_s_pred']:.0f} | `{m['general_50pct_experts']['path']}` |",
                f"| math_25pct_experts | {m['math_25pct_experts']['ppl']:.6f} | {(m['math_25pct_experts']['ppl']-base):+.4f} | {m['math_25pct_experts']['tok_s_pred']:.0f} | `{m['math_25pct_experts']['path']}` |",
                "",
                "## Reproduce",
                "",
                "```bash",
                "modal run modal/eval_structural_prune_20b_parity.py --seq-len 1024 --num-blocks 64",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[+] Wrote {out}")
    print("[RESULT]", res)
