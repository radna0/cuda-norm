"""
Soft-prune PPL parity harness (Transformers) for GPT-OSS-20B on Harmony text.

Why this exists:
- Our earlier soft-prune eval used per-row `out.loss` over *all tokens* and no packing,
  which is not comparable to the project's truth PPL harness.
- This script matches the reference methodology in
  `modal/verify_sglang_gptoss_transmla.py`:
    - concat examples + EOS
    - chunk into fixed blocks of (seq_len + 1) tokens
    - compute next-token NLL (logits at t predict token t+1)
    - apply completion-only masking (assistant message body spans only)

Deliverables (manager request):
- `reports/pruning_eval_parity.md` (rules)
- `reports/20b_soft_prune_eval_parity.md` (baseline + pruned ppl + delta)

Run (always log to unsloth_logs/ as per repo convention):
  mkdir -p unsloth_logs
  ts=$(date +%Y%m%d_%H%M%S)
  nohup modal run modal/eval_soft_prune_20b_parity.py \
    --seq-len 1024 --num-blocks 64 \
    --keep-fracs 1.0,0.5 --top-ks 4,2 \
    > "unsloth_logs/soft_prune_20b_parity_${ts}.log" 2>&1 &
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterable

import modal

APP_NAME = "eval-soft-prune-20b-parity"


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
DEFAULT_MODEL_ID_20B = os.environ.get("MODEL_ID_20B", "openai/gpt-oss-20b")

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
        "numpy==2.2.0 datasets==3.2.0 "
        "accelerate==1.10.1 "
        "transformers==4.56.2 tokenizers safetensors "
        "pyarrow==21.0.0 pandas==2.2.3 "
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


@dataclass(frozen=True)
class HarmonyMessage:
    role: str
    channel: str | None
    content_start: int
    content_end: int


START_TAG = "<|start|>"
CHANNEL_TAG = "<|channel|>"
MESSAGE_TAG = "<|message|>"
END_TAG = "<|end|>"
CALL_TAG = "<|call|>"
RETURN_TAG = "<|return|>"


def _parse_harmony(text: str) -> list[HarmonyMessage]:
    # Minimal parser copied from `harmony_text.py` behavior.
    i = 0
    n = len(text)
    out: list[HarmonyMessage] = []
    while i < n:
        start = text.find(START_TAG, i)
        if start < 0:
            break
        role_start = start + len(START_TAG)
        msg_tag_pos = text.find(MESSAGE_TAG, role_start)
        if msg_tag_pos < 0:
            raise ValueError("malformed harmony text: missing <|message|>")

        channel_pos = text.find(CHANNEL_TAG, role_start, msg_tag_pos)
        if channel_pos >= 0:
            role = text[role_start:channel_pos]
            channel = text[channel_pos + len(CHANNEL_TAG) : msg_tag_pos]
        else:
            role = text[role_start:msg_tag_pos]
            channel = None

        role = role.strip()
        channel = channel.strip() if channel is not None else None

        content_start = msg_tag_pos + len(MESSAGE_TAG)
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
            raise ValueError("malformed harmony text: missing <|end|>/<|call|>/<|return|>")
        delim_pos, delim_tag = min(candidates, key=lambda t: t[0])

        out.append(
            HarmonyMessage(
                role=role,
                channel=channel,
                content_start=content_start,
                content_end=delim_pos,
            )
        )
        i = delim_pos + len(delim_tag)
    return out


def _assistant_content_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for m in _parse_harmony(text):
        if not (m.role == "assistant" or m.role.startswith("assistant ")):
            continue
        spans.append((m.content_start, m.content_end))
    return spans


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


def _iter_gpt_oss_layers(model: Any) -> list[Any]:
    # GPT-OSS HF models typically expose `model.model.layers`.
    base = getattr(model, "model", None)
    if base is None:
        raise RuntimeError("Expected model.model to exist.")
    layers = getattr(base, "layers", None)
    if layers is None:
        raise RuntimeError("Expected model.model.layers to exist.")
    return list(layers)


def _get_router_bias(router: Any, *, torch_mod) -> Any | None:
    b = getattr(router, "bias", None)
    if torch_mod.is_tensor(b):
        return b
    linear = getattr(router, "linear", None)
    if linear is not None:
        bb = getattr(linear, "bias", None)
        if torch_mod.is_tensor(bb):
            return bb
    return None


def _apply_soft_prune(
    *,
    model: Any,
    keep_by_layer: list[list[int]] | None,
    top_k: int | None,
    torch_mod,
) -> dict[str, Any]:
    layers = _iter_gpt_oss_layers(model)
    num_layers = len(layers)
    num_experts = int(getattr(model.config, "num_local_experts", 0) or 0)
    if num_experts <= 0:
        raise RuntimeError("Could not determine num_local_experts from config.")
    if keep_by_layer is not None and len(keep_by_layer) != num_layers:
        raise ValueError(f"keep_by_layer has {len(keep_by_layer)} layers, expected {num_layers}")

    orig_bias_by_layer: dict[int, Any] = {}
    orig_cfg_top_k = int(getattr(model.config, "num_experts_per_tok", 4) or 4)
    for li, layer in enumerate(layers):
        router = getattr(getattr(layer, "mlp", None), "router", None)
        if router is None:
            raise RuntimeError(f"Layer {li} has no mlp.router")

        if top_k is not None:
            eff_top_k = int(max(1, min(int(top_k), num_experts)))
            for attr in ("top_k", "num_experts_per_tok"):
                try:
                    setattr(router, attr, int(eff_top_k))
                except Exception:
                    pass
            try:
                model.config.num_experts_per_tok = int(eff_top_k)
                model.config.experts_per_token = int(eff_top_k)
            except Exception:
                pass

        bias = _get_router_bias(router, torch_mod=torch_mod)
        if bias is None:
            continue
        if li not in orig_bias_by_layer:
            orig_bias_by_layer[li] = bias.detach().clone()
        with torch_mod.no_grad():
            bias.copy_(orig_bias_by_layer[li])
            if keep_by_layer is not None:
                allowed = torch_mod.zeros((num_experts,), dtype=torch_mod.bool, device=bias.device)
                for e in keep_by_layer[li]:
                    if 0 <= int(e) < num_experts:
                        allowed[int(e)] = True
                bias[~allowed] = bias[~allowed] - torch_mod.tensor(
                    1e9, device=bias.device, dtype=bias.dtype
                )

    return {"orig_bias_by_layer": orig_bias_by_layer, "orig_cfg_top_k": orig_cfg_top_k}


def _restore_soft_prune(model: Any, state: dict[str, Any], *, torch_mod) -> None:
    layers = _iter_gpt_oss_layers(model)
    orig_bias_by_layer: dict[int, Any] = state.get("orig_bias_by_layer") or {}
    for li, layer in enumerate(layers):
        router = getattr(getattr(layer, "mlp", None), "router", None)
        if router is None:
            continue
        bias = _get_router_bias(router, torch_mod=torch_mod)
        if bias is not None and li in orig_bias_by_layer:
            with torch_mod.no_grad():
                bias.copy_(orig_bias_by_layer[li])
    try:
        orig_cfg_top_k = int(state.get("orig_cfg_top_k") or 4)
        model.config.num_experts_per_tok = int(orig_cfg_top_k)
        model.config.experts_per_token = int(orig_cfg_top_k)
    except Exception:
        pass


def _parse_csv_floats(s: str) -> list[float]:
    out: list[float] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _parse_csv_ints(s: str) -> list[int]:
    out: list[int] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


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
def eval_soft_prune_ppl_parity(
    *,
    model_id: str,
    dataset_id: str,
    dataset_split: str,
    text_column: str,
    seq_len: int,
    num_blocks: int,
    batch_size: int,
    keep_fracs: list[float],
    top_ks: list[int],
    expert_ranking_by_layer_json: str | None,
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
    if seq_len <= 0 or num_blocks <= 0:
        raise ValueError("seq_len and num_blocks must be > 0")

    model_dir = _snapshot_download_model(model_id)
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if not getattr(tok, "is_fast", False):
        raise RuntimeError("Need a fast tokenizer for return_offsets_mapping=True")
    eos = tok.eos_token_id
    if eos is None:
        raise RuntimeError("Tokenizer missing eos_token_id")

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

    # Prepare ranking -> keep_by_layer for each keep_frac
    ranking: list[list[int]] | None = None
    if expert_ranking_by_layer_json:
        parsed = json.loads(expert_ranking_by_layer_json)
        if not isinstance(parsed, list) or len(parsed) != num_layers:
            raise ValueError(f"expert_ranking_by_layer_json must be a list of {num_layers} layers.")
        ranking = [[int(x) for x in layer] for layer in parsed]

    def build_keep_by_layer(keep_frac: float) -> list[list[int]] | None:
        if ranking is None:
            return None
        keep_n = int(max(1, min(num_experts, round(float(keep_frac) * num_experts))))
        out: list[list[int]] = []
        for li in range(num_layers):
            layer_rank = ranking[li]
            out.append([int(x) for x in layer_rank[:keep_n]])
        return out

    # Stream dataset and build packed blocks of (seq_len + 1) token ids + keep mask.
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

    t_pack0 = time.time()
    rows_seen = 0
    for text in text_iter():
        rows_seen += 1
        spans = _assistant_content_spans(text)
        enc = tok(
            text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
        ids: list[int] = enc["input_ids"]
        offs: list[tuple[int, int]] = enc["offset_mapping"]
        keep = _token_keep_mask(offs, spans)
        if len(ids) != len(keep):
            raise RuntimeError("Tokenizer returned mismatched input_ids and offset_mapping.")
        buf_ids.extend(ids)
        buf_keep.extend(keep)
        buf_ids.append(int(eos))
        buf_keep.append(False)
        while len(buf_ids) >= (seq_len + 1) and len(blocks_ids) < num_blocks:
            block_i = buf_ids[: seq_len + 1]
            block_k = buf_keep[: seq_len + 1]
            # No context for the first token in the block; exclude from loss.
            if block_k:
                block_k[0] = False
            blocks_ids.append(block_i)
            blocks_keep.append(block_k)
            del buf_ids[: seq_len + 1]
            del buf_keep[: seq_len + 1]
        if len(blocks_ids) >= num_blocks:
            break
    t_pack_s = time.time() - t_pack0

    if len(blocks_ids) < num_blocks:
        raise RuntimeError(f"Only built {len(blocks_ids)}/{num_blocks} blocks from dataset (rows_seen={rows_seen}).")

    def compute_ppl_for_current_model() -> dict[str, Any]:
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
            "total_nll": float(total_nll),
            "kept_tokens": int(kept_tokens),
            "pred_tokens": int(total_pred_tokens),
            "tokens_per_s_pred": float(total_pred_tokens / dt),
            "tokens_per_s_kept": float(kept_tokens / dt),
            "wall_s": float(dt),
        }

    results: list[dict[str, Any]] = []
    for keep_frac in keep_fracs:
        for top_k in top_ks:
            keep_by_layer = build_keep_by_layer(float(keep_frac))
            state = _apply_soft_prune(
                model=model,
                keep_by_layer=keep_by_layer,
                top_k=int(top_k),
                torch_mod=torch,
            )
            try:
                ppl_res = compute_ppl_for_current_model()
            finally:
                _restore_soft_prune(model, state, torch_mod=torch)

            keep_n = (
                int(max(1, min(num_experts, round(float(keep_frac) * num_experts))))
                if ranking is not None
                else num_experts
            )
            results.append(
                {
                    "keep_frac": float(keep_frac),
                    "keep_n": int(keep_n),
                    "top_k": int(top_k),
                    **ppl_res,
                }
            )

    return {
        "meta": {
            "model_id": str(model_id),
            "dataset_id": str(dataset_id),
            "dataset_split": str(dataset_split),
            "text_column": str(text_column),
            "seq_len": int(seq_len),
            "num_blocks": int(num_blocks),
            "rows_seen": int(rows_seen),
            "pack_wall_s": float(t_pack_s),
            "num_layers": int(num_layers),
            "num_experts": int(num_experts),
        },
        "results": results,
    }


@app.local_entrypoint()
def main(
    model_id: str = DEFAULT_MODEL_ID_20B,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_split: str = DEFAULT_DATASET_SPLIT,
    text_column: str = DEFAULT_TEXT_COLUMN,
    seq_len: int = 1024,
    num_blocks: int = 64,
    batch_size: int = 1,
    keep_fracs: str = "1.0,0.5",
    top_ks: str = "4,2",
    ranking_json_path: str = "third_party/GPT-OSS-MoE-ExpertFingerprinting/topical_analytics/all.json",
):
    """
    Runs baseline + pruned PPL on packed Harmony blocks, completion-only.

    `ranking_json_path` is a local file read by the entrypoint and passed to the
    remote worker. It should contain keys like "layer_0": [expert_ids...].
    """

    # Load ranking from the topical analytics repo (local-only file).
    expert_ranking_by_layer_json: str | None = None
    try:
        p = Path(ranking_json_path)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            # GPT-OSS-20B has 24 layers; keep it dynamic.
            layers: list[list[int]] = []
            i = 0
            while True:
                k = f"layer_{i}"
                if k not in data:
                    break
                layers.append([int(x) for x in data[k]])
                i += 1
            if layers:
                expert_ranking_by_layer_json = json.dumps(layers)
    except Exception:
        expert_ranking_by_layer_json = None

    res = eval_soft_prune_ppl_parity.remote(
        model_id=str(model_id),
        dataset_id=str(dataset_id),
        dataset_split=str(dataset_split),
        text_column=str(text_column),
        seq_len=int(seq_len),
        num_blocks=int(num_blocks),
        batch_size=int(batch_size),
        keep_fracs=_parse_csv_floats(keep_fracs),
        top_ks=_parse_csv_ints(top_ks),
        expert_ranking_by_layer_json=expert_ranking_by_layer_json,
    )

    meta = res["meta"]
    results = res["results"]
    baseline = None
    for r in results:
        if float(r["keep_frac"]) == 1.0 and int(r["top_k"]) == 4:
            baseline = r
            break

    out_rules = Path("reports/pruning_eval_parity.md")
    out_rules.parent.mkdir(parents=True, exist_ok=True)
    out_rules.write_text(
        "\n".join(
            [
                "# Pruning eval parity (PPL rules)",
                "",
                "This repo treats PPL as *next-token perplexity* computed over packed token blocks,",
                "and for Harmony chat data we compute **completion-only** loss:",
                "",
                "## Packing",
                "",
                "- Read Harmony-formatted examples from HF dataset `text`.",
                "- Tokenize with `add_special_tokens=False`.",
                "- Concatenate examples in-order, appending `eos_token_id` between examples.",
                "- Chunk into fixed blocks of length `seq_len+1` token ids.",
                "",
                "## Loss (next-token NLL)",
                "",
                "- For each block, compute logits on `input_ids[:-1]` and score targets `input_ids[1:]`.",
                "- Exclude the first token in each block (no context).",
                "",
                "## Completion-only masking",
                "",
                "- Parse Harmony tags and select **assistant message body spans** (not system/user/tool).",
                "- Build a per-token keep mask via `return_offsets_mapping=True` (token overlaps assistant span).",
                "- PPL is computed over *kept tokens only*: `ppl = exp(total_nll / kept_tokens)`.",
                "",
                "## Reference",
                "",
                "- The methodology matches the project's SGLang truth harness shape (packing + logprob NLL),",
                "  but this implementation uses Transformers for convenience with soft-prune patching.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    out_report = Path("reports/20b_soft_prune_eval_parity.md")
    out_report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 20B soft-prune eval (PPL parity)",
        "",
        f"- Model: `{meta['model_id']}`",
        f"- Dataset: `{meta['dataset_id']}` split `{meta['dataset_split']}` col `{meta['text_column']}`",
        f"- seq_len: {meta['seq_len']} | blocks: {meta['num_blocks']} | batch_size: {int(batch_size)}",
        f"- rows_seen: {meta['rows_seen']} | pack_wall_s: {meta['pack_wall_s']:.1f}s",
        "",
        "| keep_frac | keep_n | top_k | ppl | ppl_delta | kept_tokens | pred_tokens | tok/s(pred) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        delta = ""
        if baseline is not None:
            delta = f"{(float(r['ppl']) - float(baseline['ppl'])):+.4f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{float(r['keep_frac']):.2f}",
                    f"{int(r['keep_n'])}",
                    f"{int(r['top_k'])}",
                    f"{float(r['ppl']):.6f}",
                    delta,
                    f"{int(r['kept_tokens'])}",
                    f"{int(r['pred_tokens'])}",
                    f"{float(r['tokens_per_s_pred']):.0f}",
                ]
            )
            + " |"
        )
    lines.append("")
    if baseline is not None:
        lines.append(f"- Baseline (keep_frac=1.0, top_k=4) ppl={float(baseline['ppl']):.6f}")
    out_report.write_text("\n".join(lines), encoding="utf-8")

    print(f"[+] Wrote {out_rules}")
    print(f"[+] Wrote {out_report}")
    print("[RESULT]", res)
