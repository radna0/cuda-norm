"""
Decode-throughput benchmark for GPT-OSS-20B (base + structurally pruned variants).

Why:
- PPL scripts report `tok/s(pred)` which is *prefill scoring throughput* (next-token
  prediction for every position). It is not generation decode throughput.
- This benchmark measures:
  - prefill tokens/s (prompt_len / prefill_time)
  - decode tokens/s (new_tokens / decode_time) using KV-cache

Optional:
- Apply a "soft prune" routing policy (inference-only) by overriding MoE routing
  `top_k` (experts-per-token). This is the lever that can meaningfully improve
  decode tok/s; structural pruning alone keeps `top_k=4` so it mainly helps
  prefill and memory, not decode.

Models benchmarked:
- base: `openai/gpt-oss-20b`
- structural prunes (from pruning model volume):
  - /root/model/artifacts/20b_pruned_models/general_50pct_experts
  - /root/model/artifacts/20b_pruned_models/math_25pct_experts

Output:
- Writes `reports/20b_decode_throughput.md` (local entrypoint)

Run (log to unsloth_logs/):
  mkdir -p unsloth_logs
  ts=$(date +%Y%m%d_%H%M%S)
  nohup modal run modal/benchmark_decode_throughput_20b.py --prompt-len 1024 --new-tokens 128 --batch-size 1 \
    > "unsloth_logs/20b_decode_throughput_${ts}.log" 2>&1 &
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "gpt-oss-20b-decode-throughput"


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

BASE_20B_MODEL_ID = os.environ.get("MODEL_ID_20B", "openai/gpt-oss-20b")
GENERAL_PRUNED_DIR = "/root/model/artifacts/20b_pruned_models/general_50pct_experts"
MATH_PRUNED_DIR = "/root/model/artifacts/20b_pruned_models/math_25pct_experts"

_secrets: list[modal.Secret] = []
if os.environ.get("HF_TOKEN"):
    _secrets.append(modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}))

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
        "numpy==2.2.0 accelerate==1.10.1 "
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
    for p in ("/root/hf_cache", "/root/hf_cache/.cache", "/root/model"):
        try:
            Path(p).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


def _get_hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    return tok.strip() if tok else None


def _iter_gpt_oss_layers(model) -> list[Any]:
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None) if base is not None else None
    if layers is None:
        raise RuntimeError("Could not locate GPT-OSS layers at model.model.layers")
    return list(layers)


def _patch_router_soft_top_k(*, router, top_k: int, num_experts: int):
    """
    Patch router.forward to enforce a smaller top_k at inference time.
    Works for GPT-OSS routers that return (scores, indices) or (indices, scores).
    """
    import torch

    orig_forward = router.forward

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

        # Normalize shape to [N, num_experts] for top-k selection.
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

        k = int(top_k)
        k = max(1, min(k, int(num_experts)))
        idx_view = torch.topk(scores_view, k=k, dim=-1).indices

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
            return scores, idx_out
        return idx_out, scores

    return orig_forward, _forward


@app.function(
    image=image,
    gpu="H100:1",
    timeout=21600,
    cpu=16.0,
    memory=262144,
    volumes={
        "/root/model": model_volume,
        "/root/hf_cache": hf_cache_volume,
    },
    secrets=_secrets,
)
def bench_decode_one(
    *,
    model_path: str,
    prompt_len: int,
    new_tokens: int,
    batch_size: int,
    warmup: int,
    soft_top_k: int = 0,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _ensure_hf_env()
    try:
        model_volume.reload()
        hf_cache_volume.reload()
    except Exception:
        pass

    prompt_len = int(prompt_len)
    new_tokens = int(new_tokens)
    batch_size = int(batch_size)
    warmup = int(warmup)
    if prompt_len <= 0 or new_tokens <= 0 or batch_size <= 0:
        raise ValueError("prompt_len/new_tokens/batch_size must be > 0")

    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    eos = tok.eos_token_id
    if eos is None:
        raise RuntimeError("Tokenizer missing eos_token_id")

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype="auto",
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()

    # Synthetic prompt with fixed token length (avoid padding variability).
    vocab = int(getattr(model.config, "vocab_size", 0) or 0)
    token_id = int(eos) if (0 <= int(eos) < max(1, vocab)) else 1
    if vocab > 2:
        token_id = min(token_id, vocab - 1)
    input_ids = torch.full(
        (batch_size, prompt_len),
        fill_value=token_id,
        device="cuda",
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    num_experts = int(getattr(model.config, "num_local_experts", 0) or 0)
    cfg_top_k = int(
        getattr(model.config, "num_experts_per_tok", 0) or getattr(model.config, "experts_per_token", 4)
    )

    applied_top_k = int(cfg_top_k)
    if int(soft_top_k) > 0 and num_experts > 0:
        applied_top_k = max(1, min(int(soft_top_k), int(num_experts)))
        try:
            layers = _iter_gpt_oss_layers(model)
            for layer in layers:
                router = getattr(getattr(layer, "mlp", None), "router", None)
                if router is None:
                    continue
                orig_fwd, patched_fwd = _patch_router_soft_top_k(
                    router=router,
                    top_k=int(applied_top_k),
                    num_experts=int(num_experts),
                )
                router.forward = patched_fwd
                # Best-effort: update attributes some implementations read.
                for attr in ("top_k", "num_experts_per_tok"):
                    try:
                        setattr(router, attr, int(applied_top_k))
                    except Exception:
                        pass
        except Exception:
            pass

    def _prefill_once() -> tuple[Any, torch.Tensor, float]:
        torch.cuda.synchronize()
        t0 = time.time()
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        torch.cuda.synchronize()
        dt = time.time() - t0
        pkv = getattr(out, "past_key_values", None)
        if pkv is None:
            raise RuntimeError("Model did not return past_key_values; cannot benchmark decode separately.")
        logits_last = out.logits[:, -1, :]
        return pkv, logits_last, float(dt)

    @torch.inference_mode()
    def _decode_loop(past_key_values: Any, next_token: torch.Tensor) -> float:
        # next_token: [bs, 1]
        torch.cuda.synchronize()
        t0 = time.time()
        pkv = past_key_values
        nt = next_token
        for _ in range(new_tokens):
            out = model(input_ids=nt, past_key_values=pkv, use_cache=True)
            pkv = getattr(out, "past_key_values", None)
            if pkv is None:
                raise RuntimeError("Missing past_key_values during decode.")
            nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        return float(time.time() - t0)

    # Warmup (compile/caches/etc)
    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            pkv, logits_last, _ = _prefill_once()
            nt = logits_last.argmax(dim=-1, keepdim=True)
            _ = _decode_loop(pkv, nt)

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        pkv, logits_last, prefill_s = _prefill_once()
        nt = logits_last.argmax(dim=-1, keepdim=True)
        decode_s = _decode_loop(pkv, nt)

    peak_gib = float(torch.cuda.max_memory_allocated() / (1024.0**3))
    prefill_tok_s = float((batch_size * prompt_len) / max(1e-9, prefill_s))
    decode_tok_s = float((batch_size * new_tokens) / max(1e-9, decode_s))

    return {
        "model_path": str(model_path),
        "dtype": str(next(model.parameters()).dtype),
        "num_experts": int(num_experts),
        "cfg_top_k": int(cfg_top_k),
        "applied_top_k": int(applied_top_k),
        "batch_size": int(batch_size),
        "prompt_len": int(prompt_len),
        "new_tokens": int(new_tokens),
        "prefill_s": float(prefill_s),
        "decode_s": float(decode_s),
        "prefill_tok_s": float(prefill_tok_s),
        "decode_tok_s": float(decode_tok_s),
        "peak_alloc_gib": float(peak_gib),
    }


@app.local_entrypoint()
def main(
    prompt_len: int = 1024,
    new_tokens: int = 128,
    batch_size: int = 1,
    warmup: int = 1,
    soft_top_k: int = 0,
    soft_top_k_values: str = "",
):
    variants = [
        ("base", BASE_20B_MODEL_ID),
        ("general_50pct_experts", GENERAL_PRUNED_DIR),
        ("math_25pct_experts", MATH_PRUNED_DIR),
    ]

    # Allow a small sweep in a single invocation to make comparisons easier.
    topk_values: list[int] = []
    if str(soft_top_k_values).strip():
        for part in str(soft_top_k_values).split(","):
            p = part.strip()
            if not p:
                continue
            topk_values.append(int(p))
    else:
        topk_values = [int(soft_top_k)]
    if not topk_values:
        topk_values = [int(soft_top_k)]

    repo_root = Path(__file__).resolve().parent.parent
    out = repo_root / "reports/20b_decode_throughput.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# 20B decode throughput (prefill vs decode)",
        "",
        f"- prompt_len: {int(prompt_len)} | new_tokens: {int(new_tokens)} | batch_size: {int(batch_size)}",
        "",
    ]

    for topk in topk_values:
        results: list[tuple[str, dict[str, Any]]] = []
        for name, path in variants:
            results.append(
                (
                    name,
                    bench_decode_one.remote(
                        model_path=path,
                        prompt_len=int(prompt_len),
                        new_tokens=int(new_tokens),
                        batch_size=int(batch_size),
                        warmup=int(warmup),
                        soft_top_k=int(topk),
                    ),
                )
            )

        base = next((r for n, r in results if n == "base"), None)
        lines += [
            f"## soft_top_k={int(topk)}",
            "",
            "| model | prefill tok/s | decode tok/s | prefill_s | decode_s | peak_alloc_gib | experts | cfg_top_k | applied_top_k | path |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for name, r in results:
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        f"{r['prefill_tok_s']:.0f}",
                        f"{r['decode_tok_s']:.2f}",
                        f"{r['prefill_s']:.3f}",
                        f"{r['decode_s']:.3f}",
                        f"{r['peak_alloc_gib']:.1f}",
                        f"{r['num_experts']}",
                        f"{r['cfg_top_k']}",
                        f"{r['applied_top_k']}",
                        f"`{r['model_path']}`",
                    ]
                )
                + " |"
            )
        if base is not None:
            lines += [
                "",
                "Deltas vs base:",
                "",
            ]
            for name, r in results:
                if name == "base":
                    continue
                lines.append(
                    f"- {name}: decode_tok/s delta={(r['decode_tok_s']-base['decode_tok_s']):+.2f} "
                    f"({(r['decode_tok_s']/max(1e-9, base['decode_tok_s'])-1.0)*100:+.1f}%)"
                )
        lines.append("")

        print("[RESULT]", {"soft_top_k": int(topk), "results": results})

    lines += [
        "## Reproduce",
        "",
        "```bash",
        "modal run modal/benchmark_decode_throughput_20b.py --prompt-len 1024 --new-tokens 64 --batch-size 1 --soft-top-k-values 0,2",
        "```",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[+] Wrote {out}")
