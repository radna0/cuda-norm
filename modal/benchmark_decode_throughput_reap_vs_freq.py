"""
Decode-first throughput benchmark: base vs frequency-pruned vs REAP-pruned (GPT-OSS-20B).

This intentionally focuses on generation decode throughput at increasing batch sizes:
- prompt_len: small (128/256) to emphasize decode phase
- new_tokens: long (2048 default, optionally 8192 stress)
- batch sweep: 1,2,4,... until OOM

Variants benchmarked:
1) base, top_k=4
2) base, top_k=2
3) freq 50%, top_k=4
4) reap 50%, top_k=4
5) reap 50%, top_k=2
6) reap 25%, top_k=2 (optional)

Output:
- `reports/20b_decode_throughput_reap_vs_freq.md`

Run (always log to unsloth_logs/):
  mkdir -p unsloth_logs
  ts=$(date +%Y%m%d_%H%M%S)
  nohup modal run modal/benchmark_decode_throughput_reap_vs_freq.py --prompt-len 256 --new-tokens 2048 --batch-sizes 1,2,4,8,16,32 \
    > "unsloth_logs/20b_decode_throughput_reap_vs_freq_${ts}.log" 2>&1 &
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "gpt-oss-20b-decode-throughput-reap-vs-freq"


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

FREQ50_DIR = "/root/model/artifacts/20b_pruned_models_freq/general_50pct_experts_freq"
FREQ25_DIR = "/root/model/artifacts/20b_pruned_models_freq/math_25pct_experts_freq"
REAP50_DIR = "/root/model/artifacts/20b_pruned_models_reap/general_50pct_experts_reap"
REAP25_DIR = "/root/model/artifacts/20b_pruned_models_reap/math_25pct_experts_reap"

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


def _iter_gpt_oss_layers(model) -> list[Any]:
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None) if base is not None else None
    if layers is None:
        raise RuntimeError("Could not locate GPT-OSS layers at model.model.layers")
    return list(layers)


def _apply_top_k(model, top_k: int) -> dict[str, Any]:
    # Set per-layer router.top_k and config fields so the MoE kernel actually routes fewer experts.
    applied = 0
    layers = _iter_gpt_oss_layers(model)
    for layer in layers:
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
def bench_decode_sweep(
    *,
    model_path: str,
    name: str,
    top_k: int,
    prompt_len: int,
    new_tokens: int,
    batch_sizes: list[int],
    warmup: int,
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
    top_k = int(top_k)
    warmup = int(warmup)
    batch_sizes = [int(x) for x in batch_sizes]
    batch_sizes = [x for x in batch_sizes if x > 0]
    if not batch_sizes:
        raise ValueError("batch_sizes must be non-empty")

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

    num_experts = int(getattr(model.config, "num_local_experts", 0) or 0)
    top_k = max(1, min(int(top_k), max(1, num_experts) if num_experts else int(top_k)))
    topk_meta = _apply_top_k(model, top_k=int(top_k))

    vocab = int(getattr(model.config, "vocab_size", 0) or 0)
    token_id = int(eos) if (0 <= int(eos) < max(1, vocab)) else 1
    if vocab > 2:
        token_id = min(token_id, vocab - 1)

    results: list[dict[str, Any]] = []

    def _oom(e: BaseException) -> bool:
        msg = str(e).lower()
        return "out of memory" in msg or "cuda error" in msg and "out of memory" in msg

    for bs in batch_sizes:
        bs = int(bs)
        try:
            input_ids = torch.full(
                (bs, prompt_len),
                fill_value=token_id,
                device="cuda",
                dtype=torch.long,
            )
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

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
                torch.cuda.synchronize()
                t0 = time.time()
                pkv = past_key_values
                nt = next_token
                for _ in range(int(new_tokens)):
                    out = model(input_ids=nt, past_key_values=pkv, use_cache=True)
                    pkv = getattr(out, "past_key_values", None)
                    if pkv is None:
                        raise RuntimeError("Missing past_key_values during decode.")
                    nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                torch.cuda.synchronize()
                return float(time.time() - t0)

            # Warmup for this batch size (compilation / caches).
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

            peak_alloc_gib = float(torch.cuda.max_memory_allocated() / (1024.0**3))
            total_decode_tok_s = float((bs * new_tokens) / max(1e-9, decode_s))
            per_stream_decode_tok_s = float(new_tokens / max(1e-9, decode_s))
            results.append(
                {
                    "batch_size": int(bs),
                    "prefill_s": float(prefill_s),
                    "decode_s": float(decode_s),
                    "per_stream_decode_tok_s": float(per_stream_decode_tok_s),
                    "total_decode_tok_s": float(total_decode_tok_s),
                    "peak_alloc_gib": float(peak_alloc_gib),
                }
            )
        except Exception as e:
            if _oom(e):
                results.append(
                    {
                        "batch_size": int(bs),
                        "oom": True,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                break
            raise

    return {
        "name": str(name),
        "model_path": str(model_path),
        "top_k": int(top_k),
        "num_experts": int(num_experts),
        "prompt_len": int(prompt_len),
        "new_tokens": int(new_tokens),
        "topk_meta": dict(topk_meta),
        "results": results,
    }


def _parse_csv_ints(s: str) -> list[int]:
    out: list[int] = []
    for part in (s or "").split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out


@app.local_entrypoint()
def main(
    prompt_len: int = 256,
    new_tokens: int = 2048,
    batch_sizes: str = "1,2,4,8,16,32",
    warmup: int = 1,
    include_reap_25_topk2: bool = False,
):
    batch_list = _parse_csv_ints(batch_sizes)
    out_path = Path("reports/20b_decode_throughput_reap_vs_freq.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    variants: list[tuple[str, str, int]] = [
        ("base_topk4", BASE_20B_MODEL_ID, 4),
        ("base_topk2", BASE_20B_MODEL_ID, 2),
        ("freq50_topk4", FREQ50_DIR, 4),
        ("reap50_topk4", REAP50_DIR, 4),
        ("reap50_topk2", REAP50_DIR, 2),
    ]
    if bool(include_reap_25_topk2):
        variants.append(("reap25_topk2", REAP25_DIR, 2))

    runs: list[dict[str, Any]] = []
    for name, path, top_k in variants:
        runs.append(
            bench_decode_sweep.remote(
                model_path=path,
                name=name,
                top_k=int(top_k),
                prompt_len=int(prompt_len),
                new_tokens=int(new_tokens),
                batch_sizes=batch_list,
                warmup=int(warmup),
            )
        )

    # Summarize results.
    lines: list[str] = [
        "# 20B decode throughput: REAP vs frequency",
        "",
        f"- prompt_len: {int(prompt_len)} | new_tokens: {int(new_tokens)}",
        f"- batch_sweep: {','.join(str(x) for x in batch_list)}",
        "",
        "| run | top_k | max_batch | total tok/s @max | per-stream tok/s @max | peak_alloc_gib @max | model |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in runs:
        ok = [x for x in r["results"] if not x.get("oom")]
        if not ok:
            lines.append(
                f"| {r['name']} | {r['top_k']} | 0 | 0 | 0 | 0 | `{r['model_path']}` |"
            )
            continue
        best = max(ok, key=lambda x: x.get("batch_size", 0))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["name"]),
                    str(r["top_k"]),
                    str(best["batch_size"]),
                    f"{best['total_decode_tok_s']:.2f}",
                    f"{best['per_stream_decode_tok_s']:.2f}",
                    f"{best['peak_alloc_gib']:.1f}",
                    f"`{r['model_path']}`",
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Per-batch details",
        "",
    ]
    for r in runs:
        lines.append(f"### {r['name']}")
        lines.append("")
        lines.append("| batch | total tok/s | per-stream tok/s | decode_s | peak_alloc_gib | status |")
        lines.append("|---:|---:|---:|---:|---:|---|")
        for x in r["results"]:
            if x.get("oom"):
                lines.append(f"| {x['batch_size']} |  |  |  |  | OOM |")
                break
            lines.append(
                f"| {x['batch_size']} | {x['total_decode_tok_s']:.2f} | {x['per_stream_decode_tok_s']:.2f} | {x['decode_s']:.3f} | {x['peak_alloc_gib']:.1f} | ok |"
            )
        lines.append("")

    lines += [
        "## Reproduce",
        "",
        "```bash",
        "modal run modal/benchmark_decode_throughput_reap_vs_freq.py --prompt-len 256 --new-tokens 2048 --batch-sizes 1,2,4,8,16,32",
        "```",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[+] Wrote {out_path}")
