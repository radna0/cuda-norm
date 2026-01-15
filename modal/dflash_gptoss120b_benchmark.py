"""
Benchmark GPT-OSS-120B DFlash speculative decoding vs target-only decoding on Modal B200.

This script is intentionally strict:
  - Loads the 120B target in Transformers on GPU (requires MXFP4 support).
  - Loads a draft checkpoint produced by `dflash_gptoss120b_train.py`.
  - Measures:
      * decode wall time per output token (baseline and DFlash)
      * acceptance length histogram

Notes:
  - Uses `pad_token_id` as mask token when no dedicated mask token exists.
  - Requires MXFP4 kernels to avoid dequantizing to bf16 (which would OOM for 120B).
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import modal


APP_NAME = "dflash-gptoss120b-benchmark"
BASE_IMAGE = "nvidia/cuda:12.8.0-devel-ubuntu24.04"
_repo_root = Path(__file__).resolve().parents[1]


def _maybe_load_repo_dotenv() -> None:
    try:
        dotenv_path = _repo_root / ".env"
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

model_volume = modal.Volume.from_name("dflash-models", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("hf-cache-persistent", create_if_missing=True)

_secrets: list[modal.Secret] = []
if os.environ.get("HF_TOKEN"):
    _secrets.append(modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}))

cpu_image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.11")
    .apt_install("git", "python3-dev", "build-essential", "curl")
    .run_commands("python -m pip install -U pip setuptools wheel")
    .run_commands("python -m pip install huggingface-hub==0.36.0 hf-transfer")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/root/hf_cache",
            "TRANSFORMERS_CACHE": "/root/hf_cache/transformers",
            "HF_DATASETS_CACHE": "/root/hf_cache/datasets",
            "HUGGINGFACE_HUB_CACHE": "/root/hf_cache/hub",
        }
    )
)

bench_image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.11")
    .apt_install("git", "python3-dev", "build-essential", "curl")
    .run_commands("python -m pip install -U pip setuptools wheel")
    .run_commands(
        "python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128",
        # Keep transformers pinned to match training for now.
        "python -m pip install transformers==4.56.2 tokenizers safetensors accelerate datasets pyarrow",
        "python -m pip install huggingface-hub==0.36.0 hf-transfer",
        # Enable MXFP4 kernels so GPT-OSS-120B doesn't dequantize to bf16.
        "python -m pip install kernels==0.11.7",
    )
    .add_local_dir(str(_repo_root / "dflash_gptoss"), remote_path="/root/dflash_gptoss", copy=True)
    .env(
        {
            "PYTHONPATH": "/root",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/root/hf_cache",
            "TRANSFORMERS_CACHE": "/root/hf_cache/transformers",
            "HF_DATASETS_CACHE": "/root/hf_cache/datasets",
            "HUGGINGFACE_HUB_CACHE": "/root/hf_cache/hub",
        }
    )
)

app = modal.App(APP_NAME)

def _parse_bool(v: object, *, default: bool = False) -> bool:
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)


@app.function(
    image=cpu_image,
    timeout=21600,
    cpu=8.0,
    memory=65536,
    volumes={"/root/hf_cache": hf_cache_volume},
    secrets=_secrets,
)
def predownload_remote(*, model_id: str) -> str:
    from huggingface_hub import snapshot_download

    hf_cache_volume.reload()
    token = os.environ.get("HF_TOKEN")
    snapshot_download(
        repo_id=str(model_id),
        repo_type="model",
        token=token,
        local_files_only=False,
        max_workers=16,
    )
    hf_cache_volume.commit()
    return "ok"


def _resolve_mask_token_id(tok) -> int:
    if tok.mask_token_id is not None:
        return int(tok.mask_token_id)
    if tok.pad_token_id is not None:
        return int(tok.pad_token_id)
    raise ValueError("Tokenizer has no mask_token_id and no pad_token_id; cannot run DFlash without resizing vocab.")


def _cuda_time() -> float:
    import torch

    torch.cuda.synchronize()
    return time.perf_counter()


@app.function(
    image=bench_image,
    gpu="B200:1",
    timeout=21600,
    cpu=12.0,
    memory=524288,
    volumes={"/root/model": model_volume, "/root/hf_cache": hf_cache_volume},
    secrets=_secrets,
)
def bench_remote(
    *,
    ckpt_dir: str,
    model_id: str,
    max_samples: int,
    max_new_tokens: int,
    block_size: int,
    temperature: float,
    seed: int,
    stop_on_eos: bool,
) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dflash_gptoss.modeling_gptoss_dflash import GptOssDFlashDraftModel
    from dflash_gptoss.spec_decode import dflash_spec_generate
    from dflash_gptoss.spec_decode import sample_logits

    model_volume.reload()
    hf_cache_volume.reload()

    random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))

    tok = AutoTokenizer.from_pretrained(model_id)
    eos_id = tok.eos_token_id
    _resolve_mask_token_id(tok)

    # IMPORTANT: use dtype="auto" so MXFP4 stays quantized when kernels are present.
    target = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0", torch_dtype="auto").eval()

    ckpt_dir_path = Path(str(ckpt_dir))
    if not ckpt_dir_path.exists():
        raise FileNotFoundError(f"ckpt_dir not found: {ckpt_dir_path}")
    draft = GptOssDFlashDraftModel.from_pretrained(str(ckpt_dir_path), torch_dtype=torch.bfloat16).to(target.device).eval()

    @torch.inference_mode()
    def _target_decode_baseline(*, input_ids):
        from transformers import DynamicCache

        cache = DynamicCache()
        input_len = int(input_ids.shape[1])
        out_len = input_len + int(max_new_tokens)
        out_ids = torch.empty((int(input_ids.shape[0]), out_len), dtype=torch.long, device=target.device)
        out_ids[:, :input_len] = input_ids
        position_ids = torch.arange(out_len, device=target.device).unsqueeze(0)

        out = target(
            input_ids=input_ids,
            position_ids=position_ids[:, :input_len],
            past_key_values=cache,
            use_cache=True,
        )
        next_token = sample_logits(out.logits[:, -1, :], temperature=float(temperature)).to(torch.long)
        out_ids[:, input_len] = next_token

        if bool(stop_on_eos) and eos_id is not None and int(next_token[0].item()) == int(eos_id):
            return out_ids[:, : input_len + 1]

        for step in range(int(max_new_tokens) - 1):
            pos = input_len + step + 1
            out = target(
                input_ids=next_token[:, None],
                position_ids=position_ids[:, pos : pos + 1],
                past_key_values=cache,
                use_cache=True,
            )
            next_token = sample_logits(out.logits[:, -1, :], temperature=float(temperature)).to(torch.long)
            out_ids[:, pos] = next_token
            if bool(stop_on_eos) and eos_id is not None and int(next_token[0].item()) == int(eos_id):
                return out_ids[:, : pos + 1]

        return out_ids

    # Prompts: we intentionally keep it simple and deterministic for strict step comparisons.
    prompts = [
        "You are a tool-using assistant. Solve the user request and output only JSON tool calls.\nUser: compute 127*913 and return {\"result\": <int>}.",
        "User: write a short proof sketch of why the harmonic series diverges.",
        "User: given a python function signature, implement it. signature: def is_prime(n:int)->bool",
        "User: plan a 3-step agentic workflow to scrape a webpage safely and summarize it.",
        "User: solve: If f(x)=x^2+3x+2, find f(10).",
        "User: translate to French: 'The cat sits on the mat.'",
        "User: provide a concise explanation of backpropagation.",
        "User: what is 2+2?",
    ]
    prompts = prompts[: int(max_samples)]

    results = []
    acc_all: list[int] = []

    for i, prompt in enumerate(prompts):
        print(f"[sample {i+1}/{len(prompts)}] start", flush=True)
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = inp["input_ids"].to(target.device)

        t0 = _cuda_time()
        out_base = _target_decode_baseline(input_ids=input_ids)
        t1 = _cuda_time()
        base_new = int(out_base.shape[1] - input_ids.shape[1])
        base_tpt = (t1 - t0) / max(1, base_new)
        del out_base
        torch.cuda.empty_cache()

        t2 = _cuda_time()
        out_spec, stats = dflash_spec_generate(
            draft_model=draft,
            target_model=target,
            tokenizer=tok,
            input_ids=input_ids,
            max_new_tokens=int(max_new_tokens),
            block_size=int(block_size),
            temperature=float(temperature),
            stop_token_ids=[int(eos_id)] if (bool(stop_on_eos) and eos_id is not None) else None,
        )
        t3 = _cuda_time()
        spec_new = int(out_spec.shape[1] - input_ids.shape[1])
        spec_tpt = (t3 - t2) / max(1, spec_new)

        acc = stats.acceptance_lengths
        acc_all.extend(acc)
        results.append(
            {
                "i": i,
                "base_new_tokens": base_new,
                "spec_new_tokens": spec_new,
                "base_time_s": float(t1 - t0),
                "spec_time_s": float(t3 - t2),
                "base_time_per_token_s": float(base_tpt),
                "spec_time_per_token_s": float(spec_tpt),
                "speedup": float(base_tpt / max(1e-9, spec_tpt)),
                "acceptance_mean": float(sum(acc) / max(1, len(acc))),
            }
        )
        print(
            f"[sample {i+1}/{len(prompts)}] speedup={results[-1]['speedup']:.3f} "
            f"acc_mean={results[-1]['acceptance_mean']:.2f}",
            flush=True,
        )

    base_tpt_mean = sum(r["base_time_per_token_s"] for r in results) / len(results)
    spec_tpt_mean = sum(r["spec_time_per_token_s"] for r in results) / len(results)
    speedup = base_tpt_mean / max(1e-9, spec_tpt_mean)

    hist = {}
    for a in acc_all:
        hist[str(int(a))] = hist.get(str(int(a)), 0) + 1

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("/root/model") / "dflash_bench_120b" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bench.json"
    out_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "target_model": model_id,
                "ckpt_dir": str(ckpt_dir_path),
                "max_samples": int(max_samples),
                "max_new_tokens": int(max_new_tokens),
                "block_size": int(block_size),
                "temperature": float(temperature),
                "base_time_per_token_s_mean": float(base_tpt_mean),
                "spec_time_per_token_s_mean": float(spec_tpt_mean),
                "speedup_mean": float(speedup),
                "acceptance_hist": hist,
                "rows": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    model_volume.commit()
    return str(out_path)


@app.local_entrypoint()
def main(
    ckpt_dir: str,
    model_id: str = "openai/gpt-oss-120b",
    max_samples: int = 8,
    max_new_tokens: int = 2048,
    block_size: int = 16,
    temperature: float = 0.0,
    seed: int = 0,
    stop_on_eos: str = "false",
    # Accept strings so callers can pass `--predownload-only true/false`.
    predownload_only: str = "false",
):
    predownload_remote.remote(model_id=str(model_id))
    if _parse_bool(predownload_only, default=False):
        print("predownload ok")
        return
    print(
        bench_remote.remote(
            ckpt_dir=str(ckpt_dir),
            model_id=str(model_id),
            max_samples=int(max_samples),
            max_new_tokens=int(max_new_tokens),
            block_size=int(block_size),
            temperature=float(temperature),
            seed=int(seed),
            stop_on_eos=_parse_bool(stop_on_eos, default=False),
        )
    )
