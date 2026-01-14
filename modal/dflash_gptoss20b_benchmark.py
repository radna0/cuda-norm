"""
Benchmark GPT-OSS-20B DFlash speculative decoding vs target-only decoding.

This measures:
- wall time per output token (includes prefill, since both include it)
- speedup ratio
- acceptance length stats for DFlash

Run (H100; logs to unsloth_logs/):
  mkdir -p harmony/cuda-norm/unsloth_logs
  ts=$(date +%Y%m%d_%H%M%S)
  nohup env MODAL_PROFILE=phamtrinhkien1203 \
    modal run harmony/cuda-norm/modal/dflash_gptoss20b_benchmark.py \
      --ckpt-dir /root/model/dflash_gptoss20b/20260114_140507/step_000005 \
      --max-samples 16 --max-new-tokens 256 --block-size 8 \
    > harmony/cuda-norm/unsloth_logs/dflash_bench_${ts}.log 2>&1 &
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import modal


APP_NAME = "dflash-gptoss20b-benchmark"
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

image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.11")
    .apt_install("git", "python3-dev", "build-essential", "curl")
    .run_commands("python -m pip install -U pip setuptools wheel")
    .run_commands(
        "python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128",
        "python -m pip install transformers==4.56.2 tokenizers safetensors accelerate datasets pyarrow",
        "python -m pip install huggingface-hub==0.36.0 hf-transfer",
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


def _download_dataset_file(dataset_repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    return Path(
        hf_hub_download(
            repo_id=str(dataset_repo),
            repo_type="dataset",
            filename=str(filename),
            token=token,
        )
    )


def _iter_parquet_texts(parquet_path: Path, *, text_column: str = "text"):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(parquet_path))
    for rg in range(pf.num_row_groups):
        tab = pf.read_row_group(rg, columns=[text_column])
        col = tab.column(text_column)
        for v in col.to_pylist():
            if isinstance(v, str) and v.strip():
                yield v


def _ensure_mask_token(tok, *, target_model) -> int:
    if tok.mask_token_id is None:
        tok.add_special_tokens({"mask_token": "<|MASK|>"})
        target_model.resize_token_embeddings(len(tok))
    return int(tok.mask_token_id)


def _cuda_time() -> float:
    import torch

    torch.cuda.synchronize()
    return time.perf_counter()


@app.function(
    image=image,
    gpu="H100:1",
    timeout=21600,
    cpu=12.0,
    memory=262144,
    volumes={"/root/model": model_volume, "/root/hf_cache": hf_cache_volume},
    secrets=_secrets,
)
def bench_remote(
    *,
    ckpt_dir: str,
    dataset_repo: str,
    dataset_file: str,
    max_samples: int,
    max_new_tokens: int,
    block_size: int,
    temperature: float,
    seed: int,
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

    model_id = "openai/gpt-oss-20b"
    tok = AutoTokenizer.from_pretrained(model_id)
    target = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    _ensure_mask_token(tok, target_model=target)

    ckpt_dir_path = Path(str(ckpt_dir))
    if not ckpt_dir_path.exists():
        raise FileNotFoundError(f"ckpt_dir not found: {ckpt_dir_path}")
    draft = GptOssDFlashDraftModel.from_pretrained(str(ckpt_dir_path), torch_dtype=target.dtype).to(target.device).eval()

    @torch.inference_mode()
    def _target_decode_baseline(*, input_ids):
        from transformers import DynamicCache

        # Apples-to-apples baseline vs our speculative loop: explicit KV-cache
        # decoding rather than `generate()` (which adds extra overhead).
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

        return out_ids

    # Load prompts
    path = _download_dataset_file(dataset_repo, dataset_file)
    texts = list(_iter_parquet_texts(path))
    random.shuffle(texts)
    texts = texts[: int(max_samples)]
    if not texts:
        raise RuntimeError("No prompts found")

    results = []
    acc_all: list[int] = []

    for i, text in enumerate(texts):
        # Keep prompt short-ish so we are mostly measuring decode.
        prompt = text
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = inp["input_ids"].to(target.device)
        attention_mask = inp.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(target.device)

        # Baseline
        t0 = _cuda_time()
        out_base = _target_decode_baseline(input_ids=input_ids)
        t1 = _cuda_time()
        base_new = int(out_base.shape[1] - input_ids.shape[1])
        base_tpt = (t1 - t0) / max(1, base_new)
        del out_base
        torch.cuda.empty_cache()

        # DFlash
        t2 = _cuda_time()
        out_spec, stats = dflash_spec_generate(
            draft_model=draft,
            target_model=target,
            tokenizer=tok,
            input_ids=input_ids,
            max_new_tokens=int(max_new_tokens),
            block_size=int(block_size),
            temperature=float(temperature),
            stop_token_ids=[tok.eos_token_id] if tok.eos_token_id is not None else None,
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

    base_tpt_mean = sum(r["base_time_per_token_s"] for r in results) / len(results)
    spec_tpt_mean = sum(r["spec_time_per_token_s"] for r in results) / len(results)
    speedup = base_tpt_mean / max(1e-9, spec_tpt_mean)

    hist = {}
    for a in acc_all:
        hist[str(int(a))] = hist.get(str(int(a)), 0) + 1

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("/root/model") / "dflash_bench" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bench.json"
    out_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "target_model": model_id,
                "ckpt_dir": str(ckpt_dir_path),
                "dataset_repo": dataset_repo,
                "dataset_file": dataset_file,
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
    dataset_repo: str = "radna0/harmony-qwen3-calib-packs-v2-20260113",
    dataset_file: str = "packs/calib_prompt_10000_v2/calib_prompt_10000_v2.parquet",
    max_samples: int = 16,
    max_new_tokens: int = 256,
    block_size: int = 8,
    temperature: float = 0.0,
    seed: int = 0,
):
    print(
        bench_remote.remote(
            ckpt_dir=str(ckpt_dir),
            dataset_repo=str(dataset_repo),
            dataset_file=str(dataset_file),
            max_samples=int(max_samples),
            max_new_tokens=int(max_new_tokens),
            block_size=int(block_size),
            temperature=float(temperature),
            seed=int(seed),
        )
    )
