"""
Smoke test GPT-OSS-20B DFlash (untrained draft) on Modal GPU.

This verifies:
- model loads
- draft forward runs
- speculative decode is lossless for greedy (temperature=0) vs target-only generation

Run (H100, logs to unsloth_logs/):
  mkdir -p harmony/cuda-norm/unsloth_logs
  ts=$(date +%Y%m%d_%H%M%S)
  nohup env MODAL_PROFILE=phamtrinhkien1203 \
    modal run harmony/cuda-norm/modal/dflash_gptoss20b_smoke.py \
      --max-new-tokens 128 --block-size 8 \
    > harmony/cuda-norm/unsloth_logs/dflash_smoke_${ts}.log 2>&1 &
"""

from __future__ import annotations

import os
from pathlib import Path

import modal


APP_NAME = "dflash-gptoss20b-smoke"
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

image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.11")
    .apt_install("git", "python3-dev", "build-essential", "curl")
    .run_commands("python -m pip install -U pip setuptools wheel")
    .run_commands(
        "python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128",
        "python -m pip install transformers==4.56.2 tokenizers safetensors accelerate",
    )
    .add_local_dir(str(_repo_root / "dflash_gptoss"), remote_path="/root/dflash_gptoss", copy=True)
    .env(
        {
            "PYTHONPATH": "/root",
            "HF_HOME": "/root/hf_cache",
            "TRANSFORMERS_CACHE": "/root/hf_cache/transformers",
            "HF_DATASETS_CACHE": "/root/hf_cache/datasets",
            "HUGGINGFACE_HUB_CACHE": "/root/hf_cache/hub",
        }
    )
)

_secrets: list[modal.Secret] = []
if os.environ.get("HF_TOKEN"):
    _secrets.append(modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}))

app = modal.App(APP_NAME)


@app.function(image=image, gpu="H100:1", timeout=3600, cpu=8.0, secrets=_secrets)
def smoke(max_new_tokens: int, block_size: int) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dflash_gptoss.modeling_gptoss_dflash import GptOssDFlashDraftModel
    from dflash_gptoss.spec_decode import dflash_spec_generate

    model_id = "openai/gpt-oss-20b"
    tok = AutoTokenizer.from_pretrained(model_id)
    target = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()

    draft = GptOssDFlashDraftModel.from_target_config(
        target_model_id=model_id,
        target_config=target.config,
        num_hidden_layers=4,
        block_size=int(block_size),
        mlp_ratio=4.0,
    ).to(device=target.device, dtype=target.dtype).eval()

    prompt = "Write a short proof that sqrt(2) is irrational."
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(target.device)

    # Target-only greedy
    g0 = target.generate(input_ids=input_ids, max_new_tokens=int(max_new_tokens), do_sample=False)

    # Spec decode greedy (must match)
    g1, stats = dflash_spec_generate(
        draft_model=draft,
        target_model=target,
        tokenizer=tok,
        input_ids=input_ids,
        max_new_tokens=int(max_new_tokens),
        block_size=int(block_size),
        temperature=0.0,
        stop_token_ids=[tok.eos_token_id] if tok.eos_token_id is not None else None,
    )

    same = torch.equal(g0, g1)
    out = [
        f"target_only_tokens={g0.shape[1]}",
        f"spec_tokens={g1.shape[1]}",
        f"match={same}",
        f"acceptance_avg={sum(stats.acceptance_lengths)/max(1,len(stats.acceptance_lengths)):.2f}",
        f"steps={stats.total_steps}",
        f"text={tok.decode(g1[0], skip_special_tokens=False)[:1200]}",
    ]
    return "\\n".join(out)


@app.local_entrypoint()
def main(max_new_tokens: int = 128, block_size: int = 8):
    print(smoke.remote(max_new_tokens=int(max_new_tokens), block_size=int(block_size)))
