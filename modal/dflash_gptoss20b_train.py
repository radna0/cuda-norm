"""
Train a GPT-OSS-20B DFlash draft model (Phase A: 4K context) on Modal GPU.

This trains ONLY the draft model. The GPT-OSS-20B target is frozen and used to:
- produce hidden states for conditioning (teacher forcing)
- provide lm_head for logits (so vocab/logit space matches target)

Run (H100; logs to unsloth_logs/):
  mkdir -p harmony/cuda-norm/unsloth_logs
  ts=$(date +%Y%m%d_%H%M%S)
  nohup env MODAL_PROFILE=phamtrinhkien1203 \
    modal run harmony/cuda-norm/modal/dflash_gptoss20b_train.py \
      --max-steps 50 --seq-len 4096 --block-size 8 \
      --dataset-repo radna0/harmony-qwen3-calib-packs-v2-20260113 \
      --train-files-csv packs/reasoning_style_10k_v2/reasoning_style_10k_v2.parquet,tool_agentic_10k_v6.parquet,packs/calib_prompt_10000_v2/calib_prompt_10000_v2.parquet \
    > harmony/cuda-norm/unsloth_logs/dflash_train_${ts}.log 2>&1 &
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import modal


APP_NAME = "dflash-gptoss20b-train"
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

SGLANG_PY_SRC = _repo_root / "sglang-flashinfer" / "python" / "sglang"
SGL_KERNEL_PY_SRC = _repo_root / "sglang-flashinfer" / "sgl-kernel" / "python" / "sgl_kernel"

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

train_image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.11")
    .apt_install("git", "python3-dev", "build-essential", "curl", "libnuma1")
    .run_commands("python -m pip install -U pip setuptools wheel")
    .run_commands(
        "python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128",
        "python -m pip install transformers==4.56.2 tokenizers safetensors accelerate datasets pyarrow",
        "python -m pip install huggingface-hub==0.36.0 hf-transfer",
        # Optional (gated by env at runtime): in-process SGLang teacher forward.
        "python -m pip install 'sglang[all]'",
    )
    .add_local_dir(str(SGLANG_PY_SRC), remote_path="/root/sglang-src", copy=True)
    .add_local_dir(str(SGL_KERNEL_PY_SRC), remote_path="/root/sgl-kernel-src", copy=True)
    .add_local_dir(str(_repo_root / "dflash_gptoss"), remote_path="/root/dflash_gptoss", copy=True)
    .run_commands(
        "cp -rfv /root/sglang-src/* /usr/local/lib/python3.11/site-packages/sglang/",
        "find /usr/local/lib/python3.11/site-packages/sglang -name '__pycache__' -type d -exec rm -rf {} +",
        "cp -rfv /root/sgl-kernel-src/* /usr/local/lib/python3.11/site-packages/sgl_kernel/",
        "find /usr/local/lib/python3.11/site-packages/sgl_kernel -name '__pycache__' -type d -exec rm -rf {} +",
    )
    .env(
        {
            "PYTHONPATH": "/root",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/root/hf_cache",
            "TRANSFORMERS_CACHE": "/root/hf_cache/transformers",
            "HF_DATASETS_CACHE": "/root/hf_cache/datasets",
            "HUGGINGFACE_HUB_CACHE": "/root/hf_cache/hub",
            "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
            # Training uses torch autograd; we still avoid torch compile workers unless explicitly enabled.
            "TORCHINDUCTOR_DISABLE": "1",
        }
    )
)

app = modal.App(APP_NAME)

@app.function(
    image=cpu_image,
    timeout=21600,
    cpu=8.0,
    memory=65536,
    volumes={"/root/hf_cache": hf_cache_volume},
    secrets=_secrets,
)
def predownload_remote(*, model_id: str, dataset_repo: str, train_files: list[str]) -> str:
    from huggingface_hub import hf_hub_download
    from huggingface_hub import snapshot_download

    hf_cache_volume.reload()
    token = os.environ.get("HF_TOKEN")

    # Model weights/config/tokenizer.
    snapshot_download(
        repo_id=str(model_id),
        repo_type="model",
        token=token,
        local_files_only=False,
        max_workers=16,
    )

    # Dataset pack files.
    for f in train_files:
        hf_hub_download(
            repo_id=str(dataset_repo),
            repo_type="dataset",
            filename=str(f),
            token=token,
        )

    hf_cache_volume.commit()
    return "ok"


def _ensure_mask_token(tok, *, target_model) -> int:
    # Prefer an existing token id to avoid resizing (works with SGLang/TRTLLM).
    if tok.mask_token_id is not None:
        return int(tok.mask_token_id)
    if tok.pad_token_id is not None:
        return int(tok.pad_token_id)
    tok.add_special_tokens({"mask_token": "<|MASK|>"})
    target_model.resize_token_embeddings(len(tok))
    return int(tok.mask_token_id)


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


def _iter_parquet_texts(parquet_path: Path, *, text_column: str = "text") -> Iterable[str]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(parquet_path))
    for rg in range(pf.num_row_groups):
        tab = pf.read_row_group(rg, columns=[text_column])
        col = tab.column(text_column)
        for v in col.to_pylist():
            if isinstance(v, str) and v.strip():
                yield v


@dataclass
class TrainBatch:
    context_ids: "torch.LongTensor"
    block_ids: "torch.LongTensor"
    noise_block_ids: "torch.LongTensor"
    mask_pos: "torch.BoolTensor"


def _make_training_stream(tok, *, eos_id: int, seq_len: int, block_size: int, texts: Iterable[str], mask_token_id: int):
    import torch

    buf: list[int] = []
    need = int(seq_len) + int(block_size)
    for t in texts:
        ids = tok(t, add_special_tokens=False).input_ids
        if eos_id is not None:
            ids = ids + [int(eos_id)]
        buf.extend(ids)
        while len(buf) >= need:
            chunk = buf[:need]
            buf = buf[need:]
            context = torch.tensor(chunk[:seq_len], dtype=torch.long)
            block = torch.tensor(chunk[seq_len:], dtype=torch.long)
            # DFlash block diffusion training: token0 is the anchor/current token;
            # ALL subsequent positions are masked (the inference-time regime).
            noise = block.clone()
            mask = torch.zeros((block_size,), dtype=torch.bool)
            for i in range(1, block_size):
                noise[i] = int(mask_token_id)
                mask[i] = True
            yield TrainBatch(
                context_ids=context,
                block_ids=block,
                noise_block_ids=noise,
                mask_pos=mask,
            )


@app.function(
    image=train_image,
    gpu="H100:1",
    timeout=21600,
    cpu=12.0,
    memory=262144,
    volumes={"/root/model": model_volume, "/root/hf_cache": hf_cache_volume},
    secrets=_secrets,
)
def train_remote(
    *,
    dataset_repo: str,
    train_files: list[str],
    model_id: str,
    seq_len: int,
    block_size: int,
    num_hidden_layers: int,
    mlp_ratio: float,
    max_steps: int,
    lr: float,
    log_every: int,
    save_every: int,
) -> str:
    import torch
    from torch.nn import functional as F
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from dflash_gptoss.modeling_gptoss_dflash import GptOssDFlashDraftModel
    from dflash_gptoss.sglang_inproc_teacher import SGLangInprocTeacher

    model_volume.reload()
    hf_cache_volume.reload()

    tok = AutoTokenizer.from_pretrained(model_id)
    use_sglang_teacher = os.environ.get("DFLASH_USE_SGLANG_TEACHER", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    sglang_attention_backend = os.environ.get("DFLASH_SGLANG_ATTENTION_BACKEND", "fa3").strip() or "fa3"
    cfg = AutoConfig.from_pretrained(model_id)

    target = None
    if not use_sglang_teacher:
        target = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cuda:0"
        ).eval()
        for p in target.parameters():
            p.requires_grad_(False)

    if target is not None:
        mask_token_id = _ensure_mask_token(tok, target_model=target)
    else:
        # Do NOT add a new token: SGLang teacher weights cannot be resized.
        mask_token_id = int(tok.mask_token_id) if tok.mask_token_id is not None else int(tok.pad_token_id)
    eos_id = tok.eos_token_id

    draft = GptOssDFlashDraftModel.from_target_config(
        target_model_id=model_id,
        target_config=(target.config if target is not None else cfg),
        num_hidden_layers=int(num_hidden_layers),
        block_size=int(block_size),
        mlp_ratio=float(mlp_ratio),
    ).to(device="cuda:0", dtype=torch.bfloat16)

    draft.train()
    opt = torch.optim.AdamW(draft.parameters(), lr=float(lr), betas=(0.9, 0.95), weight_decay=0.01)

    teacher: SGLangInprocTeacher | None = None
    if use_sglang_teacher:
        # Capture per-layer features and concatenate them, matching DFlash conditioning.
        # Note: draft.target_layer_ids are indices into the target's hidden layers (0..L-1).
        teacher = SGLangInprocTeacher(
            model_path=str(model_id),
            attention_backend=str(sglang_attention_backend),
            context_length=int(seq_len),
            dtype="bfloat16",
            mem_fraction_static=0.80,
            layers_to_capture=list(draft.target_layer_ids),
        )

    # Download parquet files once into the HF cache volume.
    paths = [_download_dataset_file(dataset_repo, f) for f in train_files]

    def union_iter():
        iters = [_iter_parquet_texts(p) for p in paths]
        # round-robin
        active = list(iters)
        while active:
            nxt = []
            for it in active:
                try:
                    yield next(it)
                    nxt.append(it)
                except StopIteration:
                    pass
            active = nxt

    stream = _make_training_stream(
        tok,
        eos_id=int(eos_id) if eos_id is not None else None,
        seq_len=int(seq_len),
        block_size=int(block_size),
        texts=union_iter(),
        mask_token_id=int(mask_token_id),
    )

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("/root/model") / "dflash_gptoss20b" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for step in range(1, int(max_steps) + 1):
        batch = next(stream)
        device = torch.device("cuda:0")
        context_ids = batch.context_ids.unsqueeze(0).to(device)
        block_ids = batch.block_ids.unsqueeze(0).to(device)
        noise_block_ids = batch.noise_block_ids.unsqueeze(0).to(device)
        mask_pos = batch.mask_pos.unsqueeze(0).to(device)

        attn = torch.ones_like(context_ids)
        with torch.no_grad():
            if teacher is None:
                assert target is not None
                out = target(
                    context_ids, attention_mask=attn, use_cache=False, output_hidden_states=True
                )
                target_hidden = draft.extract_context_feature(out.hidden_states)
                base_noise_embedding = target.model.embed_tokens(noise_block_ids)
            else:
                # Teacher returns [seq_len, layers*hidden]; add batch dim.
                t_out = teacher.prefill_hidden_states(context_ids)
                target_hidden = t_out.hidden_states.unsqueeze(0)
                base_noise_embedding = (
                    teacher.embed_tokens(noise_block_ids)
                    .reshape(1, int(block_size), -1)
                    .contiguous()
                )

        # Replace masked positions with a learned mask embedding. Use a
        # differentiable `torch.where` so gradients can flow to `mask_embedding`.
        if hasattr(draft, "mask_embedding"):
            mask = (noise_block_ids == int(mask_token_id)).view(1, int(block_size), 1)
            mask_embed = draft.mask_embedding.to(base_noise_embedding.dtype).view(1, 1, -1)
            noise_embedding = torch.where(mask, mask_embed, base_noise_embedding)
        else:
            noise_embedding = base_noise_embedding

        # Draft predicts block tokens 1..end (token0 is anchor)
        pos = torch.arange(int(seq_len) + int(block_size), device=device).unsqueeze(0)
        draft_out = draft(
            position_ids=pos,
            attention_mask=None,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            use_cache=False,
        )
        if teacher is None:
            assert target is not None
            logits = target.lm_head(draft_out[:, 1:, :])
        else:
            # Use SGLang lm_head weights to map draft hidden states -> logits.
            flat = draft_out[:, 1:, :].reshape(-1, draft_out.size(-1))
            flat_logits = teacher.lm_head_logits(flat)
            logits = flat_logits.view(1, -1, flat_logits.size(-1))
        labels = block_ids[:, 1:]

        # Compute CE in fp32 for numerical stability.
        loss_tok = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none")
        loss_tok = loss_tok.view_as(labels)
        # DFlash training uses completion over all masked positions (1..end).
        loss = loss_tok.mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
        opt.step()

        if step % int(log_every) == 0 or step == 1:
            dt = max(1e-9, time.time() - t0)
            tok_s = (step * int(seq_len + block_size)) / dt
            print(f"[step {step}] loss={loss.item():.4f} tok_s(train_proxy)={tok_s:.1f}", flush=True)

        if step % int(save_every) == 0 or step == int(max_steps):
            ckpt_dir = out_dir / f"step_{step:06d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            draft.save_pretrained(str(ckpt_dir))
            tok.save_pretrained(str(ckpt_dir))
            (ckpt_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "target_model_id": model_id,
                        "dataset_repo": dataset_repo,
                        "train_files": train_files,
                        "seq_len": int(seq_len),
                        "block_size": int(block_size),
                        "num_hidden_layers": int(num_hidden_layers),
                        "mlp_ratio": float(mlp_ratio),
                        "lr": float(lr),
                        "step": int(step),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            model_volume.commit()
            print(f"[+] Saved {ckpt_dir}", flush=True)

    if teacher is not None:
        teacher.close()

    return str(out_dir)


@app.local_entrypoint()
def main(
    dataset_repo: str = "radna0/harmony-qwen3-calib-packs-v2-20260113",
    train_files_csv: str = "packs/reasoning_style_10k_v2/reasoning_style_10k_v2.parquet,tool_agentic_10k_v6.parquet,packs/calib_prompt_10000_v2/calib_prompt_10000_v2.parquet",
    model_id: str = "openai/gpt-oss-20b",
    seq_len: int = 4096,
    block_size: int = 16,
    num_hidden_layers: int = 4,
    mlp_ratio: float = 4.0,
    max_steps: int = 50,
    lr: float = 2e-4,
    log_every: int = 1,
    save_every: int = 25,
    predownload_only: bool = False,
):
    train_files = [x.strip() for x in (train_files_csv or "").split(",") if x.strip()]
    predownload_remote.remote(model_id=model_id, dataset_repo=dataset_repo, train_files=train_files)
    if predownload_only:
        print("predownload ok")
        return
    out = train_remote.remote(
        dataset_repo=dataset_repo,
        train_files=train_files,
        model_id=model_id,
        seq_len=int(seq_len),
        block_size=int(block_size),
        num_hidden_layers=int(num_hidden_layers),
        mlp_ratio=float(mlp_ratio),
        max_steps=int(max_steps),
        lr=float(lr),
        log_every=int(log_every),
        save_every=int(save_every),
    )
    print(out)
