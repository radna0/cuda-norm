"""
Train a GPT-OSS-120B DFlash draft model (block diffusion drafter) on Modal B200.

Key points:
  - Target/verifier is GPT-OSS-120B. We use SGLang in-process teacher-forward to:
      * extract prompt context features (concat of selected target layers)
      * provide embeddings and lm_head projection (no HF model load for 120B)
  - Training task matches DFlash inference regime:
      * block token0 is the anchor/current token
      * block tokens 1..B-1 are ALWAYS masked
      * loss is CE over ALL positions 1..B-1
  - Checkpoints saved every `save_every` steps (default 500). Also saves step_000000
    before training for strict comparisons.

Logs:
  Write `nohup modal run ... > harmony/cuda-norm/unsloth_logs/...log 2>&1 &`
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


APP_NAME = "dflash-gptoss120b-train"
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
        # In-process SGLang teacher forward (uses TRTLLM/FlashInfer backends for 120B).
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
            # Avoid inductor workers in this training loop.
            "TORCHINDUCTOR_DISABLE": "1",
            # Reduce fragmentation risk during large weight materialization.
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
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
def predownload_remote(*, model_id: str, dataset_repo: str, train_files: list[str]) -> str:
    from huggingface_hub import hf_hub_download
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
    for f in train_files:
        hf_hub_download(
            repo_id=str(dataset_repo),
            repo_type="dataset",
            filename=str(f),
            token=token,
        )

    hf_cache_volume.commit()
    return "ok"


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
            noise = block.clone()
            # DFlash regime: token0 anchor, all others masked.
            for i in range(1, int(block_size)):
                noise[i] = int(mask_token_id)
            yield TrainBatch(
                context_ids=context,
                block_ids=block,
                noise_block_ids=noise,
            )


@app.function(
    image=train_image,
    gpu="B200:1",
    timeout=21600,
    cpu=16.0,
    memory=524288,
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
    seed: int,
    resume_from: str,
    resume_skip_data: bool,
    log_every: int,
    save_every: int,
    save_step0: bool,
) -> str:
    import torch
    from torch.nn import functional as F
    from transformers import AutoConfig, AutoTokenizer

    from dflash_gptoss.modeling_gptoss_dflash import GptOssDFlashDraftModel
    from dflash_gptoss.sglang_inproc_teacher import SGLangInprocTeacher

    model_volume.reload()
    hf_cache_volume.reload()

    random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        # Do not resize embeddings. GPT-OSS tokenizer has pad by default; if not, fail fast.
        raise ValueError("Tokenizer has no pad_token_id; cannot run mask-token fallback safely.")
    mask_token_id = int(tok.mask_token_id) if tok.mask_token_id is not None else int(tok.pad_token_id)
    eos_id = tok.eos_token_id

    cfg = AutoConfig.from_pretrained(model_id)

    start_step = 0
    out_dir: Path
    ckpt_path = Path(str(resume_from)).expanduser()
    is_resume = bool(str(resume_from).strip())
    if is_resume:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume-from not found: {ckpt_path}")
        meta_path = ckpt_path / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                start_step = int(meta.get("step", 0))
            except Exception:
                start_step = 0
        out_dir = ckpt_path.parent
        draft = GptOssDFlashDraftModel.from_pretrained(str(ckpt_path), torch_dtype=torch.bfloat16).to("cuda:0")
        draft.train()
    else:
        out_dir = Path("/root/model") / "dflash_gptoss120b" / time.strftime("%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        draft = GptOssDFlashDraftModel.from_target_config(
            target_model_id=model_id,
            target_config=cfg,
            num_hidden_layers=int(num_hidden_layers),
            block_size=int(block_size),
            mlp_ratio=float(mlp_ratio),
        ).to(device="cuda:0", dtype=torch.bfloat16)
        draft.train()

    # B200 path: prefer TRTLLM MHA for 120B teacher if available.
    # Capture per-layer features and concatenate them (required for draft.fc).
    sglang_attention_backend = (
        os.environ.get("DFLASH_SGLANG_ATTENTION_BACKEND", "trtllm_mha").strip() or "trtllm_mha"
    )
    sglang_moe_backend = os.environ.get("DFLASH_SGLANG_MOE_BACKEND", "flashinfer_mxfp4").strip() or "flashinfer_mxfp4"
    # For GPT-OSS-120B on B200, keep KV/cache reservation conservative so MXFP4
    # weight post-processing has headroom (prevents OOM during load).
    sglang_mem_frac = float(os.environ.get("DFLASH_SGLANG_MEM_FRACTION_STATIC", "0.45"))
    teacher = SGLangInprocTeacher(
        model_path=str(model_id),
        attention_backend=str(sglang_attention_backend),
        context_length=int(seq_len),
        dtype="bfloat16",
        moe_runner_backend=str(sglang_moe_backend),
        mem_fraction_static=float(sglang_mem_frac),
        max_total_tokens=int(seq_len),
        layers_to_capture=list(draft.target_layer_ids),
    )

    opt = torch.optim.AdamW(draft.parameters(), lr=float(lr), betas=(0.9, 0.95), weight_decay=0.01)
    if is_resume:
        opt_path = ckpt_path / "optimizer.pt"
        if opt_path.exists():
            try:
                state = torch.load(str(opt_path), map_location="cpu")
                opt.load_state_dict(state["opt"])
                if "torch_rng_state" in state:
                    torch.set_rng_state(state["torch_rng_state"])
                if "cuda_rng_state_all" in state:
                    torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
                if "py_random_state" in state:
                    random.setstate(state["py_random_state"])
            except Exception:
                # Best-effort: training can still proceed without optimizer resume.
                pass

    # Download parquet files once into the HF cache volume.
    paths = [_download_dataset_file(dataset_repo, f) for f in train_files]

    def union_iter():
        iters = [_iter_parquet_texts(p) for p in paths]
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

    def _save(step: int):
        ckpt_dir = out_dir / f"step_{step:06d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        draft.save_pretrained(str(ckpt_dir))
        tok.save_pretrained(str(ckpt_dir))
        torch.save(
            {
                "opt": opt.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                "py_random_state": random.getstate(),
            },
            str(ckpt_dir / "optimizer.pt"),
        )
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
                    "seed": int(seed),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        model_volume.commit()
        print(f"[+] Saved {ckpt_dir}", flush=True)

    if bool(save_step0) and start_step <= 0:
        _save(0)

    if int(start_step) >= int(max_steps):
        raise ValueError(f"resume step {start_step} >= max_steps {max_steps}; nothing to do")

    print(
        f"[*] Training: model_id={model_id} seq_len={seq_len} block_size={block_size} "
        f"draft_layers={num_hidden_layers} lr={lr} seed={seed} "
        f"resume_from={str(resume_from) if is_resume else ''} start_step={start_step} max_steps={max_steps} "
        f"resume_skip_data={bool(resume_skip_data)}",
        flush=True,
    )

    t0 = time.time()
    # NOTE: Aligning the dataset cursor exactly on resume is expensive (it requires
    # re-tokenizing `start_step` batches). Default is to *not* skip and just
    # continue training from the checkpoint weights on fresh batches.
    if bool(resume_skip_data) and int(start_step) > 0:
        for _ in range(int(start_step)):
            next(stream)

    for step in range(int(start_step) + 1, int(max_steps) + 1):
        batch = next(stream)
        device = torch.device("cuda:0")
        context_ids = batch.context_ids.unsqueeze(0).to(device)
        block_ids = batch.block_ids.unsqueeze(0).to(device)
        noise_block_ids = batch.noise_block_ids.unsqueeze(0).to(device)

        # Teacher prefill features (GPU tensor) + embeddings.
        with torch.no_grad():
            t_out = teacher.prefill_hidden_states(context_ids)
            target_hidden = t_out.hidden_states.unsqueeze(0)
            expect_last = int(draft.fc.weight.shape[1])
            if int(target_hidden.shape[-1]) != expect_last:
                raise RuntimeError(
                    f"Teacher feature dim mismatch: got {tuple(target_hidden.shape)}, "
                    f"expected last dim {expect_last}. "
                    f"(len(target_layer_ids)={len(draft.target_layer_ids)}, hidden_size={draft.config.hidden_size})"
                )
            # Keep teacher embeddings detached from autograd.
            base_noise_embedding = (
                teacher.embed_tokens(noise_block_ids)
                .reshape(1, int(block_size), -1)
                .contiguous()
            )
        # Replace masked positions with a learned mask embedding. Use a
        # differentiable `torch.where` so gradients can flow to `mask_embedding`.
        mask = (noise_block_ids == int(mask_token_id)).view(1, int(block_size), 1)
        mask_embed = draft.mask_embedding.to(base_noise_embedding.dtype).view(1, 1, -1)
        noise_embedding = torch.where(mask, mask_embed, base_noise_embedding)

        pos = torch.arange(int(seq_len) + int(block_size), device=device).unsqueeze(0)
        draft_out = draft(
            position_ids=pos,
            attention_mask=None,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            use_cache=False,
        )

        # Project to vocab using teacher lm_head weights.
        flat = draft_out[:, 1:, :].reshape(-1, draft_out.size(-1))
        flat_logits = teacher.lm_head_logits(flat)
        logits = flat_logits.view(1, -1, flat_logits.size(-1))

        labels = block_ids[:, 1:]
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="mean")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
        opt.step()

        if step % int(log_every) == 0 or step == 1:
            dt = max(1e-9, time.time() - t0)
            local_step = int(step) - int(start_step)
            tok_s = (local_step * int(seq_len + block_size)) / dt
            print(
                f"[step {step}] loss={loss.item():.4f} tok_s(train_proxy)={tok_s:.1f} "
                f"(local_step={local_step}/{int(max_steps)-int(start_step)})",
                flush=True,
            )

        if step % int(save_every) == 0 or step == int(max_steps):
            _save(step)

    teacher.close()
    return str(out_dir)


@app.local_entrypoint()
def main(
    dataset_repo: str = "radna0/harmony-nemotron-cpu-artifacts",
    # Focus on tool-calling and agentic domains (distilled from 120B).
    train_files_csv: str = (
        "normalized/nvidia__Nemotron-Agentic-v1/data/tool_calling/part-seg00000-00000.parquet,"
        "normalized/nvidia__Nemotron-Agentic-v1/data/tool_calling/part-seg00001-00000.parquet,"
        "normalized/nvidia__Nemotron-Agentic-v1/data/interactive_agent/part-seg00000-00000.parquet,"
        "normalized/nvidia__Nemotron-Agentic-v1/data/interactive_agent/part-seg00001-00000.parquet"
    ),
    model_id: str = "openai/gpt-oss-120b",
    seq_len: int = 4096,
    block_size: int = 16,
    num_hidden_layers: int = 8,
    mlp_ratio: float = 4.0,
    max_steps: int = 500,
    lr: float = 2e-4,
    seed: int = 3407,
    resume_from: str = "",
    resume_skip_data: str = "false",
    log_every: int = 20,
    save_every: int = 500,
    # Modal/typer treats `bool=True` as a flag (no value). Accept strings so users can pass
    # `--save-step0 true/false` without "unexpected extra argument".
    save_step0: str = "true",
    predownload_only: str = "false",
):
    train_files = [x.strip() for x in (train_files_csv or "").split(",") if x.strip()]
    predownload_remote.remote(model_id=model_id, dataset_repo=dataset_repo, train_files=train_files)
    if _parse_bool(predownload_only, default=False):
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
        seed=int(seed),
        resume_from=str(resume_from),
        resume_skip_data=_parse_bool(resume_skip_data, default=False),
        log_every=int(log_every),
        save_every=int(save_every),
        save_step0=_parse_bool(save_step0, default=True),
    )
    print(out)
