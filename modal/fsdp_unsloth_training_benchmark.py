# modal/fsdp_unsloth_training_benchmark.py
# GPT-OSS + Unsloth LoRA on 8x B200 (Modal).
#
# Key features:
# - Predownload model + dataset into Modal Volumes (one-time) before training
# - Trains on `radna0/nemotron-math-v2-harmony-tools`, split `high_part00`
# - Dataset is a single `text` column already formatted for GPT-OSS Harmony; we
#   apply completion-only loss by masking prompt tokens.

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import modal

APP_NAME = "gpt-oss-fsdp-unsloth-benchmark"

DEFAULT_MODEL_ID = os.environ.get("MODEL_ID", "unsloth/gpt-oss-20b")
DEFAULT_FALLBACK_MODEL_ID = os.environ.get(
    "FALLBACK_MODEL_ID", "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
)
DEFAULT_DATASET_ID = os.environ.get(
    "DATASET_ID", "radna0/nemotron-math-v2-harmony-tools"
)
DEFAULT_DATASET_SPLIT = os.environ.get("DATASET_SPLIT", "high_part00")

# Optional: pass your local `HF_TOKEN` into the Modal container at launch time.
_secrets = []
_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    _secrets.append(modal.Secret.from_dict({"HF_TOKEN": _hf_token}))

# -------------------------
# Volumes (persist across runs)
# -------------------------
data_volume = modal.Volume.from_name("mhc-data-volume", create_if_missing=True)
model_volume = modal.Volume.from_name("gpt-oss-models", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("hf-cache-persistent", create_if_missing=True)

# -------------------------
# Image
# -------------------------
BASE_IMAGE = "nvidia/cuda:12.8.0-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.12")
    .apt_install("git", "build-essential", "clang", "python3-dev", "curl")
    .run_commands("python -m pip install -U pip setuptools wheel")
    # Torch CUDA 12.8
    .run_commands(
        "python -m pip install "
        "torch==2.9.0 torchvision torchaudio "
        "--extra-index-url https://download.pytorch.org/whl/cu128"
    )
    # Core stack
    .run_commands(
        "python -m pip install "
        "numpy==2.2.0 datasets==3.2.0 accelerate==1.10.1 trl==0.22.2 peft bitsandbytes "
        "sentencepiece protobuf msgspec ninja wandb huggingface_hub hf_transfer "
        "transformers==4.57.3 timm"
    )
    # Unsloth (from source)
    .run_commands(
        "python -m pip install --upgrade --force-reinstall --no-cache-dir --no-deps "
        "git+https://github.com/unslothai/unsloth-zoo.git",
        "python -m pip install --upgrade --force-reinstall --no-cache-dir --no-deps "
        "git+https://github.com/unslothai/unsloth.git",
    )
    # Required for GPT-OSS MXFP4 fastpath (per Unsloth docs).
    .run_commands(
        "python -m pip install "
        "git+https://github.com/triton-lang/triton.git@0add68262ab0a2e33b84524346cb27cbb2787356"
        "#subdirectory=python/triton_kernels"
    )
)

app = modal.App(APP_NAME)


def _ensure_hf_env() -> None:
    os.environ.setdefault("HF_HOME", "/root/hf_cache")
    os.environ.setdefault("XDG_CACHE_HOME", "/root/hf_cache/.cache")
    # Keep HF cache persistent, but use local (non-volume) paths for compiler caches.
    # TorchInductor / Triton file locks can be problematic on networked filesystems.
    os.environ.setdefault("TORCH_HOME", "/root/hf_cache/torch")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torch_inductor_cache")
    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
    os.environ.setdefault("CUDA_CACHE_PATH", "/tmp/cuda_cache")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    # Backwards compat for older torch / docs.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", os.environ["PYTORCH_ALLOC_CONF"])
    # Unsloth compile knobs (see Unsloth source `unsloth/models/_utils.py`)
    # Default to Unsloth's upstream default ("0") for stability in multi-GPU;
    # you can still opt into maximum compile by setting it to "1".
    os.environ.setdefault("UNSLOTH_COMPILE_MAXIMUM", "0")
    os.environ.setdefault("UNSLOTH_COMPILE_IGNORE_ERRORS", "1")
    # GPT-OSS long-context fastpath
    os.environ.setdefault("UNSLOTH_ENABLE_FLEX_ATTENTION", "1")
    for envvar in (
        "HF_HOME",
        "XDG_CACHE_HOME",
        "TORCH_HOME",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
    ):
        try:
            Path(os.environ[envvar]).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


def _patch_transformers_gpt_oss_init() -> None:
    # Some transformer + Unsloth combinations can hit GPT-OSS `_init_weights` bugs
    # (missing attrs on router / experts modules). Weight initialization is
    # needed for `_initialize_missing_keys` during `from_pretrained`, so we keep
    # init but make it robust to missing attrs introduced by patching.
    try:
        from transformers.models.gpt_oss import modeling_gpt_oss as gpt_oss_mod
    except Exception:
        return

    if getattr(gpt_oss_mod, "_harmony_router_init_patched", False):
        return

    base_cls = getattr(gpt_oss_mod, "GptOssPreTrainedModel", None)
    if base_cls is None:
        return

    def patched_init_weights(self, module):
        import torch
        import torch.nn as nn

        std = float(getattr(self.config, "initializer_range", 0.02))
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
            return
        if isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=std)
            return
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if getattr(module, "padding_idx", None) is not None:
                module.weight.data[module.padding_idx].zero_()
            return

        GptOssRMSNorm = getattr(gpt_oss_mod, "GptOssRMSNorm", None)
        if GptOssRMSNorm is not None and isinstance(module, GptOssRMSNorm):
            module.weight.data.fill_(1.0)
            return

        GptOssExperts = getattr(gpt_oss_mod, "GptOssExperts", None)
        if GptOssExperts is not None and isinstance(module, GptOssExperts):
            for name in ("gate_up_proj", "down_proj"):
                if hasattr(module, name):
                    getattr(module, name).data.normal_(mean=0.0, std=std)
            for name in ("gate_up_proj_bias", "down_proj_bias"):
                if hasattr(module, name):
                    getattr(module, name).data.zero_()
            return

        GptOssAttention = getattr(gpt_oss_mod, "GptOssAttention", None)
        if GptOssAttention is not None and isinstance(module, GptOssAttention):
            if hasattr(module, "sinks"):
                module.sinks.data.normal_(mean=0.0, std=std)
            return

        GptOssTopKRouter = getattr(gpt_oss_mod, "GptOssTopKRouter", None)
        if GptOssTopKRouter is not None and isinstance(module, GptOssTopKRouter):
            if hasattr(module, "weight"):
                module.weight.data.normal_(mean=0.0, std=std)
            if hasattr(module, "bias"):
                module.bias.data.normal_(mean=0.0, std=std)
            return

    base_cls._init_weights = patched_init_weights
    gpt_oss_mod._harmony_router_init_patched = True


def _iter_high_part00_texts_from_parquet(parquet_files: list[str]) -> Iterable[str]:
    import pyarrow.parquet as pq

    for fp in parquet_files:
        pf = pq.ParquetFile(fp)
        for row_group in range(pf.num_row_groups):
            table = pf.read_row_group(row_group, columns=["text"])
            for text in table.column("text").to_pylist():
                if text:
                    yield text


def _split_prompt_completion(harmony_text: str) -> tuple[str, str]:
    marker = "<|start|>assistant"
    idx = harmony_text.find(marker)
    if idx == -1:
        return "", harmony_text
    return harmony_text[:idx], harmony_text[idx:]


def _local_model_dir(model_id: str) -> Path:
    return Path("/root/model") / model_id


def _local_dataset_dir(dataset_id: str) -> Path:
    return Path("/root/data/datasets") / dataset_id


def _local_tokenized_dir(
    dataset_id: str,
    dataset_split: str,
    max_seq_length: int,
    train_samples: int,
    eval_samples: int,
) -> Path:
    # Tokenization depends on the tokenizer, but for GPT-OSS model variants
    # (MXFP4 vs bnb-4bit) it's the same; keep the cache keyed by dataset + length.
    return (
        Path("/root/data/tokenized")
        / _sanitize_for_path(dataset_id)
        / dataset_split
        / f"msl_{max_seq_length}"
        / f"train_{train_samples}_eval_{eval_samples}"
    )


def _sanitize_for_path(s: str) -> str:
    return s.replace("/", "__")


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _find_decoder_layer_cls(model):
    import torch.nn as nn

    if not isinstance(model, nn.Module):
        return None
    for m in model.modules():
        if m.__class__.__name__.endswith("DecoderLayer"):
            return m.__class__
    return None


def _collect_lora_modules(model) -> list:
    # Identify PEFT LoRA layers so we can exclude them from FSDP sharding.
    # This makes adapter saving straightforward (rank0 writes full LoRA weights).
    lora_modules = []
    for module in model.modules():
        if any(
            hasattr(module, attr)
            for attr in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
        ):
            lora_modules.append(module)
    return lora_modules


@dataclass(frozen=True)
class _WorkerArgs:
    strategy: str
    load_in_4bit: bool
    model_id: str
    dataset_id: str
    dataset_split: str
    train_samples: int
    eval_samples: int
    max_seq_length: int
    max_steps: int
    run_dir: str
    tokenized_cache_dir: str


def _tokenize_to_disk_if_needed(
    tokenizer,
    dataset_id: str,
    dataset_split: str,
    train_samples: int,
    eval_samples: int,
    max_seq_length: int,
    tokenized_cache_dir: Path,
) -> None:
    import json

    from datasets import Dataset

    train_dir = tokenized_cache_dir / "train"
    eval_dir = tokenized_cache_dir / "eval"
    meta_path = tokenized_cache_dir / "meta.json"

    if train_dir.exists() and (not eval_samples or eval_dir.exists()):
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                expected = {
                    "dataset_id": dataset_id,
                    "dataset_split": dataset_split,
                    "train_samples": int(train_samples),
                    "eval_samples": int(eval_samples),
                    "max_seq_length": int(max_seq_length),
                }
                for k, v in expected.items():
                    if meta.get(k) != v:
                        raise RuntimeError(
                            f"Tokenized cache mismatch at {tokenized_cache_dir} (key={k}). "
                            f"Expected {v}, found {meta.get(k)}.\n"
                            f"Delete the directory to rebuild."
                        )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to validate tokenized cache at {tokenized_cache_dir}: {e}"
                ) from e
        else:
            print(
                f"[warn] tokenized dataset present but missing meta.json at {tokenized_cache_dir}; using it as-is"
            )
        print(f"[skip] tokenized dataset already present at {tokenized_cache_dir}")
        return

    dataset_dir = _local_dataset_dir(dataset_id)
    parquet_files = sorted((dataset_dir / "data").glob(f"{dataset_split}-*.parquet"))
    if not parquet_files:
        raise RuntimeError(
            f"Dataset split not found in volume at {dataset_dir}/data/{dataset_split}-*.parquet."
        )

    need = train_samples + eval_samples
    print(f"[*] Tokenizing {need} rows from {len(parquet_files)} parquet shards...")

    train_tokenized: list[dict] = []
    eval_tokenized: list[dict] = []
    skipped_no_completion = 0
    total_seen = 0

    for text in _iter_high_part00_texts_from_parquet([str(p) for p in parquet_files]):
        total_seen += 1
        prompt, completion = _split_prompt_completion(text)

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + completion_ids
        completion_mask = ([0] * len(prompt_ids)) + ([1] * len(completion_ids))

        if max_seq_length is not None and len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            completion_mask = completion_mask[:max_seq_length]

        if 1 not in completion_mask:
            skipped_no_completion += 1
            continue

        record = {"input_ids": input_ids, "completion_mask": completion_mask}
        if len(train_tokenized) < train_samples:
            train_tokenized.append(record)
        elif len(eval_tokenized) < eval_samples:
            eval_tokenized.append(record)
        else:
            break

    if len(train_tokenized) < train_samples:
        raise RuntimeError(
            f"Only produced {len(train_tokenized)} train rows; expected {train_samples}. "
            f"(seen={total_seen}, skipped_no_completion={skipped_no_completion})"
        )
    if eval_samples and len(eval_tokenized) < eval_samples:
        raise RuntimeError(
            f"Only produced {len(eval_tokenized)} eval rows; expected {eval_samples}. "
            f"(seen={total_seen}, skipped_no_completion={skipped_no_completion})"
        )

    tokenized_cache_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    if eval_samples:
        eval_dir.mkdir(parents=True, exist_ok=True)

    train_ds = Dataset.from_list(train_tokenized)
    train_ds.save_to_disk(str(train_dir))

    if eval_samples:
        eval_ds = Dataset.from_list(eval_tokenized)
        eval_ds.save_to_disk(str(eval_dir))

    meta = {
        "dataset_id": dataset_id,
        "dataset_split": dataset_split,
        "train_samples": int(train_samples),
        "eval_samples": int(eval_samples),
        "max_seq_length": int(max_seq_length),
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(
        f"[+] Saved tokenized datasets to {tokenized_cache_dir} "
        f"(train={len(train_tokenized)} eval={len(eval_tokenized)})"
    )


def _ddp_qlora_train_worker(w: _WorkerArgs) -> None:
    import math
    from contextlib import nullcontext

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    from torch.utils.data.distributed import DistributedSampler

    _ensure_hf_env()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))

    if world_size > 1 and not dist.is_initialized():
        from datetime import timedelta

        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=120))
    is_dist = dist.is_initialized()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this worker.")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.empty(1, device=device)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    seed = int(os.environ.get("SEED", "3407"))
    _set_seed(seed)

    is_main = (not is_dist) or rank == 0
    if is_main:
        print(
            f"[*] DDP init OK: world_size={world_size} "
            f"strategy={w.strategy} load_in_4bit={w.load_in_4bit}",
            flush=True,
        )

    # ----
    # Tokenized dataset (precomputed on CPU and stored in the data volume)
    # ----
    tokenized_cache_dir = Path(w.tokenized_cache_dir)
    train_path = tokenized_cache_dir / "train"
    eval_path = tokenized_cache_dir / "eval"

    if not train_path.exists():
        raise RuntimeError(
            f"Tokenized train dataset missing at {train_path}. "
            "Run the CPU pretokenize step before training."
        )
    if w.eval_samples and not eval_path.exists():
        raise RuntimeError(
            f"Tokenized eval dataset missing at {eval_path}. "
            "Run the CPU pretokenize step before training."
        )

    from datasets import load_from_disk

    train_ds = load_from_disk(str(train_path))
    eval_ds = load_from_disk(str(eval_path)) if w.eval_samples else None

    model_dir = _local_model_dir(w.model_id)
    if not (model_dir / "config.json").exists():
        raise RuntimeError(
            f"Model not found in volume at {model_dir}. Run predownload first."
        )

    # ----
    # Model + LoRA (QLoRA 4-bit, DDP)
    # ----
    import unsloth  # must be imported before transformers for full optimizations

    _patch_transformers_gpt_oss_init()
    from unsloth import FastLanguageModel

    lora_r = int(os.environ.get("LORA_R", "16"))
    lora_alpha = int(os.environ.get("LORA_ALPHA", str(lora_r * 2)))

    if is_main:
        print(
            f"[*] Loading model on GPU for QLoRA: {w.model_id} "
            f"(load_in_4bit={w.load_in_4bit})",
            flush=True,
        )

    os.environ["UNSLOTH_MODEL_NAME"] = ""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_dir),
        max_seq_length=w.max_seq_length,
        dtype=None,  # let Unsloth pick bf16 on B200
        load_in_4bit=bool(w.load_in_4bit),
        full_finetuning=False,
        unsloth_tiled_mlp=True,
        device_map={"": local_rank},
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        use_rslora=True,
        max_seq_length=w.max_seq_length,
    )
    try:
        model.config.use_cache = False
    except Exception:
        pass

    if is_dist and world_size > 1:
        import inspect

        ddp_kwargs = dict(
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        sig = inspect.signature(DDP)
        if "broadcast_buffers" in sig.parameters:
            ddp_kwargs["broadcast_buffers"] = False
        if "init_sync" in sig.parameters:
            ddp_kwargs["init_sync"] = False
        model = DDP(
            model,
            **ddp_kwargs,
        )

    # ----
    # Data
    # ----
    class TokenizedCompletionDataset(TorchDataset):
        def __init__(self, hf_ds):
            self._ds = hf_ds

        def __len__(self) -> int:
            return len(self._ds)

        def __getitem__(self, idx: int) -> dict:
            row = self._ds[int(idx)]
            return {
                "input_ids": row["input_ids"],
                "completion_mask": row["completion_mask"],
            }

    class CompletionOnlyCollator:
        def __init__(self, pad_token_id: int):
            self.pad_token_id = int(pad_token_id)

        def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
            max_len = max(len(x["input_ids"]) for x in batch)
            input_ids, attention_mask, labels = [], [], []
            for x in batch:
                ids = torch.tensor(x["input_ids"], dtype=torch.long)
                cm = torch.tensor(x["completion_mask"], dtype=torch.long)

                lbl = ids.clone()
                lbl[cm == 0] = -100

                att = torch.ones_like(ids, dtype=torch.long)
                pad_len = max_len - ids.numel()
                if pad_len:
                    ids = torch.cat(
                        [
                            ids,
                            torch.full(
                                (pad_len,),
                                self.pad_token_id,
                                dtype=torch.long,
                            ),
                        ],
                        dim=0,
                    )
                    att = torch.cat([att, torch.zeros(pad_len, dtype=torch.long)], dim=0)
                    lbl = torch.cat(
                        [lbl, torch.full((pad_len,), -100, dtype=torch.long)], dim=0
                    )

                input_ids.append(ids)
                attention_mask.append(att)
                labels.append(lbl)

            return {
                "input_ids": torch.stack(input_ids, dim=0),
                "attention_mask": torch.stack(attention_mask, dim=0),
                "labels": torch.stack(labels, dim=0),
            }

    train_tds = TokenizedCompletionDataset(train_ds)
    eval_tds = TokenizedCompletionDataset(eval_ds) if eval_ds is not None else None

    per_device_train_batch_size = int(
        os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "2")
    )
    per_device_eval_batch_size = int(os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", "2"))
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", "2"))
    lr = float(os.environ.get("LEARNING_RATE", "2e-5"))
    grad_clip = float(os.environ.get("GRAD_CLIP", "1.0"))
    log_every = int(os.environ.get("LOG_EVERY", "1"))
    save_steps = int(os.environ.get("SAVE_STEPS", "16"))
    eval_steps = int(os.environ.get("EVAL_STEPS", "16"))

    if is_main:
        eff = per_device_train_batch_size * world_size * max(1, grad_accum_steps)
        print(
            "[*] Train config: "
            f"train_rows={len(train_tds)} eval_rows={(len(eval_tds) if eval_tds is not None else 0)} "
            f"max_len={w.max_seq_length} micro={per_device_train_batch_size} "
            f"grad_accum={grad_accum_steps} global_batch={eff} lr={lr} max_steps={w.max_steps}",
            flush=True,
        )

    collate = CompletionOnlyCollator(pad_token_id=int(tokenizer.pad_token_id))

    train_sampler = DistributedSampler(
        train_tds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    )
    train_loader = DataLoader(
        train_tds,
        batch_size=per_device_train_batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=collate,
    )

    eval_loader = None
    if eval_tds is not None:
        eval_sampler = DistributedSampler(
            eval_tds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        eval_loader = DataLoader(
            eval_tds,
            batch_size=per_device_eval_batch_size,
            sampler=eval_sampler,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=collate,
        )

    # ----
    # Optimizer (LoRA params only)
    # ----
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found (LoRA not applied?)")

    optimizer = None
    try:
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(trainable, lr=lr)
        if is_main:
            print("[*] Optimizer: bitsandbytes AdamW8bit", flush=True)
    except Exception:
        from torch.optim import AdamW

        optimizer = AdamW(trainable, lr=lr)
        if is_main:
            print("[*] Optimizer: torch AdamW", flush=True)

    warmup_steps = int(os.environ.get("WARMUP_STEPS", "1"))
    warmup_steps = max(1, warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ----
    # Helpers
    # ----
    def run_eval(step: int) -> None:
        if eval_loader is None:
            return

        model.train()  # Flex Attention eval path can differ in eval(); keep train-mode.
        total_loss = torch.zeros((), device=device, dtype=torch.float32)
        total_tokens = torch.zeros((), device=device, dtype=torch.float32)

        with torch.inference_mode():
            for batch in eval_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                active = (batch["labels"] != -100).sum().to(torch.float32)
                if active.item() == 0:
                    continue
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(**batch)
                    loss = out.loss.to(torch.float32)
                total_loss += loss * active
                total_tokens += active

        if is_dist:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

        if is_main:
            mean_loss = (total_loss / torch.clamp_min(total_tokens, 1.0)).item()
            ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")
            print(
                f"[eval step {step}] loss={mean_loss:.4f} ppl={ppl:.2f} tokens={int(total_tokens.item())}",
                flush=True,
            )

    def save_adapter(out_dir: Path) -> None:
        if not is_main:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        model_to_save = getattr(model, "module", model)
        lora_state = {
            n: p.detach().cpu()
            for n, p in model_to_save.named_parameters()
            if "lora_" in n
        }
        model_to_save.save_pretrained(str(out_dir), state_dict=lora_state)
        tokenizer.save_pretrained(str(out_dir))
        print(f"[+] Saved LoRA adapter to {out_dir}", flush=True)

    # ----
    # Training
    # ----
    model.train()
    if is_dist:
        dist.barrier()

    t0 = time.time()
    global_step = 0
    micro_step = 0

    run_dir = Path(w.run_dir)
    checkpoints_dir = run_dir / "checkpoints"
    final_adapter_dir = run_dir / "lora_adapter"

    for epoch in range(1):
        train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)

        for batch in train_loader:
            micro_step += 1
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            sync = (micro_step % max(1, grad_accum_steps)) == 0
            sync_ctx = (
                nullcontext()
                if (sync or not hasattr(model, "no_sync"))
                else model.no_sync()
            )

            with sync_ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(**batch)
                    loss = out.loss / max(1, grad_accum_steps)
                loss.backward()

            if not sync:
                continue

            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if is_main and (global_step <= 5 or global_step % max(1, log_every) == 0):
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"[step {global_step}] loss={loss.detach().float().item() * max(1, grad_accum_steps):.4f} lr={lr_now:.2e}",
                    flush=True,
                )

            if eval_steps and global_step % eval_steps == 0:
                run_eval(global_step)

            if save_steps and global_step % save_steps == 0:
                save_adapter(checkpoints_dir / f"step_{global_step}")

            if w.max_steps and w.max_steps > 0 and global_step >= w.max_steps:
                break

        if w.max_steps and w.max_steps > 0 and global_step >= w.max_steps:
            break

    if is_dist:
        dist.barrier()
    save_adapter(final_adapter_dir)

    if is_main:
        dt = time.time() - t0
        max_mem_gb = torch.cuda.max_memory_allocated(device=device) / (1024**3)
        print(
            f"[+] Train done. steps={global_step} wall={dt:.1f}s max_mem={max_mem_gb:.1f}GB",
            flush=True,
        )

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


def _fsdp_bf16_train_worker(w: _WorkerArgs) -> None:
    import math
    from contextlib import nullcontext

    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    from torch.utils.data.distributed import DistributedSampler
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from functools import partial

    _ensure_hf_env()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))

    if world_size > 1 and not dist.is_initialized():
        from datetime import timedelta

        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=120))
    is_dist = dist.is_initialized()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this worker.")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.empty(1, device=device)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    seed = int(os.environ.get("SEED", "3407"))
    _set_seed(seed)

    is_main = (not is_dist) or rank == 0
    if is_main:
        print(f"[*] Distributed init OK: world_size={world_size}", flush=True)

    # ----
    # Tokenized dataset (precomputed on CPU and stored in the data volume)
    # ----
    tokenized_cache_dir = Path(w.tokenized_cache_dir)
    train_path = tokenized_cache_dir / "train"
    eval_path = tokenized_cache_dir / "eval"

    if not train_path.exists():
        raise RuntimeError(
            f"Tokenized train dataset missing at {train_path}. "
            "Run the CPU pretokenize step before training."
        )
    if w.eval_samples and not eval_path.exists():
        raise RuntimeError(
            f"Tokenized eval dataset missing at {eval_path}. "
            "Run the CPU pretokenize step before training."
        )

    from datasets import load_from_disk

    train_ds = load_from_disk(str(train_path))
    eval_ds = load_from_disk(str(eval_path)) if w.eval_samples else None

    model_dir = _local_model_dir(w.model_id)
    if not (model_dir / "config.json").exists():
        raise RuntimeError(
            f"Model not found in volume at {model_dir}. Run predownload first."
        )

    # ----
    # Model + LoRA (BF16 LoRA under FSDP)
    # ----
    import unsloth  # must be imported before transformers for full optimizations

    _patch_transformers_gpt_oss_init()
    from unsloth import FastLanguageModel

    lora_r = int(os.environ.get("LORA_R", "16"))
    lora_alpha = int(os.environ.get("LORA_ALPHA", str(lora_r * 2)))

    if is_main:
        print(
            f"[*] Loading model on CPU for FSDP: {w.model_id} (LoRA r={lora_r} alpha={lora_alpha})",
            flush=True,
        )

    # NOTE: For FSDP, use BF16 weights (no bitsandbytes 4-bit) to avoid FSDP+quant
    # incompatibilities. This is primarily a multi-GPU debug/scale path.
    os.environ["UNSLOTH_MODEL_NAME"] = ""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_dir),
        max_seq_length=w.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        full_finetuning=False,
        unsloth_tiled_mlp=True,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    FastLanguageModel.for_training(model)
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        use_rslora=True,
        max_seq_length=w.max_seq_length,
    )

    # Keep LoRA weights replicated (not sharded) so we can save adapters cheaply.
    ignored_modules = _collect_lora_modules(model)
    if is_main:
        print(f"[*] FSDP ignored_modules (LoRA layers): {len(ignored_modules)}")

    dec_cls = _find_decoder_layer_cls(model)
    if dec_cls is None:
        raise RuntimeError("Could not find a *DecoderLayer class for FSDP auto-wrap.")

    auto_wrap = partial(transformer_auto_wrap_policy, transformer_layer_cls={dec_cls})
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap,
        ignored_modules=ignored_modules,
        device_id=device,
        use_orig_params=True,
        sync_module_states=True,
        limit_all_gathers=True,
        mixed_precision=mp_policy,
    )

    # ----
    # Data
    # ----
    class TokenizedCompletionDataset(TorchDataset):
        def __init__(self, hf_ds):
            self._ds = hf_ds

        def __len__(self) -> int:
            return len(self._ds)

        def __getitem__(self, idx: int) -> dict:
            row = self._ds[int(idx)]
            return {
                "input_ids": row["input_ids"],
                "completion_mask": row["completion_mask"],
            }

    class CompletionOnlyCollator:
        def __init__(self, pad_token_id: int):
            self.pad_token_id = int(pad_token_id)

        def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
            max_len = max(len(x["input_ids"]) for x in batch)
            input_ids, attention_mask, labels = [], [], []
            for x in batch:
                ids = torch.tensor(x["input_ids"], dtype=torch.long)
                cm = torch.tensor(x["completion_mask"], dtype=torch.long)

                lbl = ids.clone()
                lbl[cm == 0] = -100

                att = torch.ones_like(ids, dtype=torch.long)
                pad_len = max_len - ids.numel()
                if pad_len:
                    ids = torch.cat(
                        [
                            ids,
                            torch.full(
                                (pad_len,),
                                self.pad_token_id,
                                dtype=torch.long,
                            ),
                        ],
                        dim=0,
                    )
                    att = torch.cat([att, torch.zeros(pad_len, dtype=torch.long)], dim=0)
                    lbl = torch.cat(
                        [lbl, torch.full((pad_len,), -100, dtype=torch.long)], dim=0
                    )

                input_ids.append(ids)
                attention_mask.append(att)
                labels.append(lbl)

            return {
                "input_ids": torch.stack(input_ids, dim=0),
                "attention_mask": torch.stack(attention_mask, dim=0),
                "labels": torch.stack(labels, dim=0),
            }

    train_tds = TokenizedCompletionDataset(train_ds)
    eval_tds = TokenizedCompletionDataset(eval_ds) if eval_ds is not None else None

    per_device_train_batch_size = int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "2"))
    per_device_eval_batch_size = int(os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", "2"))
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", "4"))
    lr = float(os.environ.get("LEARNING_RATE", "2e-5"))
    grad_clip = float(os.environ.get("GRAD_CLIP", "1.0"))
    log_every = int(os.environ.get("LOG_EVERY", "1"))
    save_steps = int(os.environ.get("SAVE_STEPS", "16"))
    eval_steps = int(os.environ.get("EVAL_STEPS", "16"))

    if is_main:
        eff = per_device_train_batch_size * world_size * max(1, grad_accum_steps)
        print(
            "[*] Train config: "
            f"train_rows={len(train_tds)} eval_rows={(len(eval_tds) if eval_tds is not None else 0)} "
            f"max_len={w.max_seq_length} micro={per_device_train_batch_size} "
            f"grad_accum={grad_accum_steps} global_batch={eff} lr={lr} max_steps={w.max_steps}",
            flush=True,
        )

    collate = CompletionOnlyCollator(pad_token_id=int(tokenizer.pad_token_id))

    train_sampler = DistributedSampler(
        train_tds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    )
    train_loader = DataLoader(
        train_tds,
        batch_size=per_device_train_batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=collate,
    )

    eval_loader = None
    if eval_tds is not None:
        eval_sampler = DistributedSampler(
            eval_tds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        eval_loader = DataLoader(
            eval_tds,
            batch_size=per_device_eval_batch_size,
            sampler=eval_sampler,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=collate,
        )

    # ----
    # Optimizer (LoRA params only)
    # ----
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found (LoRA not applied?)")

    optimizer = None
    try:
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(trainable, lr=lr)
        if is_main:
            print("[*] Optimizer: bitsandbytes AdamW8bit", flush=True)
    except Exception:
        from torch.optim import AdamW

        optimizer = AdamW(trainable, lr=lr)
        if is_main:
            print("[*] Optimizer: torch AdamW", flush=True)

    # Simple linear warmup then constant LR (keep it predictable for debug).
    warmup_steps = int(os.environ.get("WARMUP_STEPS", "1"))
    warmup_steps = max(1, warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ----
    # Helpers
    # ----
    def run_eval(step: int) -> None:
        if eval_loader is None:
            return

        model.train()  # Flex Attention eval path can differ in eval(); keep train-mode.
        total_loss = torch.zeros((), device=device, dtype=torch.float32)
        total_tokens = torch.zeros((), device=device, dtype=torch.float32)

        with torch.inference_mode():
            for batch in eval_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                active = (batch["labels"] != -100).sum().to(torch.float32)
                if active.item() == 0:
                    continue
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(**batch)
                    loss = out.loss.to(torch.float32)
                total_loss += loss * active
                total_tokens += active

        if is_dist:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

        if is_main:
            mean_loss = (total_loss / torch.clamp_min(total_tokens, 1.0)).item()
            ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")
            print(
                f"[eval step {step}] loss={mean_loss:.4f} ppl={ppl:.2f} tokens={int(total_tokens.item())}",
                flush=True,
            )

    def allreduce_trainable_grads() -> None:
        # FSDP does not necessarily synchronize grads for parameters in
        # `ignored_modules`. LoRA params are small, so we all-reduce explicitly.
        if (not is_dist) or world_size <= 1:
            return
        for p in trainable:
            if p.grad is None:
                continue
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)

    def save_adapter(out_dir: Path) -> None:
        if not is_main:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        model_to_save = model.module if isinstance(model, FSDP) else model
        lora_state = {
            n: p.detach().cpu()
            for n, p in model_to_save.named_parameters()
            if "lora_" in n
        }
        model_to_save.save_pretrained(str(out_dir), state_dict=lora_state)
        tokenizer.save_pretrained(str(out_dir))
        print(f"[+] Saved LoRA adapter to {out_dir}", flush=True)

    # ----
    # Training
    # ----
    model.train()
    if is_dist:
        dist.barrier()

    t0 = time.time()
    global_step = 0
    micro_step = 0

    run_dir = Path(w.run_dir)
    checkpoints_dir = run_dir / "checkpoints"
    final_adapter_dir = run_dir / "lora_adapter"

    for epoch in range(1):
        train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)

        for batch in train_loader:
            micro_step += 1
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            sync = (micro_step % max(1, grad_accum_steps)) == 0
            sync_ctx = nullcontext() if sync else model.no_sync()

            with sync_ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(**batch)
                    loss = out.loss / max(1, grad_accum_steps)
                loss.backward()

            if not sync:
                continue

            allreduce_trainable_grads()
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if is_main and (global_step <= 5 or global_step % max(1, log_every) == 0):
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"[step {global_step}] loss={loss.detach().float().item() * max(1, grad_accum_steps):.4f} lr={lr_now:.2e}",
                    flush=True,
                )

            if eval_steps and global_step % eval_steps == 0:
                run_eval(global_step)

            if save_steps and global_step % save_steps == 0:
                save_adapter(checkpoints_dir / f"step_{global_step}")

            if w.max_steps and w.max_steps > 0 and global_step >= w.max_steps:
                break

        if w.max_steps and w.max_steps > 0 and global_step >= w.max_steps:
            break

    # Final save
    if is_dist:
        dist.barrier()
    save_adapter(final_adapter_dir)

    if is_main:
        dt = time.time() - t0
        max_mem_gb = torch.cuda.max_memory_allocated(device=device) / (1024**3)
        print(
            f"[+] Train done. steps={global_step} wall={dt:.1f}s max_mem={max_mem_gb:.1f}GB",
            flush=True,
        )

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


@app.function(
    image=image,
    volumes={
        "/root/data": data_volume,
        "/root/model": model_volume,
        "/root/hf_cache": hf_cache_volume,
    },
    secrets=_secrets,
    timeout=21600,
    cpu=16.0,
    memory=262144,
)
def predownload_model(model_id: str = DEFAULT_MODEL_ID):
    import json
    from huggingface_hub import snapshot_download

    _ensure_hf_env()
    model_volume.reload()
    hf_cache_volume.reload()

    local_dir = _local_model_dir(model_id)

    def snapshot_complete(path: Path) -> bool:
        if not path.exists():
            return False
        for idx_name in (
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ):
            idx_path = path / idx_name
            if not idx_path.exists():
                continue
            try:
                with idx_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                weight_map = data.get("weight_map") or {}
                needed = {Path(v).name for v in weight_map.values()}
                missing = [
                    name for name in sorted(needed) if not (path / name).exists()
                ]
                if missing:
                    print(
                        f"[warn] model snapshot incomplete ({len(missing)} missing shards), will resume download"
                    )
                    return False
                return True
            except Exception:
                return False
        return (path / "config.json").exists() and any(path.glob("*.safetensors"))

    if snapshot_complete(local_dir):
        print(f"[skip] model already present: {local_dir}")
        return str(local_dir)

    cache_dir = Path("/root/model/.hf_cache")
    print(f"[*] Downloading model {model_id} -> {cache_dir}")
    snapshot_path = snapshot_download(
        repo_id=model_id,
        repo_type="model",
        cache_dir=str(cache_dir),
        token=os.environ.get("HF_TOKEN"),
        resume_download=True,
    )

    print(f"[*] Linking {local_dir} -> {snapshot_path}")
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    if local_dir.is_symlink():
        if str(local_dir.resolve()) != str(snapshot_path):
            local_dir.unlink()
            local_dir.symlink_to(snapshot_path, target_is_directory=True)
    elif not local_dir.exists():
        local_dir.symlink_to(snapshot_path, target_is_directory=True)

    print("[*] Committing volumes...")
    model_volume.commit()
    hf_cache_volume.commit()
    return str(local_dir)


@app.function(
    image=image,
    volumes={
        "/root/data": data_volume,
        "/root/model": model_volume,
        "/root/hf_cache": hf_cache_volume,
    },
    secrets=_secrets,
    timeout=21600,
    cpu=16.0,
    memory=262144,
)
def predownload_dataset(
    dataset_id: str = DEFAULT_DATASET_ID,
    split: str = DEFAULT_DATASET_SPLIT,
):
    from huggingface_hub import snapshot_download

    _ensure_hf_env()
    data_volume.reload()
    hf_cache_volume.reload()

    local_dir = _local_dataset_dir(dataset_id)
    parquet_glob = str(local_dir / "data" / f"{split}-*.parquet")
    existing = sorted(Path(local_dir).glob(f"data/{split}-*.parquet"))
    if existing:
        print(f"[skip] dataset already present: {parquet_glob} ({len(existing)} files)")
        return str(local_dir)

    cache_dir = Path("/root/data/.hf_cache")
    print(f"[*] Downloading dataset {dataset_id} split={split} -> {cache_dir}")
    snapshot_path = snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        cache_dir=str(cache_dir),
        allow_patterns=[f"data/{split}-*.parquet", "README.md", ".gitattributes"],
        token=os.environ.get("HF_TOKEN"),
    )

    print(f"[*] Linking {local_dir} -> {snapshot_path}")
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    if not local_dir.exists():
        local_dir.symlink_to(snapshot_path, target_is_directory=True)

    downloaded = sorted(Path(local_dir).glob(f"data/{split}-*.parquet"))
    if not downloaded:
        raise RuntimeError(
            f"Download completed but no parquet files found at {parquet_glob}"
        )

    print(f"[+] Downloaded {len(downloaded)} parquet shards for split={split}")
    print("[*] Committing volumes...")
    data_volume.commit()
    hf_cache_volume.commit()
    return str(local_dir)


@app.function(
    image=image,
    volumes={
        "/root/data": data_volume,
        "/root/model": model_volume,
        "/root/hf_cache": hf_cache_volume,
    },
    secrets=_secrets,
    timeout=21600,
    cpu=16.0,
    memory=262144,
)
def pretokenize_high_part00(
    model_id: str = DEFAULT_MODEL_ID,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_split: str = DEFAULT_DATASET_SPLIT,
    train_samples: int = 8192,
    eval_samples: int = 256,
    max_seq_length: int = 131072,
):
    from transformers import AutoTokenizer

    _ensure_hf_env()
    model_volume.reload()
    data_volume.reload()
    hf_cache_volume.reload()

    model_dir = _local_model_dir(model_id)
    if not (model_dir / "config.json").exists():
        raise RuntimeError(
            f"Model not found in volume at {model_dir}. Run predownload first:\n"
            f"  modal run modal/fsdp_unsloth_training_benchmark.py --predownload-only --model-id {model_id}\n"
        )

    tokenized_cache_dir = _local_tokenized_dir(
        dataset_id=dataset_id,
        dataset_split=dataset_split,
        max_seq_length=int(max_seq_length),
        train_samples=int(train_samples),
        eval_samples=int(eval_samples),
    )

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    _tokenize_to_disk_if_needed(
        tokenizer=tokenizer,
        dataset_id=dataset_id,
        dataset_split=dataset_split,
        train_samples=int(train_samples),
        eval_samples=int(eval_samples),
        max_seq_length=int(max_seq_length),
        tokenized_cache_dir=tokenized_cache_dir,
    )

    print("[*] Committing volumes...")
    data_volume.commit()
    hf_cache_volume.commit()
    return str(tokenized_cache_dir)


@app.function(
    image=image,
    gpu="B200:8",
    volumes={
        "/root/data": data_volume,
        "/root/model": model_volume,
        "/root/hf_cache": hf_cache_volume,
    },
    secrets=_secrets,
    timeout=21600,
    cpu=32.0,
    memory=524288,  # allow large CPU RAM for multi-proc load + tokenization
)
def train_high_part00_fsdp(
    model_id: str = DEFAULT_MODEL_ID,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_split: str = DEFAULT_DATASET_SPLIT,
    train_samples: int = 8192,
    eval_samples: int = 256,
    max_seq_length: int = 131072,
    max_steps: int = -1,
    strategy: str = "ddp_qlora",
    load_in_4bit: bool = True,
):
    import subprocess
    import torch

    _ensure_hf_env()
    model_volume.reload()
    data_volume.reload()
    hf_cache_volume.reload()

    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError(f"Expected multi-GPU instance, but saw {world_size} GPUs.")
    if world_size != 8:
        print(f"[warn] expected 8 GPUs, but saw {world_size}. Proceeding anyway.")

    out_root = Path("/root/model/finetuned")
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = (
        out_root
        / _sanitize_for_path(dataset_id)
        / dataset_split
        / _sanitize_for_path(model_id)
        / f"run_{int(time.time())}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenized_cache_dir = _local_tokenized_dir(
        dataset_id=dataset_id,
        dataset_split=dataset_split,
        max_seq_length=int(max_seq_length),
        train_samples=int(train_samples),
        eval_samples=int(eval_samples),
    )

    script_path = Path(__file__).resolve()
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={world_size}",
        str(script_path),
        "--worker",
        "--strategy",
        strategy,
        "--load-in-4bit",
        ("1" if load_in_4bit else "0"),
        "--model-id",
        model_id,
        "--dataset-id",
        dataset_id,
        "--dataset-split",
        dataset_split,
        "--train-samples",
        str(int(train_samples)),
        "--eval-samples",
        str(int(eval_samples)),
        "--max-seq-length",
        str(int(max_seq_length)),
        "--max-steps",
        str(int(max_steps)),
        "--run-dir",
        str(run_dir),
        "--tokenized-cache-dir",
        str(tokenized_cache_dir),
    ]

    print(f"[*] torchrun: {' '.join(cmd[:6])} ...")
    subprocess.run(cmd, check=True)

    print("[*] Committing volumes...")
    model_volume.commit()
    hf_cache_volume.commit()
    data_volume.commit()

    return str(run_dir)


@app.function(
    image=image,
    gpu="B200:1",
    volumes={
        "/root/data": data_volume,
        "/root/model": model_volume,
        "/root/hf_cache": hf_cache_volume,
    },
    secrets=_secrets,
    timeout=21600,
    cpu=16.0,
    memory=262144,
)
def merge_run_to_mxfp4(
    run_dir: str,
    base_model_id: str,
    max_seq_length: int = 131072,
):
    from unsloth import FastLanguageModel
    from unsloth.save import unsloth_generic_save_pretrained_merged
    from peft import PeftModel
    from transformers import AutoTokenizer

    _ensure_hf_env()
    model_volume.reload()
    hf_cache_volume.reload()
    _patch_transformers_gpt_oss_init()
    os.environ.setdefault("UNSLOTH_ENABLE_FLEX_ATTENTION", "1")
    os.environ["UNSLOTH_MODEL_NAME"] = ""

    run_path = Path(run_dir)
    adapter_dir = run_path / "lora_adapter"
    out_dir = run_path / "merged_mxfp4"

    if not (adapter_dir / "adapter_config.json").exists():
        raise RuntimeError(f"Adapter not found at {adapter_dir}")

    base_dir = _local_model_dir(base_model_id)
    if not (base_dir / "config.json").exists():
        raise RuntimeError(f"Base model not found in volume at {base_dir}")

    print(
        f"[*] Loading MXFP4 base model for merge: {base_model_id} (load_in_4bit=False)"
    )
    base_model, _ = FastLanguageModel.from_pretrained(
        model_name=str(base_dir),
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
        full_finetuning=False,
        unsloth_tiled_mlp=True,
        device_map={"": 0},
    )
    try:
        base_model.config._name_or_path = base_model_id
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    merged_model = PeftModel.from_pretrained(
        base_model, str(adapter_dir), is_trainable=False
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Saving merged MXFP4 model to {out_dir} ...")
    unsloth_generic_save_pretrained_merged(
        merged_model,
        str(out_dir),
        tokenizer=tokenizer,
        save_method="mxfp4",
    )
    print(f"[+] Saved merged MXFP4 model to {out_dir}")

    model_volume.commit()
    hf_cache_volume.commit()
    return str(out_dir)


@app.local_entrypoint()
def main(
    model_id: str = DEFAULT_MODEL_ID,
    fallback_model_id: str = DEFAULT_FALLBACK_MODEL_ID,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_split: str = DEFAULT_DATASET_SPLIT,
    train_samples: int = 8192,
    eval_samples: int = 256,
    max_seq_length: int = 131072,
    max_steps: int = -1,
    predownload_only: bool = False,
    save_merged_mxfp4: bool = True,
    strategy: str = "ddp_qlora",
    load_in_4bit: bool = True,
):
    predownload_model.remote(model_id=model_id)
    predownload_dataset.remote(dataset_id=dataset_id, split=dataset_split)
    pretokenize_high_part00.remote(
        model_id=model_id,
        dataset_id=dataset_id,
        dataset_split=dataset_split,
        train_samples=train_samples,
        eval_samples=eval_samples,
        max_seq_length=max_seq_length,
    )
    if predownload_only:
        return

    run_dir = None
    try:
        run_dir = train_high_part00_fsdp.remote(
            model_id=model_id,
            dataset_id=dataset_id,
            dataset_split=dataset_split,
            train_samples=train_samples,
            eval_samples=eval_samples,
            max_seq_length=max_seq_length,
            max_steps=max_steps,
            strategy=strategy,
            load_in_4bit=load_in_4bit,
        )
    except Exception as e:
        if not fallback_model_id or fallback_model_id == model_id:
            raise
        print(f"[warn] primary model training failed ({model_id}): {type(e).__name__}: {e}")
        print(f"[*] Falling back to: {fallback_model_id}")
        predownload_model.remote(model_id=fallback_model_id)
        pretokenize_high_part00.remote(
            model_id=fallback_model_id,
            dataset_id=dataset_id,
            dataset_split=dataset_split,
            train_samples=train_samples,
            eval_samples=eval_samples,
            max_seq_length=max_seq_length,
        )
        run_dir = train_high_part00_fsdp.remote(
            model_id=fallback_model_id,
            dataset_id=dataset_id,
            dataset_split=dataset_split,
            train_samples=train_samples,
            eval_samples=eval_samples,
            max_seq_length=max_seq_length,
            max_steps=max_steps,
            strategy=strategy,
            load_in_4bit=load_in_4bit,
        )

    if save_merged_mxfp4 and run_dir:
        suffix = "-unsloth-bnb-4bit"
        base_model_id = (
            model_id[: -len(suffix)] if model_id.endswith(suffix) else model_id
        )
        predownload_model.remote(model_id=base_model_id)
        merge_run_to_mxfp4.remote(
            run_dir=run_dir,
            base_model_id=base_model_id,
            max_seq_length=max_seq_length,
        )


if __name__ == "__main__":
    # When invoked via `torchrun` inside the Modal GPU container, run the worker.
    import argparse

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--strategy",
        type=str,
        default="ddp_qlora",
        choices=["ddp_qlora", "fsdp_bf16"],
    )
    parser.add_argument("--load-in-4bit", type=int, default=1)
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--dataset-id", type=str, required=True)
    parser.add_argument("--dataset-split", type=str, required=True)
    parser.add_argument("--train-samples", type=int, required=True)
    parser.add_argument("--eval-samples", type=int, required=True)
    parser.add_argument("--max-seq-length", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--tokenized-cache-dir", type=str, required=True)
    # torchrun may inject this flag in some configurations; accept it even though
    # we read ranks from environment variables.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=None)

    args, _unknown = parser.parse_known_args()
    if not args.worker:
        raise SystemExit(
            "This file is meant to be run via `modal run ...` (client) or "
            "`torchrun ... --worker` (inside the container)."
        )

    w = _WorkerArgs(
        strategy=str(args.strategy),
        load_in_4bit=bool(int(args.load_in_4bit)),
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        dataset_split=args.dataset_split,
        train_samples=int(args.train_samples),
        eval_samples=int(args.eval_samples),
        max_seq_length=int(args.max_seq_length),
        max_steps=int(args.max_steps),
        run_dir=str(args.run_dir),
        tokenized_cache_dir=str(args.tokenized_cache_dir),
    )
    if args.strategy == "ddp_qlora":
        _ddp_qlora_train_worker(w)
    else:
        _fsdp_bf16_train_worker(w)
