# modal/unsloth_training_benchmark.py
# gpt-oss-120b + Unsloth 4-bit + Modal B200.
#
# This version:
# - Predownloads model + dataset into Modal Volumes (one-time) before training
# - Trains on `radna0/nemotron-math-v2-harmony-tools`, split `high_part00`
# - Dataset is already Harmony-formatted in a single `text` column; we create
#   prompt+completion pairs so TRL can apply completion-only loss.

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable

import modal

APP_NAME = "gpt-oss-120b-unsloth-benchmark"

DEFAULT_MODEL_ID = os.environ.get("MODEL_ID", "unsloth/gpt-oss-120b")
DEFAULT_FALLBACK_MODEL_ID = os.environ.get(
    "FALLBACK_MODEL_ID", "unsloth/gpt-oss-120b-unsloth-bnb-4bit"
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
# Volumes
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
    # Core stack (no accelerate CLI)
    .run_commands(
        "python -m pip install "
        "numpy==2.2.0 datasets==3.2.0 accelerate==1.10.1 trl==0.22.2 peft bitsandbytes "
        "sentencepiece protobuf msgspec ninja wandb huggingface_hub hf_transfer "
        "transformers==4.57.3 timm"
    )
    # Unsloth
    .run_commands(
        "python -m pip install git+https://github.com/unslothai/unsloth-zoo.git",
        "python -m pip install git+https://github.com/unslothai/unsloth.git",
    )
    .run_commands(
        "pip install --upgrade --force-reinstall --no-cache-dir --no-deps "
        "unsloth unsloth_zoo transformers==4.57.3 timm"
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
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


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
    # Keep a stable, human-readable location in the model volume.
    return Path("/root/model") / model_id


def _local_dataset_dir(dataset_id: str) -> Path:
    # Keep a stable, human-readable location in the data volume.
    return Path("/root/data/datasets") / dataset_id


@app.function(
    image=image,
    volumes={
        "/root/data": data_volume,
        "/root/model": model_volume,
        "/root/hf_cache": hf_cache_volume,
    },
    secrets=_secrets,
    timeout=21600,  # downloads can take a long time for 120B
    cpu=16.0,
    memory=262144,
)
def predownload_model(model_id: str = DEFAULT_MODEL_ID):
    import json
    from huggingface_hub import snapshot_download

    _ensure_hf_env()

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
        # Fallback: if we at least have some weights, assume complete.
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
        # Update if pointing somewhere else (eg revision change)
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
    gpu="B200:1",
    volumes={
        "/root/data": data_volume,
        "/root/model": model_volume,
        "/root/hf_cache": hf_cache_volume,
    },
    secrets=_secrets,
    timeout=21600,
    cpu=16.0,
    memory=262144,  # 256GB for 120B model loading
)
def train_high_part00(
    model_id: str = DEFAULT_MODEL_ID,
    fallback_model_id: str = DEFAULT_FALLBACK_MODEL_ID,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_split: str = DEFAULT_DATASET_SPLIT,
    train_samples: int = 8192,
    eval_samples: int = 256,
    max_seq_length: int = 131072,  # 128K context
    max_steps: int = -1,  # set to a small value for a quick smoke test
    save_merged_mxfp4: bool = True,
):
    import torch
    import math
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    _ensure_hf_env()
    _patch_transformers_gpt_oss_init()
    os.environ.setdefault("UNSLOTH_ENABLE_FLEX_ATTENTION", "1")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # ----
    # Model (local path, predownloaded)
    # ----
    def require_model_dir(mid: str) -> Path:
        d = _local_model_dir(mid)
        if (d / "config.json").exists():
            return d
        raise RuntimeError(
            f"Model not found in volume at {d}. Run predownload first:\n"
            f"  modal run modal/unsloth_training_benchmark.py --predownload-only --model-id {mid}\n"
        )

    candidates = [model_id]
    if fallback_model_id and fallback_model_id != model_id:
        candidates.append(fallback_model_id)

    model = None
    tokenizer = None
    loaded_model_id = None
    load_failures: list[str] = []

    for candidate in candidates:
        try:
            model_dir = require_model_dir(candidate)
            print(f"[*] Loading model ({candidate}) from: {model_dir}")
            # Unsloth accumulates `UNSLOTH_MODEL_NAME` across loads; reset to avoid
            # stale `_load_in_4bit_` flags affecting subsequent loads.
            os.environ["UNSLOTH_MODEL_NAME"] = ""
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(model_dir),
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=True,
                full_finetuning=False,
                unsloth_tiled_mlp=True,
                device_map={"": 0},
            )
            loaded_model_id = candidate
            break
        except Exception as e:
            load_failures.append(f"{candidate}: {type(e).__name__}: {e}")
            print(f"[warn] model load failed for {candidate}: {type(e).__name__}: {e}")

    if model is None or tokenizer is None or loaded_model_id is None:
        raise RuntimeError(
            "All model load attempts failed:\n- " + "\n- ".join(load_failures)
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora_r = int(os.environ.get("LORA_R", "16"))
    lora_alpha = int(os.environ.get("LORA_ALPHA", str(lora_r * 2)))
    print(f"[*] Applying LoRA: r={lora_r} alpha={lora_alpha}")

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
        random_state=3407,
        use_rslora=True,
        max_seq_length=max_seq_length,
    )

    # ----
    # Dataset (local parquet shards, predownloaded)
    # ----
    dataset_dir = _local_dataset_dir(dataset_id)
    parquet_files = sorted((dataset_dir / "data").glob(f"{dataset_split}-*.parquet"))
    if not parquet_files:
        raise RuntimeError(
            f"Dataset split not found in volume at {dataset_dir}/data/{dataset_split}-*.parquet.\n"
            f"Run predownload first:\n"
            f"  modal run modal/unsloth_training_benchmark.py --predownload-only\n"
        )

    need = train_samples + eval_samples
    print(f"[*] Tokenizing {need} rows from {len(parquet_files)} parquet shards...")

    train_tokenized = []
    eval_tokenized = []
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

    train_ds = Dataset.from_list(train_tokenized)
    eval_ds = Dataset.from_list(eval_tokenized) if eval_tokenized else None
    print(
        f"[*] Tokenized rows: train={len(train_ds)} eval={(len(eval_ds) if eval_ds is not None else 0)} "
        f"seen={total_seen} skipped_no_completion={skipped_no_completion}"
    )

    # ----
    # TRL SFT (completion-only loss)
    # ----
    out_root = Path("/root/model/finetuned")
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = (
        out_root
        / dataset_id.replace("/", "__")
        / dataset_split
        / loaded_model_id.replace("/", "__")
        / f"run_{int(time.time())}"
    )
    trainer_out_dir = run_dir / "trainer"
    adapter_dir = run_dir / "lora_adapter"
    merged_mxfp4_dir = run_dir / "merged_mxfp4"
    merged_forced_4bit_dir = run_dir / "merged_forced_4bit"

    per_device_train_batch_size = int(
        os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "4")
    )
    per_device_eval_batch_size = int(os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", "64"))
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", "16"))
    lr = float(os.environ.get("LEARNING_RATE", "2e-5"))

    sft_args = SFTConfig(
        output_dir=str(trainer_out_dir),
        do_train=True,
        do_eval=bool(eval_ds is not None),
        eval_strategy="steps" if bool(eval_ds) else "no",
        eval_steps=16,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        max_steps=max_steps,
        num_train_epochs=1.0,
        learning_rate=lr,
        logging_steps=1,
        warmup_steps=1,
        optim="adamw_8bit",
        seed=3407,
        report_to="none",
        save_strategy="steps",
        save_steps=16,
        max_length=max_seq_length,
        packing=True,
        completion_only_loss=True,
        bf16=True,
        tf32=True,
        ddp_find_unused_parameters=False,
    )

    print(
        "[*] Training config: "
        f"train_rows={len(train_ds)} eval_rows={(len(eval_ds) if eval_ds is not None else 0)} "
        f"max_length={max_seq_length} batch={per_device_train_batch_size} "
        f"grad_accum={grad_accum_steps} max_steps={max_steps}"
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_args,
    )

    def eval_in_train_mode() -> dict:
        if eval_ds is None:
            return {}

        model_ref = trainer.model
        was_training = model_ref.training
        model_ref.train()

        losses = []
        with torch.inference_mode():
            for inputs in trainer.get_eval_dataloader():
                loss, _, _ = trainer.prediction_step(
                    model_ref,
                    inputs,
                    prediction_loss_only=True,
                    ignore_keys=None,
                )
                if loss is None:
                    continue
                losses.append(loss.detach().float().cpu())

        if not was_training:
            model_ref.eval()

        if not losses:
            return {}

        mean_loss = torch.stack(losses).mean().item()
        return {
            "eval_loss": mean_loss,
            "eval_ppl": (math.exp(mean_loss) if mean_loss < 20 else float("inf")),
            "eval_steps": len(losses),
        }

    try:
        t0 = time.time()
        trainer_stats = trainer.train()
        dt = time.time() - t0
        max_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(
            f"[+] Train complete. wall={dt:.1f}s "
            f"train_runtime={trainer_stats.metrics.get('train_runtime')} "
            f"max_mem={max_mem_gb:.1f}GB"
        )

        if eval_ds is not None:
            metrics = eval_in_train_mode()
            if metrics:
                print(f"[+] Eval metrics (Flex Attention, train-mode): {metrics}")

        adapter_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        print(f"[+] Saved LoRA adapter to {adapter_dir}")
        return str(run_dir)
    finally:
        print("[*] Committing volumes...")
        model_volume.commit()
        hf_cache_volume.commit()
        data_volume.commit()


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
    # Ensure save utilities resolve the correct base repo id.
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
):
    predownload_model.remote(model_id=model_id)
    predownload_dataset.remote(dataset_id=dataset_id, split=dataset_split)
    if predownload_only:
        return

    run_dir = None
    try:
        run_dir = train_high_part00.remote(
            model_id=model_id,
            fallback_model_id="",
            dataset_id=dataset_id,
            dataset_split=dataset_split,
            train_samples=train_samples,
            eval_samples=eval_samples,
            max_seq_length=max_seq_length,
            max_steps=max_steps,
            save_merged_mxfp4=save_merged_mxfp4,
        )
    except Exception as e:
        if not fallback_model_id or fallback_model_id == model_id:
            raise
        print(
            f"[warn] primary model training failed ({model_id}): {type(e).__name__}: {e}"
        )
        print(f"[*] Falling back to: {fallback_model_id}")
        predownload_model.remote(model_id=fallback_model_id)
        run_dir = train_high_part00.remote(
            model_id=fallback_model_id,
            fallback_model_id="",
            dataset_id=dataset_id,
            dataset_split=dataset_split,
            train_samples=train_samples,
            eval_samples=eval_samples,
            max_seq_length=max_seq_length,
            max_steps=max_steps,
            save_merged_mxfp4=save_merged_mxfp4,
        )

    if save_merged_mxfp4 and run_dir:
        # Merge in a fresh Modal invocation so Unsloth's 4-bit patching from training
        # doesn't persist and break MXFP4 base model loading.
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
