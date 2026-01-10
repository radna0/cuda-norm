# modal/acclerate_training_benchmark.py
# gpt-oss-120b + Unsloth 4-bit + 1x B200 (Accelerate CLI)

import os
import subprocess
from pathlib import Path
import modal

APP_NAME = "gpt-oss-120b-unsloth-accelerate-benchmark"

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
    # Core stack
    .run_commands(
        "python -m pip install "
        "numpy==2.2.0 datasets==3.2.0 accelerate peft bitsandbytes "
        "sentencepiece protobuf msgspec ninja wandb huggingface_hub hf_transfer "
        "trl==0.22.2"
    )
    # Unsloth
    .run_commands(
        "python -m pip install git+https://github.com/unslothai/unsloth-zoo.git",
        "python -m pip install git+https://github.com/unslothai/unsloth.git",
    )
    .run_commands(
        "pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo transformers==4.56.2 timm trl==0.22.2"
    )
)

app = modal.App(APP_NAME)

# -------------------------
# Worker Content (Standalone Script)
# -------------------------
WORKER_SCRIPT = """
import os
import sys
import torch
import torch.distributed as dist
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
from huggingface_hub import snapshot_download
from pathlib import Path
import math

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    from datetime import timedelta
    if world_size > 1 and not dist.is_initialized():
        # Increase timeout for the massive 120B model load/download
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
    
    torch.cuda.set_device(local_rank)
    print(f"[*] [RANK {rank}] Local {local_rank} starting. World Size: {world_size}")

    model_dir = "/root/model/unsloth/gpt-oss-120b"
    repo_id = "unsloth/gpt-oss-120b"
    max_seq_length = 65536

    # 1. Weights Readiness Check
    if rank == 0:
        config_path = Path(model_dir) / "config.json"
        if not config_path.exists():
            print(f"[!] [RANK 0] FATAL: Weights not found in {model_dir}. Run download script first.")
            sys.exit(1)
        else:
            print(f"[+] [RANK 0] Weights verified in volume.")

    if world_size > 1:
        dist.barrier()

    # 2. Load Model
    print(f"[*] [RANK {rank}] Loading Unsloth 120B (4-bit) BEGIN...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
            full_finetuning=False,
            unsloth_tiled_mlp=True,
            device_map={"": local_rank}, 
        )
        print(f"[+] [RANK {rank}] Loading COMPLETE.")
    except Exception as e:
        print(f"[!] [RANK {rank}] Loading FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 3. Chat Template (Harmony-style)
    harmony_template = (
        "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
        "{% for message in messages %}"
        "<|start|>{{ message['role'] }}"
        "{% if message.get('channel') is not none %}<|channel|>{{ message['channel'] }}{% endif %}"
        "<|message|>{{ message['content'] }}<|end|>"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|start|>assistant<|channel|>final<|message|>{% endif %}"
    )
    tokenizer.chat_template = harmony_template
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. LoRA
    print(f"[*] [RANK {rank}] Applying LoRA (r=256)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=512,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=True,
        max_seq_length=max_seq_length,
    )

    # 5. Dataset Prep (Nemotron-Math-v2) - Load from pre-downloaded local files
    print(f"[*] [RANK {rank}] Loading and formatting dataset from local cache...")
    dataset_path = "/root/hf_cache/datasets/nvidia/Nemotron-Math-v2/data/high.part_00.jsonl"
    if not os.path.exists(dataset_path):
         # Fallback to secondary path found in ls -R
         dataset_path = "/root/hf_cache/hub/datasets--nvidia--Nemotron-Math-v2/data/low.jsonl"
    
    print(f"[*] [RANK {rank}] Using dataset file: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train", streaming=True)
    
    # Take samples and format
    samples = []
    print(f"[*] [RANK {rank}] Processing samples...")
    for item in dataset:
        messages = item.get("messages", [])
        if not messages: continue
        
        # Format reasoning_content to Harmony channels
        formatted_messages = []
        for m in messages:
            if m["role"] == "user":
                formatted_messages.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                # Thought (analysis channel)
                if m.get("reasoning_content"):
                    formatted_messages.append({
                        "role": "assistant",
                        "channel": "analysis",
                        "content": m["reasoning_content"]
                    })
                # Answer (final channel)
                formatted_messages.append({
                    "role": "assistant",
                    "channel": "final",
                    "content": m["content"]
                })
        
        text = tokenizer.apply_chat_template(formatted_messages, tokenize=False) + tokenizer.eos_token
        samples.append({"text": text})
        
        if len(samples) >= 2000: break # Benchmark size

    ds = Dataset.from_list(samples)

    # 6. Training Config
    max_steps = int(os.environ.get("BENCHMARK_MAX_STEPS", 10))
    print(f"[*] [RANK {rank}] Starting benchmark for {max_steps} steps...")
    
    sft_config = SFTConfig(
        max_steps=max_steps,
        learning_rate=2e-5,
        logging_steps=1,
        optim="adamw_8bit",
        seed=3407,
        output_dir="/root/data/outputs",
        save_strategy="no",
        report_to="none",
        per_device_train_batch_size=2, # Conservatively small for 120B on B200 
        gradient_accumulation_steps=4,
        max_seq_length=max_seq_length,
        padding_side="right",
        packing=True,
        dataset_text_field="text",
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        tokenizer=tokenizer,
        args=sft_config,
    )

    trainer.train()
    print(f"[*] [RANK {rank}] Benchmark complete.")

if __name__ == "__main__":
    main()
"""


@app.function(
    image=image,
    gpu="B200:1",
    volumes={
        "/root/data": data_volume,
        "/root/model": model_volume,
        "/root/hf_cache": hf_cache_volume,
    },
    timeout=7200,
    cpu=16.0,
    memory=262144,  # 256GB for 120B model loading
)
def launch_benchmark(max_steps: int = 10):
    script_path = "/root/train.py"
    with open(script_path, "w") as f:
        f.write(WORKER_SCRIPT)

    print(f"[*] Launching Unsloth benchmark via Accelerate (max_steps={max_steps})...")

    os.environ["BENCHMARK_MAX_STEPS"] = str(max_steps)
    os.environ["NCCL_TIMEOUT"] = "3600"
    os.environ.setdefault("HF_HOME", "/root/hf_cache")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    cmd = ["accelerate", "launch", "--num_processes", "1", script_path]

    # Commit volumes after work
    try:
        subprocess.run(cmd, check=True)
    finally:
        print("[*] Committing volumes...")
        model_volume.commit()
        hf_cache_volume.commit()
        data_volume.commit()

    print("[+] Activity finished.")


@app.local_entrypoint()
def main(max_steps: int = 10):
    launch_benchmark.remote(max_steps=max_steps)
