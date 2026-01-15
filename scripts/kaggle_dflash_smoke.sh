#!/usr/bin/env bash
set -euo pipefail

cd /kaggle/working

nvidia-smi -L || true
python -V

# Avoid importing TensorFlow / JAX / Flax in Transformers on Kaggle (can pull in
# incompatible binary deps like sklearn vs numpy).
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export TRANSFORMERS_NO_JAX=1

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
PY

PYDEPS=/kaggle/working/.pydeps
mkdir -p "$PYDEPS"

# Kaggle system site-packages can be read-only; install into a local target.
python -m pip -q install --no-cache-dir -U -t "$PYDEPS" safetensors
python -m pip -q install --no-cache-dir -U --no-deps -t "$PYDEPS" sglang==0.5.7
export PYTHONPATH="$PYDEPS:${PYTHONPATH:-}"

python - <<'PY'
import sglang
print("sglang", getattr(sglang, "__version__", "unknown"))
PY

export SGLANG_OVERLAY_SRC=/kaggle/working/cuda-norm-sync/sglang_overlay/sglang
python /kaggle/working/cuda-norm-sync/scripts/sglang_overlay_install.py

# converter unit smoke (CPU-only; validates key rewrite).
python /kaggle/working/cuda-norm-sync/scripts/test_convert_hf_dflash_ckpt_to_sglang_smoke.py

# end-to-end: create tiny HF-style weights (torch), convert, then load via SGLang DFlashDraftModel
python - <<'PY'
import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

src_root = Path("/kaggle/working")
with tempfile.TemporaryDirectory(dir=str(src_root)) as td:
    td = Path(td)
    src = td / "hf"
    dst = td / "sg"
    src.mkdir(parents=True, exist_ok=True)

    cfg = {
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_target_layers": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 32,
        "max_position_embeddings": 512,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "block_size": 8,
        "mlp_ratio": 2.0,
        "hidden_act": "silu",
    }
    (src / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    inter = int(round(cfg["hidden_size"] * cfg["mlp_ratio"]))
    weights = {
        # Minimal attention weights (HF-style q/k/v/o) which SGLang's fused QKV
        # loader should accept.
        "layers.0.self_attn.q_proj.weight": torch.randn(cfg["hidden_size"], cfg["hidden_size"], dtype=torch.float16),
        "layers.0.self_attn.k_proj.weight": torch.randn(cfg["hidden_size"], cfg["hidden_size"], dtype=torch.float16),
        "layers.0.self_attn.v_proj.weight": torch.randn(cfg["hidden_size"], cfg["hidden_size"], dtype=torch.float16),
        "layers.0.self_attn.o_proj.weight": torch.randn(cfg["hidden_size"], cfg["hidden_size"], dtype=torch.float16),
        "layers.1.self_attn.q_proj.weight": torch.randn(cfg["hidden_size"], cfg["hidden_size"], dtype=torch.float16),
        "layers.1.self_attn.k_proj.weight": torch.randn(cfg["hidden_size"], cfg["hidden_size"], dtype=torch.float16),
        "layers.1.self_attn.v_proj.weight": torch.randn(cfg["hidden_size"], cfg["hidden_size"], dtype=torch.float16),
        "layers.1.self_attn.o_proj.weight": torch.randn(cfg["hidden_size"], cfg["hidden_size"], dtype=torch.float16),
        # Layer norms.
        "layers.0.input_layernorm.weight": torch.ones(cfg["hidden_size"], dtype=torch.float16),
        "layers.0.post_attention_layernorm.weight": torch.ones(cfg["hidden_size"], dtype=torch.float16),
        "layers.1.input_layernorm.weight": torch.ones(cfg["hidden_size"], dtype=torch.float16),
        "layers.1.post_attention_layernorm.weight": torch.ones(cfg["hidden_size"], dtype=torch.float16),
        "layers.0.mlp.gate_up.weight": torch.randn(
            2 * inter, cfg["hidden_size"], dtype=torch.float16
        ),
        "layers.0.mlp.down.weight": torch.randn(
            inter, cfg["hidden_size"], dtype=torch.float16
        ),
        "layers.1.mlp.gate_up.weight": torch.randn(
            2 * inter, cfg["hidden_size"], dtype=torch.float16
        ),
        "layers.1.mlp.down.weight": torch.randn(
            inter, cfg["hidden_size"], dtype=torch.float16
        ),
        "mask_embedding": torch.zeros(cfg["hidden_size"], dtype=torch.float16),
        "fc.weight": torch.zeros(cfg["hidden_size"], cfg["hidden_size"], dtype=torch.float16),
        "hidden_norm.weight": torch.ones(cfg["hidden_size"], dtype=torch.float16),
        "norm.weight": torch.ones(cfg["hidden_size"], dtype=torch.float16),
    }
    save_file(weights, str(src / "model.safetensors"), metadata={"unit_test": "1"})

    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "/kaggle/working/cuda-norm-sync/scripts/convert_hf_dflash_ckpt_to_sglang.py",
            "--src",
            str(src),
            "--dst",
            str(dst),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "/kaggle/working/cuda-norm-sync/scripts/sglang_dflash_draft_load_smoke.py",
            "--ckpt",
            str(dst),
        ],
        check=True,
    )
PY

echo "[+] Kaggle DFlash smoke OK"
