from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def _write_min_hf_dflash_ckpt(dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    cfg = {
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "num_target_layers": 8,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 16,
        "max_position_embeddings": 128,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "block_size": 8,
        "mlp_ratio": 2.0,
        "hidden_act": "silu",
    }
    (dst / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # Minimal weights: only tensors that exercise the rename/split logic.
    # gate_up is [2*inter, hidden]; inter = hidden * mlp_ratio = 8.
    weights = {
        "layers.0.mlp.gate_up.weight": np.arange(2 * 8 * 4, dtype=np.float32).reshape(16, 4),
        "layers.0.mlp.down.weight": np.arange(8 * 4, dtype=np.float32).reshape(8, 4),
        "mask_embedding": np.zeros((4,), dtype=np.float32),
        "fc.weight": np.zeros((4, 4), dtype=np.float32),
        "hidden_norm.weight": np.ones((4,), dtype=np.float32),
        "norm.weight": np.ones((4,), dtype=np.float32),
    }
    save_file(weights, str(dst / "model.safetensors"), metadata={"unit_test": "1"})


def _assert_keys(path: Path) -> None:
    # Our converter rewrites unsharded → sharded + index.json.
    idx = path / "model.safetensors.index.json"
    assert idx.exists(), f"missing index: {idx}"
    j = json.loads(idx.read_text(encoding="utf-8"))
    wmap = j["weight_map"]

    expected = {
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.down_proj.weight",
    }
    missing = sorted(k for k in expected if k not in wmap)
    assert not missing, f"missing rewritten keys: {missing}"

    forbidden = {
        "layers.0.mlp.gate_up.weight",
        "layers.0.mlp.down.weight",
    }
    present_forbidden = sorted(k for k in forbidden if k in wmap)
    assert not present_forbidden, f"forbidden keys still present: {present_forbidden}"

    # Spot-check that gate_proj/up_proj are each half of gate_up.
    gate_file = path / wmap["layers.0.mlp.gate_proj.weight"]
    up_file = path / wmap["layers.0.mlp.up_proj.weight"]
    with safe_open(str(gate_file), framework="np", device="cpu") as f:
        gate = f.get_tensor("layers.0.mlp.gate_proj.weight")
    with safe_open(str(up_file), framework="np", device="cpu") as f:
        up = f.get_tensor("layers.0.mlp.up_proj.weight")
    assert gate.shape == (8, 4)
    assert up.shape == (8, 4)
    assert int(gate[0, 0]) == 0
    assert int(up[0, 0]) == int(8 * 4)


def main() -> int:
    converter = Path(__file__).with_name("convert_hf_dflash_ckpt_to_sglang.py")
    if not converter.exists():
        raise FileNotFoundError(converter)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src"
        dst = td / "dst"
        _write_min_hf_dflash_ckpt(src)
        subprocess.run(
            [sys.executable, str(converter), "--src", str(src), "--dst", str(dst)],
            check=True,
        )
        _assert_keys(dst)

    print("[+] smoke test passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

