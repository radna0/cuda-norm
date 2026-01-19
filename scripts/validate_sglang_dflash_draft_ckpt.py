#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


def _die(msg: str) -> None:
    print(f"[!] {msg}", file=sys.stderr, flush=True)
    raise SystemExit(2)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Validate an HF-style SGLang DFlashDraftModel checkpoint directory.")
    ap.add_argument("--ckpt", required=True, help="Path to exported DFlashDraftModel dir (config.json + model*.safetensors).")
    args = ap.parse_args()

    root = Path(args.ckpt).expanduser().resolve()
    cfg_path = root / "config.json"
    if not cfg_path.exists():
        _die(f"Missing {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        _die("config.json is not a JSON object")

    arch = cfg.get("architectures")
    if not (isinstance(arch, list) and "DFlashDraftModel" in arch):
        _die(f"config.json architectures must include DFlashDraftModel, got {arch!r}")

    dflash_cfg = cfg.get("dflash_config")
    if not isinstance(dflash_cfg, dict):
        _die("config.json missing dflash_config")
    for k in ("block_size", "target_layer_ids", "mask_token", "mask_token_id"):
        if k not in dflash_cfg:
            _die(f"dflash_config missing {k}")

    num_layers = int(cfg.get("num_hidden_layers", 0) or 0)
    hidden = int(cfg.get("hidden_size", 0) or 0)
    if num_layers <= 0 or hidden <= 0:
        _die(f"Invalid num_hidden_layers={num_layers} hidden_size={hidden}")

    index_path = root / "model.safetensors.index.json"
    if not index_path.exists():
        _die(f"Missing {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        _die("index missing weight_map")

    required = [
        "model.fc.weight",
        "model.fc.bias",
        "model.hidden_norm.weight",
        "model.mask_embedding",
        "model.norm.weight",
    ]
    for i in range(num_layers):
        required.extend(
            [
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.q_proj.bias",
                f"model.layers.{i}.self_attn.k_proj.bias",
                f"model.layers.{i}.self_attn.v_proj.bias",
                f"model.layers.{i}.self_attn.o_proj.bias",
                f"model.layers.{i}.self_attn.q_norm.weight",
                f"model.layers.{i}.self_attn.k_norm.weight",
                f"model.layers.{i}.mlp.gate_up_proj.weight",
                f"model.layers.{i}.mlp.gate_up_proj.bias",
                f"model.layers.{i}.mlp.down_proj.weight",
                f"model.layers.{i}.mlp.down_proj.bias",
            ]
        )
    missing = [k for k in required if k not in weight_map]
    if missing:
        _die(f"Missing {len(missing)} required tensors (first 10): {missing[:10]}")

    # Shape sanity checks (sample a few tensors).
    from safetensors import safe_open

    def _read_tensor(name: str):
        shard = weight_map[name]
        with safe_open(str(root / shard), framework="numpy") as f:
            return f.get_tensor(name)

    fc_w = _read_tensor("model.fc.weight")
    if tuple(fc_w.shape)[0] != hidden:
        _die(f"model.fc.weight out dim mismatch: got {tuple(fc_w.shape)} expected out={hidden}")
    if tuple(fc_w.shape)[1] % hidden != 0:
        _die(f"model.fc.weight in dim must be multiple of hidden: got {tuple(fc_w.shape)} hidden={hidden}")
    k = tuple(fc_w.shape)[1] // hidden
    if k != int(dflash_cfg["target_layer_ids"].__len__()):
        print(
            "[!] Warning: inferred num_context_features from fc.weight "
            f"(K={k}) != len(target_layer_ids)={len(dflash_cfg['target_layer_ids'])}",
            flush=True,
        )

    print(
        "[+] OK: DFlashDraftModel checkpoint looks structurally valid\n"
        f"    ckpt={root}\n"
        f"    layers={num_layers} hidden={hidden} K={k} block_size={dflash_cfg.get('block_size')}\n"
        f"    target_layer_ids={dflash_cfg.get('target_layer_ids')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
