from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_weight_slices(model_dir: Path):
    from safetensors import safe_open

    idx_path = model_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        raise FileNotFoundError(idx_path)
    idx = _read_json(idx_path)
    weight_map = idx.get("weight_map", {})
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid weight_map in {idx_path}")

    # Iterate each shard once and yield (name, shape, dtype)
    for shard in sorted(set(weight_map.values())):
        shard_path = model_dir / shard
        with safe_open(str(shard_path), framework="np", device="cpu") as f:
            for name in f.keys():
                sl = f.get_slice(name)
                yield name, tuple(int(x) for x in sl.get_shape()), str(sl.get_dtype())


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="SGLang DFlashDraftModel checkpoint dir.")
    args = ap.parse_args()

    model_dir = Path(args.ckpt).expanduser().resolve()
    cfg_path = model_dir / "config.json"
    _expect(cfg_path.exists(), f"Missing {cfg_path}")
    cfg = _read_json(cfg_path)
    _expect(isinstance(cfg, dict), f"Invalid {cfg_path}")
    _expect(cfg.get("architectures") == ["DFlashDraftModel"], "config.architectures must be ['DFlashDraftModel']")

    hs = int(cfg["hidden_size"])
    nh = int(cfg["num_attention_heads"])
    nkv = int(cfg.get("num_key_value_heads", nh))
    hd = int(cfg.get("head_dim", hs // max(1, nh)))
    nl = int(cfg["num_hidden_layers"])
    inter = int(cfg["intermediate_size"])

    dflash_cfg = cfg.get("dflash_config", {}) or {}
    _expect(isinstance(dflash_cfg, dict), "config.dflash_config must be dict")
    k = len(dflash_cfg.get("target_layer_ids") or [])
    _expect(k > 0, "dflash_config.target_layer_ids must be non-empty")
    use_qk_norm = bool(dflash_cfg.get("use_qk_norm", False))
    mlp_bias = bool(dflash_cfg.get("mlp_bias", False))
    fc_bias = bool(dflash_cfg.get("fc_bias", False))

    # Required global weights.
    required = {
        "model.fc.weight": (hs, k * hs),
        "model.hidden_norm.weight": (hs,),
        "model.norm.weight": (hs,),
        "model.mask_embedding": (hs,),
    }
    if fc_bias:
        required["model.fc.bias"] = (hs,)

    # Required per-layer weights.
    for i in range(nl):
        prefix = f"model.layers.{i}"
        required[f"{prefix}.input_layernorm.weight"] = (hs,)
        required[f"{prefix}.post_attention_layernorm.weight"] = (hs,)

        # Attn projections.
        required[f"{prefix}.self_attn.q_proj.weight"] = (nh * hd, hs)
        required[f"{prefix}.self_attn.k_proj.weight"] = (nkv * hd, hs)
        required[f"{prefix}.self_attn.v_proj.weight"] = (nkv * hd, hs)
        required[f"{prefix}.self_attn.o_proj.weight"] = (hs, nh * hd)
        required[f"{prefix}.self_attn.q_proj.bias"] = (nh * hd,)
        required[f"{prefix}.self_attn.k_proj.bias"] = (nkv * hd,)
        required[f"{prefix}.self_attn.v_proj.bias"] = (nkv * hd,)
        required[f"{prefix}.self_attn.o_proj.bias"] = (hs,)
        if use_qk_norm:
            required[f"{prefix}.self_attn.q_norm.weight"] = (hd,)
            required[f"{prefix}.self_attn.k_norm.weight"] = (hd,)

        # MLP projections (bias optional).
        required[f"{prefix}.mlp.gate_proj.weight"] = (inter, hs)
        required[f"{prefix}.mlp.up_proj.weight"] = (inter, hs)
        required[f"{prefix}.mlp.down_proj.weight"] = (hs, inter)
        if mlp_bias:
            required[f"{prefix}.mlp.gate_proj.bias"] = (inter,)
            required[f"{prefix}.mlp.up_proj.bias"] = (inter,)
            required[f"{prefix}.mlp.down_proj.bias"] = (hs,)

    got = {}
    for name, shape, dtype in _iter_weight_slices(model_dir):
        got[name] = (shape, dtype)

    missing = [k for k in required.keys() if k not in got]
    extra = [k for k in got.keys() if k not in required]

    # Shape mismatches for required.
    mismatched = []
    for name, exp_shape in required.items():
        if name not in got:
            continue
        got_shape, _dtype = got[name]
        if tuple(got_shape) != tuple(exp_shape):
            mismatched.append((name, exp_shape, got_shape))

    _expect(not missing, f"Missing required weights: {missing[:20]}{' ...' if len(missing)>20 else ''}")
    _expect(
        not mismatched,
        "Shape mismatches:\n"
        + "\n".join(f"- {n}: expected={e} got={g}" for n, e, g in mismatched[:50]),
    )

    print(
        json.dumps(
            {
                "ok": True,
                "ckpt": str(model_dir),
                "num_required": len(required),
                "num_total": len(got),
                "num_extra": len(extra),
                "extras_sample": sorted(extra)[:30],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

