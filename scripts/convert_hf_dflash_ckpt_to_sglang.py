from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path


def _build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    # Mirror DFlash helper (and our HF draft) behavior.
    if int(num_draft_layers) <= 0:
        raise ValueError("num_draft_layers must be positive")
    if int(num_target_layers) <= 0:
        raise ValueError("num_target_layers must be positive")
    if int(num_draft_layers) == 1:
        return [int(num_target_layers) // 2]
    start = 1
    end = int(num_target_layers) - 3
    span = end - start
    return [
        int(round(start + (i * span) / (int(num_draft_layers) - 1)))
        for i in range(int(num_draft_layers))
    ]


def _safe_round_int(x: float) -> int:
    return int(math.floor(float(x) + 0.5))


def _dtype_nbytes(dtype: str) -> int:
    # safetensors dtype strings
    table = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "F8_E4M3FN": 1,
        "F8_E5M2": 1,
        "I16": 2,
        "U16": 2,
        "F16": 2,
        "BF16": 2,
        "I32": 4,
        "U32": 4,
        "F32": 4,
        "I64": 8,
        "U64": 8,
        "F64": 8,
    }
    if dtype not in table:
        raise ValueError(f"Unsupported safetensors dtype: {dtype}")
    return int(table[dtype])


def _shape_numel(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return int(n)


def _rewrite_dflash_tensor_name_and_value(name: str, tensor):
    """Rewrite our HF draft keys into SGLang DFlashDraftModel-compatible keys.

    Our HF draft uses:
      - mlp.gate_up.{weight,bias}  (fused gate+up)
      - mlp.down.{weight,bias}

    SGLang expects (via load_weights mapping):
      - mlp.gate_proj.{weight,bias}
      - mlp.up_proj.{weight,bias}
      - mlp.down_proj.{weight,bias}
    """
    if ".mlp.gate_up." in name:
        suffix = name.split(".mlp.gate_up.", 1)[1]
        if suffix not in ("weight", "bias"):
            return [(name, tensor)]
        dim0 = int(tensor.shape[0]) if tensor.ndim >= 1 else 0
        if dim0 % 2 != 0:
            raise ValueError(f"Cannot split gate_up with odd dim0: {name} shape={tuple(tensor.shape)}")
        half = dim0 // 2
        first = tensor[:half]
        second = tensor[half:]
        gate_name = name.replace(".mlp.gate_up.", ".mlp.gate_proj.", 1)
        up_name = name.replace(".mlp.gate_up.", ".mlp.up_proj.", 1)
        # Keep contiguous layout where possible (torch or numpy).
        try:
            first = first.contiguous()  # type: ignore[attr-defined]
            second = second.contiguous()  # type: ignore[attr-defined]
        except Exception:
            try:
                import numpy as np

                first = np.ascontiguousarray(first)
                second = np.ascontiguousarray(second)
            except Exception:
                pass
        return [(gate_name, first), (up_name, second)]

    if ".mlp.down." in name:
        suffix = name.split(".mlp.down.", 1)[1]
        if suffix in ("weight", "bias"):
            return [(name.replace(".mlp.down.", ".mlp.down_proj.", 1), tensor)]
        return [(name, tensor)]

    return [(name, tensor)]


def _torch_is_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _rewrite_safetensors_file(src_file: Path, dst_file: Path) -> dict[str, str]:
    """Rewrite a single safetensors file and return output weight_map entries."""
    from safetensors import safe_open

    out_weight_map: dict[str, str] = {}

    # Prefer torch tensors when available so BF16 weights can be rewritten on
    # environments where NumPy lacks a bfloat16 dtype (e.g., Kaggle images).
    if _torch_is_available():
        from safetensors.torch import save_file

        out_tensors: dict[str, "torch.Tensor"] = {}
        with safe_open(str(src_file), framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                for new_name, new_tensor in _rewrite_dflash_tensor_name_and_value(name, tensor):
                    out_tensors[new_name] = new_tensor
                    out_weight_map[new_name] = dst_file.name
        save_file(out_tensors, str(dst_file), metadata={"converted_from": str(src_file.name)})
        return out_weight_map

    from safetensors.numpy import save_file

    out_tensors_np = {}
    with safe_open(str(src_file), framework="np", device="cpu") as f:
        for name in f.keys():
            tensor = f.get_tensor(name)
            for new_name, new_tensor in _rewrite_dflash_tensor_name_and_value(name, tensor):
                out_tensors_np[new_name] = new_tensor
                out_weight_map[new_name] = dst_file.name
    save_file(out_tensors_np, str(dst_file), metadata={"converted_from": str(src_file.name)})
    return out_weight_map


def _plan_shards_for_single_file(src_file: Path, *, max_shard_bytes: int) -> list[list[str]]:
    from safetensors import safe_open

    groups: list[list[str]] = []
    cur: list[str] = []
    cur_bytes = 0
    with safe_open(str(src_file), framework="np", device="cpu") as f:
        for name in f.keys():
            sl = f.get_slice(name)
            shape = tuple(int(x) for x in sl.get_shape())
            dtype = str(sl.get_dtype())
            bytes_ = _shape_numel(shape) * _dtype_nbytes(dtype)
            # gate_up will turn into 2 tensors, but same total size; down rename no size change.
            if cur and (cur_bytes + bytes_ > int(max_shard_bytes)):
                groups.append(cur)
                cur = []
                cur_bytes = 0
            cur.append(name)
            cur_bytes += int(bytes_)
    if cur:
        groups.append(cur)
    return groups


def _rewrite_single_safetensors_to_shards(
    *,
    src_file: Path,
    dst_dir: Path,
    max_shard_bytes: int,
) -> Path:
    """Rewrite an unsharded safetensors file into sharded safetensors + index.json.

    Returns the path to the written index json.
    """
    from safetensors import safe_open

    groups = _plan_shards_for_single_file(src_file, max_shard_bytes=int(max_shard_bytes))
    n_shards = len(groups)
    weight_map: dict[str, str] = {}
    total_size = 0

    if _torch_is_available():
        from safetensors.torch import save_file

        with safe_open(str(src_file), framework="pt", device="cpu") as f:
            for shard_idx, names in enumerate(groups, start=1):
                shard_name = f"model-{shard_idx:05d}-of-{n_shards:05d}.safetensors"
                shard_path = dst_dir / shard_name
                out_tensors: dict[str, "torch.Tensor"] = {}
                for name in names:
                    tensor = f.get_tensor(name)
                    for new_name, new_tensor in _rewrite_dflash_tensor_name_and_value(name, tensor):
                        out_tensors[new_name] = new_tensor
                        weight_map[new_name] = shard_name
                save_file(out_tensors, str(shard_path), metadata={"converted_from": str(src_file.name)})
                total_size += int(shard_path.stat().st_size)
    else:
        from safetensors.numpy import save_file

        with safe_open(str(src_file), framework="np", device="cpu") as f:
            for shard_idx, names in enumerate(groups, start=1):
                shard_name = f"model-{shard_idx:05d}-of-{n_shards:05d}.safetensors"
                shard_path = dst_dir / shard_name
                out_tensors = {}
                for name in names:
                    tensor = f.get_tensor(name)
                    for new_name, new_tensor in _rewrite_dflash_tensor_name_and_value(name, tensor):
                        out_tensors[new_name] = new_tensor
                        weight_map[new_name] = shard_name
                save_file(out_tensors, str(shard_path), metadata={"converted_from": str(src_file.name)})
                total_size += int(shard_path.stat().st_size)

    index = {
        "metadata": {"total_size": str(total_size)},
        "weight_map": weight_map,
    }
    idx_path = dst_dir / "model.safetensors.index.json"
    idx_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return idx_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="HF-style draft checkpoint directory (our GptOssDFlashDraftModel).")
    ap.add_argument("--dst", required=True, help="Output directory for SGLang DFlashDraftModel checkpoint.")
    ap.add_argument(
        "--mask-token",
        default="",
        help="Override dflash_config.mask_token. Default: use tokenizer.pad_token from --src if available.",
    )
    ap.add_argument(
        "--use-qk-norm",
        action="store_true",
        help="Set dflash_config.use_qk_norm=true (for Qwen3-style checkpoints). Default: false (GPT-OSS-style).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite dst if it exists.",
    )
    ap.add_argument(
        "--max-shard-size-gb",
        type=float,
        default=float(os.environ.get("MAX_SHARD_SIZE_GB", "4")),
        help="When converting an unsharded model.safetensors, rewrite into shards of this max size (GB).",
    )
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"src not found: {src}")

    if dst.exists():
        if not args.force:
            raise FileExistsError(f"dst exists: {dst} (use --force to overwrite)")
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    cfg_path = src / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"src config.json not found: {cfg_path}")

    src_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    def get_int(name: str, default: int = 0) -> int:
        v = src_cfg.get(name, default)
        try:
            return int(v)
        except Exception:
            return int(default)

    def get_float(name: str, default: float = 0.0) -> float:
        v = src_cfg.get(name, default)
        try:
            return float(v)
        except Exception:
            return float(default)

    hidden_size = get_int("hidden_size")
    num_hidden_layers = get_int("num_hidden_layers")
    num_target_layers = get_int("num_target_layers")
    num_attention_heads = get_int("num_attention_heads")
    num_key_value_heads = get_int("num_key_value_heads", num_attention_heads)
    head_dim = get_int("head_dim", hidden_size // max(1, num_attention_heads))
    vocab_size = get_int("vocab_size")
    max_position_embeddings = get_int("max_position_embeddings")
    rope_theta = get_float("rope_theta", 150000.0)
    rms_norm_eps = get_float("rms_norm_eps", 1e-5)
    attention_bias = bool(src_cfg.get("attention_bias", True))
    rope_scaling = src_cfg.get("rope_scaling", None)
    block_size = get_int("block_size", 16)
    mlp_ratio = get_float("mlp_ratio", 4.0)
    hidden_act = str(src_cfg.get("hidden_act", "silu"))

    if hidden_size <= 0 or num_hidden_layers <= 0 or num_attention_heads <= 0:
        raise ValueError(
            "src config.json missing required fields: hidden_size/num_hidden_layers/num_attention_heads"
        )

    intermediate_size = get_int("intermediate_size", 0)
    if intermediate_size <= 0:
        intermediate_size = _safe_round_int(hidden_size * mlp_ratio)
    target_layer_ids = _build_target_layer_ids(num_target_layers, num_hidden_layers)

    mask_token = str(args.mask_token).strip()
    if not mask_token:
        # Prefer using the tokenizer metadata saved alongside the checkpoint,
        # but avoid importing Transformers (Kaggle images can have incompatible
        # optional deps like sklearn vs numpy).
        for fname in ("special_tokens_map.json", "tokenizer_config.json"):
            try:
                p = src / fname
                if not p.exists():
                    continue
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("pad_token"):
                    mask_token = str(data["pad_token"])
                    break
            except Exception:
                continue
    if not mask_token:
        # Fallback: a common GPT-OSS pad token string (still must exist in vocab at runtime).
        mask_token = "<|pad|>"

    # Use a known Transformers config type so `AutoConfig` can load it everywhere.
    # SGLang uses `architectures` to select the runtime module (EntryClass).
    out_cfg = {
        "architectures": ["DFlashDraftModel"],
        "model_type": "llama",
        "vocab_size": int(vocab_size),
        "hidden_size": int(hidden_size),
        "intermediate_size": int(intermediate_size),
        "num_hidden_layers": int(num_hidden_layers),
        "num_attention_heads": int(num_attention_heads),
        "num_key_value_heads": int(num_key_value_heads),
        "head_dim": int(head_dim),
        "max_position_embeddings": int(max_position_embeddings),
        "rope_theta": float(rope_theta),
        "rope_scaling": rope_scaling,
        "rms_norm_eps": float(rms_norm_eps),
        "attention_bias": bool(attention_bias),
        # Keep a top-level block_size for ServerArgs inference.
        "block_size": int(block_size),
        # DFLASH-specific metadata consumed by SGLang:
        "dflash_config": {
            "block_size": int(block_size),
            "target_layer_ids": [int(x) for x in target_layer_ids],
            "mask_token": str(mask_token),
            "use_qk_norm": bool(args.use_qk_norm),
        },
        # Preserve for diagnostics / warnings.
        "num_target_layers": int(num_target_layers),
        "mlp_ratio": float(mlp_ratio),
        "hidden_act": str(hidden_act),
        "target_model_id": str(src_cfg.get("target_model_id", "")),
    }

    # Copy everything except config.json, then write our shim config.json.
    for p in src.iterdir():
        if p.name == "config.json":
            continue
        # We rewrite safetensors weights, so don't copy them verbatim.
        if p.suffix == ".safetensors" or p.name.endswith(".safetensors.index.json"):
            continue
        if p.is_dir():
            shutil.copytree(p, dst / p.name)
        else:
            shutil.copy2(p, dst / p.name)

    (dst / "config.json").write_text(json.dumps(out_cfg, indent=2), encoding="utf-8")

    # ---- Rewrite weights into SGLang-compatible naming
    index_path = None
    for candidate in ("model.safetensors.index.json",):
        p = src / candidate
        if p.exists():
            index_path = p
            break

    if index_path is not None:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid weight_map in {index_path}")
        shard_files = sorted({str(v) for v in weight_map.values()})
        out_weight_map: dict[str, str] = {}
        total_size = 0
        for shard_name in shard_files:
            src_shard = src / shard_name
            if not src_shard.exists():
                raise FileNotFoundError(f"Missing shard referenced by index: {src_shard}")
            dst_shard = dst / shard_name
            out_weight_map.update(_rewrite_safetensors_file(src_shard, dst_shard))
            total_size += int(dst_shard.stat().st_size)

        out_index = {
            "metadata": {"total_size": str(total_size)},
            "weight_map": out_weight_map,
        }
        (dst / "model.safetensors.index.json").write_text(
            json.dumps(out_index, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"[+] Wrote rewritten weights + index: {dst}", flush=True)
    else:
        # Unsharded or unindexed. Prefer converting model.safetensors if present.
        safes = sorted(src.glob("*.safetensors"))
        if not safes:
            raise FileNotFoundError(f"No .safetensors files found under {src}")
        if len(safes) == 1 and safes[0].name == "model.safetensors":
            max_shard_bytes = int(float(args.max_shard_size_gb) * 1024**3)
            idx_path = _rewrite_single_safetensors_to_shards(
                src_file=safes[0], dst_dir=dst, max_shard_bytes=max_shard_bytes
            )
            print(f"[+] Rewrote unsharded weights into shards: {idx_path}", flush=True)
        else:
            # Convert each safetensors file as-is and write an index for HF/SGLang loaders.
            out_weight_map: dict[str, str] = {}
            total_size = 0
            for src_file in safes:
                dst_file = dst / src_file.name
                out_weight_map.update(_rewrite_safetensors_file(src_file, dst_file))
                total_size += int(dst_file.stat().st_size)
            out_index = {
                "metadata": {"total_size": str(total_size)},
                "weight_map": out_weight_map,
            }
            (dst / "model.safetensors.index.json").write_text(
                json.dumps(out_index, indent=2, sort_keys=True), encoding="utf-8"
            )
            print(f"[+] Wrote rewritten weights + synthesized index: {dst}", flush=True)

    print(f"[+] Wrote SGLang DFlash config shim: {dst / 'config.json'}", flush=True)


if __name__ == "__main__":
    main()
