from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class _ZarrSpec:
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: np.dtype
    dimension_separator: str
    compressor_id: str
    compressor_level: int | None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _parse_zarr_spec(zarr_dir: Path) -> _ZarrSpec:
    meta_path = zarr_dir / ".zarray"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing .zarray: {meta_path}")
    meta = _read_json(meta_path)
    if not isinstance(meta, dict):
        raise ValueError(f"Invalid .zarray JSON: {meta_path}")

    dtype = np.dtype(str(meta.get("dtype")))
    shape = tuple(int(x) for x in meta.get("shape", []))
    chunks = tuple(int(x) for x in meta.get("chunks", []))
    if not shape or not chunks or len(shape) != len(chunks):
        raise ValueError(f"Invalid zarr shape/chunks in {meta_path}: shape={shape} chunks={chunks}")

    dimension_separator = str(meta.get("dimension_separator", "."))
    compressor = meta.get("compressor", None)
    compressor_id = ""
    compressor_level: int | None = None
    if isinstance(compressor, dict):
        compressor_id = str(compressor.get("id", ""))
        if "level" in compressor:
            try:
                compressor_level = int(compressor.get("level"))
            except Exception:
                compressor_level = None
    if compressor_id != "zstd":
        raise ValueError(
            "Only zstd-compressed zarr v2 arrays are supported right now. "
            f"Got compressor_id={compressor_id!r} in {meta_path}."
        )
    if meta.get("filters", None) is not None:
        raise ValueError(f"Unsupported zarr filters in {meta_path}: filters={meta.get('filters')!r}")

    return _ZarrSpec(
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        dimension_separator=dimension_separator,
        compressor_id=compressor_id,
        compressor_level=compressor_level,
    )


def _zstd_decompress(path: Path) -> bytes:
    # Use the system zstd binary to avoid Python deps (zarr/tensorstore/zstandard).
    proc = subprocess.run(
        ["zstd", "-q", "-d", "-c", str(path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout


def _read_zarr_array(zarr_dir: Path) -> np.ndarray:
    spec = _parse_zarr_spec(zarr_dir)
    out = np.empty(spec.shape, dtype=spec.dtype, order="C")

    grid = tuple(_ceil_div(s, c) for s, c in zip(spec.shape, spec.chunks))
    if any(g <= 0 for g in grid):
        raise ValueError(f"Invalid zarr grid for {zarr_dir}: shape={spec.shape} chunks={spec.chunks}")

    def iter_coords() -> Iterable[tuple[int, ...]]:
        if len(grid) == 1:
            for i0 in range(grid[0]):
                yield (i0,)
            return
        # Generic cartesian product.
        cur = [0] * len(grid)
        while True:
            yield tuple(cur)
            for d in reversed(range(len(grid))):
                cur[d] += 1
                if cur[d] < grid[d]:
                    break
                cur[d] = 0
                if d == 0:
                    return

    for coord in iter_coords():
        fname = spec.dimension_separator.join(str(x) for x in coord)
        chunk_path = zarr_dir / fname
        if not chunk_path.exists():
            raise FileNotFoundError(f"Missing zarr chunk: {chunk_path}")
        raw = _zstd_decompress(chunk_path)

        starts = [int(ci) * int(cs) for ci, cs in zip(coord, spec.chunks)]
        stops = [min(int(st) + int(cs), int(sh)) for st, cs, sh in zip(starts, spec.chunks, spec.shape)]
        chunk_shape = tuple(int(b - a) for a, b in zip(starts, stops))
        expected_nbytes = int(np.prod(chunk_shape)) * int(spec.dtype.itemsize)
        if len(raw) != expected_nbytes:
            raise ValueError(
                "Decompressed chunk size mismatch for "
                f"{chunk_path}: got={len(raw)} expected={expected_nbytes} "
                f"(dtype={spec.dtype} chunk_shape={chunk_shape})"
            )

        arr = np.frombuffer(raw, dtype=spec.dtype).reshape(chunk_shape, order="C")
        slc = tuple(slice(a, b) for a, b in zip(starts, stops))
        out[slc] = arr

    return out


def _target_cfg_from_training_args(run_dir: Path) -> dict[str, Any]:
    args_path = run_dir / "easydel-training-arguments.json"
    if not args_path.exists():
        return {}
    train_args = _read_json(args_path)
    if not isinstance(train_args, dict):
        return {}
    # Our EasyDeL trainer uses `teacher_snapshot_dir`. Older experiments used
    # `model_snapshot_dir`.
    snap = train_args.get("teacher_snapshot_dir", "") or train_args.get("model_snapshot_dir", "")
    if not snap:
        # Our trainer may store target snapshot info in cache_dir/meta.json.
        cache_dir = train_args.get("cache_dir", "")
        if cache_dir:
            meta_path = Path(str(cache_dir)) / "meta.json"
            if meta_path.exists():
                meta = _read_json(meta_path)
                if isinstance(meta, dict) and meta.get("model_snapshot_dir"):
                    snap = meta.get("model_snapshot_dir", "")
    if not snap:
        return {}
    cfg_path = Path(str(snap)) / "config.json"
    if not cfg_path.exists():
        return {}
    cfg = _read_json(cfg_path)
    return cfg if isinstance(cfg, dict) else {}


def _get_target_layer_ids(run_dir: Path) -> list[int]:
    args_path = run_dir / "easydel-training-arguments.json"
    if not args_path.exists():
        raise FileNotFoundError(f"Missing {args_path} (needed for target_layer_ids).")
    train_args = _read_json(args_path)
    if not isinstance(train_args, dict):
        raise ValueError(f"Invalid JSON in {args_path}")
    tli = train_args.get("target_layer_ids", None)
    if not isinstance(tli, list) or not tli:
        cache_dir = train_args.get("cache_dir", "")
        if cache_dir:
            meta_path = Path(str(cache_dir)) / "meta.json"
            if meta_path.exists():
                meta = _read_json(meta_path)
                if isinstance(meta, dict):
                    tli = meta.get("target_layer_ids", None)
    if not isinstance(tli, list) or not tli:
        # Cache dirs are often ephemeral on TPU boxes (/dev/shm). Our EasyDeL
        # run directory always includes `config.json` with the resolved
        # `target_layer_ids`, so fall back to that to make export reproducible.
        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            cfg = _read_json(cfg_path)
            if isinstance(cfg, dict):
                tli = cfg.get("target_layer_ids", None)
    if not isinstance(tli, list) or not tli:
        raise ValueError(
            f"Missing/invalid target_layer_ids in {args_path} (and cache_dir/meta.json and run/config.json)."
        )
    return [int(x) for x in tli]


def _plan_shards(num_layers: int) -> list[str]:
    # 1 global shard + 1 shard per layer.
    total = int(num_layers) + 1
    return [f"model-{i:05d}-of-{total:05d}.safetensors" for i in range(1, total + 1)]


def _assign_shard(name: str, *, num_layers: int) -> int:
    # shard index in [0..num_layers] where 0 is global, i+1 is layer i.
    if ".layers." not in name:
        return 0
    try:
        after = name.split(".layers.", 1)[1]
        layer_str = after.split(".", 1)[0]
        layer_id = int(layer_str)
    except Exception:
        return 0
    if 0 <= layer_id < int(num_layers):
        return 1 + layer_id
    return 0


def _map_tensor_name(path: str) -> tuple[str, bool, bool] | None:
    # Returns (out_name, transpose_kernel, is_bias) or None to skip.
    # input path examples:
    #   model/layers/0/attn/q_proj/kernel/value
    #   model/layers/0/in_norm/scale/value
    #   model/fc/kernel/value
    #   model/mask_embedding/value
    p = path.strip("/")
    if not p.startswith("model/"):
        return None
    parts = p.split("/")
    # model/<...>
    if parts == ["model", "mask_embedding", "value"]:
        return ("model.mask_embedding", False, False)
    if parts == ["model", "final_norm", "scale", "value"]:
        return ("model.norm.weight", False, False)
    if parts == ["model", "hidden_norm", "scale", "value"]:
        return ("model.hidden_norm.weight", False, False)
    if parts == ["model", "fc", "kernel", "value"]:
        return ("model.fc.weight", True, False)
    if parts == ["model", "fc", "bias", "value"]:
        return ("model.fc.bias", False, True)

    if len(parts) >= 4 and parts[1] == "layers":
        layer_id = parts[2]
        if parts[3] == "in_norm" and parts[4:] == ["scale", "value"]:
            return (f"model.layers.{layer_id}.input_layernorm.weight", False, False)
        if parts[3] == "post_norm" and parts[4:] == ["scale", "value"]:
            return (f"model.layers.{layer_id}.post_attention_layernorm.weight", False, False)
        if parts[3] == "attn":
            proj = parts[4]
            kind = parts[5:]
            if proj not in ("q_proj", "k_proj", "v_proj", "o_proj"):
                return None
            if kind == ["kernel", "value"]:
                return (f"model.layers.{layer_id}.self_attn.{proj}.weight", True, False)
            if kind == ["bias", "value"]:
                return (f"model.layers.{layer_id}.self_attn.{proj}.bias", False, True)
            return None
        if parts[3] == "mlp":
            proj = parts[4]
            kind = parts[5:]
            if proj not in ("gate_proj", "up_proj", "down_proj"):
                return None
            if kind == ["kernel", "value"]:
                return (f"model.layers.{layer_id}.mlp.{proj}.weight", True, False)
            if kind == ["bias", "value"]:
                return (f"model.layers.{layer_id}.mlp.{proj}.bias", False, True)
            return None

    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="EasyDeL run directory (contains model/ + tensorstore_index.json).")
    ap.add_argument("--dst", required=True, help="Output directory for SGLang DFlashDraftModel checkpoint.")
    ap.add_argument(
        "--mask-token",
        default="",
        help="dflash_config.mask_token (defaults to <|MASK|>).",
    )
    ap.add_argument(
        "--dtype",
        default=os.environ.get("DFLASH_CONVERT_DTYPE", "float16"),
        choices=["float16", "float32"],
        help="Output weight dtype. float16 recommended for GPU inference.",
    )
    ap.add_argument("--keep-fc-bias", action="store_true", help="Keep and export fc.bias (requires patched SGLang).")
    ap.add_argument(
        "--target-layer-ids-mode",
        default=os.environ.get("DFLASH_TARGET_LAYER_IDS_MODE", "prelayer"),
        choices=["prelayer", "afterlayer"],
        help=(
            "How to interpret `target_layer_ids` stored in the EasyDeL run metadata. "
            "`prelayer` means they index EasyDeL's output_hidden_states entries directly "
            "(our current TPU pipeline). "
            "`afterlayer` means they follow SGLang PR #16818 semantics and must be shifted by -1 "
            "when writing the HF config so that SGLang's +1 pre-layer capture lands on the same activations."
        ),
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite dst if it exists.",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    if dst.exists():
        if not args.force:
            raise FileExistsError(f"dst exists: {dst} (use --force)")
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    draft_cfg_path = run_dir / "config.json"
    if not draft_cfg_path.exists():
        raise FileNotFoundError(f"Missing draft config: {draft_cfg_path}")
    draft_cfg = _read_json(draft_cfg_path)
    if not isinstance(draft_cfg, dict):
        raise ValueError(f"Invalid draft config JSON: {draft_cfg_path}")

    target_cfg = _target_cfg_from_training_args(run_dir)
    target_layer_ids_raw = _get_target_layer_ids(run_dir)
    if str(args.target_layer_ids_mode).lower() == "afterlayer":
        # If the run stored pre-layer capture indices (EasyDeL hidden_states indices),
        # convert them back to HF-style "after layer" ids expected by SGLang PR #16818.
        target_layer_ids = [int(x) - 1 for x in target_layer_ids_raw]
    else:
        target_layer_ids = [int(x) for x in target_layer_ids_raw]

    hidden_size = int(draft_cfg["hidden_size"])
    num_layers = int(draft_cfg["num_layers"])
    num_attention_heads = int(draft_cfg["num_attention_heads"])
    num_key_value_heads = int(draft_cfg["num_key_value_heads"])
    head_dim = int(draft_cfg.get("head_dim", hidden_size // max(1, num_attention_heads)))
    mlp_ratio = float(draft_cfg.get("mlp_ratio", 4.0))
    intermediate_size = int(round(hidden_size * mlp_ratio))
    block_size = int(draft_cfg.get("block_size", 16))
    rms_norm_eps = float(draft_cfg.get("rms_norm_eps", 1e-5))
    hidden_act = str(draft_cfg.get("hidden_act", "silu"))
    use_qk_norm = bool(draft_cfg.get("qk_norm", False))
    num_context_features = int(draft_cfg.get("num_context_features", len(target_layer_ids)))
    if int(num_context_features) != len(target_layer_ids_raw):
        raise ValueError(
            f"Draft num_context_features={num_context_features} != len(target_layer_ids)={len(target_layer_ids_raw)}; "
            "this checkpoint won't match SGLang hidden capture."
        )

    vocab_size = int(target_cfg.get("vocab_size", 0) or target_cfg.get("padded_vocab_size", 0) or 0)
    if vocab_size <= 0:
        # Keep config loadable; the draft itself never uses embeddings/lm_head.
        vocab_size = 32000
    max_position_embeddings = int(target_cfg.get("max_position_embeddings", 131072))
    rope_theta = float(target_cfg.get("rope_theta", 150000.0))
    rope_scaling = target_cfg.get("rope_scaling", None)
    attention_bias = bool(target_cfg.get("attention_bias", True))

    mask_token = str(args.mask_token).strip() or "<|MASK|>"
    mask_token_id = None
    if isinstance(target_cfg, dict):
        # Prefer using the target model's tokenizer-resolved pad id for `<|pad|>`.
        if mask_token == "<|pad|>" and target_cfg.get("pad_token_id", None) is not None:
            try:
                mask_token_id = int(target_cfg["pad_token_id"])
            except Exception:
                mask_token_id = None
    if mask_token_id is None and int(vocab_size) > 200000:
        # Match upstream DFLASH defaults (mask token id 200000) when we can't
        # resolve an explicit id from the target config.
        mask_token_id = 200000

    out_cfg = {
        "architectures": ["DFlashDraftModel"],
        "model_type": "llama",
        "vocab_size": int(vocab_size),
        "hidden_size": int(hidden_size),
        "intermediate_size": int(intermediate_size),
        "num_hidden_layers": int(num_layers),
        "num_attention_heads": int(num_attention_heads),
        "num_key_value_heads": int(num_key_value_heads),
        "head_dim": int(head_dim),
        "max_position_embeddings": int(max_position_embeddings),
        "rope_theta": float(rope_theta),
        "rope_scaling": rope_scaling,
        "rms_norm_eps": float(rms_norm_eps),
        "attention_bias": bool(attention_bias),
        "block_size": int(block_size),
        "hidden_act": str(hidden_act),
        "dflash_config": {
            "block_size": int(block_size),
            "target_layer_ids": [int(x) for x in target_layer_ids],
            "mask_token": str(mask_token),
            # SGLang-JAX resolves the mask token id without access to the tokenizer
            # inside the model worker. Provide an explicit id when we can.
            "mask_token_id": mask_token_id,
            "use_qk_norm": bool(use_qk_norm),
            "use_mask_embedding": True,
            # Our EasyDeL draft uses biases in MLP and (often) in fc.
            "mlp_bias": True,
            "fc_bias": bool(args.keep_fc_bias),
        },
        "source_checkpoint": str(run_dir),
    }
    (dst / "config.json").write_text(json.dumps(out_cfg, indent=2), encoding="utf-8")

    # ---- Read model tensors from zarr
    ts_index = run_dir / "tensorstore_index.json"
    if not ts_index.exists():
        raise FileNotFoundError(f"Missing {ts_index}")
    idx = _read_json(ts_index)
    if not isinstance(idx, dict) or "prefixes" not in idx or not isinstance(idx["prefixes"], dict):
        raise ValueError(f"Invalid {ts_index}")
    model_entries = idx["prefixes"].get("model", None)
    if not isinstance(model_entries, list) or not model_entries:
        raise ValueError(f"Missing prefixes.model entries in {ts_index}")

    from safetensors.numpy import save_file

    out_dtype = np.float16 if str(args.dtype) == "float16" else np.float32

    shard_names = _plan_shards(num_layers=num_layers)
    shard_tensors: list[dict[str, np.ndarray]] = [dict() for _ in shard_names]
    weight_map: dict[str, str] = {}

    def put_tensor(name: str, arr: np.ndarray) -> None:
        shard_idx = _assign_shard(name, num_layers=num_layers)
        shard_tensors[shard_idx][name] = arr
        weight_map[name] = shard_names[shard_idx]

    # Add q_norm/k_norm weights for parameter-free qk_norm (set to ones).
    if use_qk_norm:
        ones = np.ones((head_dim,), dtype=out_dtype)
        for layer_id in range(num_layers):
            put_tensor(f"model.layers.{layer_id}.self_attn.q_norm.weight", ones.copy())
            put_tensor(f"model.layers.{layer_id}.self_attn.k_norm.weight", ones.copy())

    # Load arrays from zarr and map them to SGLang param names.
    for entry in model_entries:
        if not isinstance(entry, dict) or "path" not in entry:
            continue
        path = str(entry["path"])
        mapped = _map_tensor_name(path)
        if mapped is None:
            continue
        out_name, transpose_kernel, _is_bias = mapped
        if out_name.endswith("fc.bias") and not bool(args.keep_fc_bias):
            continue

        zarr_dir = run_dir / path
        arr = _read_zarr_array(zarr_dir)

        if transpose_kernel and arr.ndim == 2:
            arr = arr.T

        if arr.dtype != out_dtype:
            arr = arr.astype(out_dtype, copy=False)

        put_tensor(out_name, arr)

    # ---- Fuse EasyDeL MLP gate/up into SGLang DFLASH `gate_up_proj`.
    #
    # EasyDeL draft MLP uses (gate_proj, up_proj, down_proj) like LLaMA, while
    # the upstream SGLang DFLASH draft model uses a single `gate_up_proj`
    # projection (concatenated on the output dimension). Produce the expected
    # HF tensor names so SGLang can load the trained weights without leaving
    # random-initialized parameters.
    def drop_tensor(name: str) -> None:
        shard = weight_map.pop(name, None)
        if shard is None:
            return
        try:
            shard_idx = shard_names.index(shard)
        except ValueError:
            return
        shard_tensors[shard_idx].pop(name, None)

    for layer_id in range(num_layers):
        gate_w = f"model.layers.{layer_id}.mlp.gate_proj.weight"
        up_w = f"model.layers.{layer_id}.mlp.up_proj.weight"
        gate_b = f"model.layers.{layer_id}.mlp.gate_proj.bias"
        up_b = f"model.layers.{layer_id}.mlp.up_proj.bias"
        if gate_w not in weight_map or up_w not in weight_map:
            continue

        # Both tensors should live in the same layer shard.
        shard_name = weight_map[gate_w]
        shard_idx = shard_names.index(shard_name)
        gw = shard_tensors[shard_idx][gate_w]
        uw = shard_tensors[shard_idx][up_w]
        fused_w = np.concatenate([gw, uw], axis=0)
        put_tensor(f"model.layers.{layer_id}.mlp.gate_up_proj.weight", fused_w)
        drop_tensor(gate_w)
        drop_tensor(up_w)

        if gate_b in weight_map and up_b in weight_map:
            gb = shard_tensors[shard_idx][gate_b]
            ub = shard_tensors[shard_idx][up_b]
            fused_b = np.concatenate([gb, ub], axis=0)
            put_tensor(f"model.layers.{layer_id}.mlp.gate_up_proj.bias", fused_b)
            drop_tensor(gate_b)
            drop_tensor(up_b)

    # Write shards + index
    total_size = 0
    for shard_name, tensors in zip(shard_names, shard_tensors):
        if not tensors:
            continue
        out_path = dst / shard_name
        save_file(tensors, str(out_path), metadata={"converted_from": str(run_dir)})
        total_size += int(out_path.stat().st_size)

    index_out = {"metadata": {"total_size": str(total_size)}, "weight_map": weight_map}
    (dst / "model.safetensors.index.json").write_text(
        json.dumps(index_out, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"[+] Wrote SGLang DFlashDraftModel checkpoint: {dst}", flush=True)
    print(f"[+] Shards: {len([p for p in dst.glob('model-*.safetensors')])}", flush=True)
    print(f"[+] Index: {dst / 'model.safetensors.index.json'}", flush=True)


if __name__ == "__main__":
    main()
