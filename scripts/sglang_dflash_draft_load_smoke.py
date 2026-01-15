from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace


def _iter_weights_from_index(model_dir: Path):
    from safetensors.torch import safe_open

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")
    idx = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = idx.get("weight_map", {})
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid weight_map in {index_path}")

    for shard in sorted(set(weight_map.values())):
        shard_path = model_dir / shard
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for k in f.keys():
                yield k, f.get_tensor(k)

def _load_minimal_hf_config(model_dir: Path) -> SimpleNamespace:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    cfg_dict = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg_dict, dict):
        raise ValueError(f"Invalid config.json type: {type(cfg_dict).__name__}")
    return SimpleNamespace(**cfg_dict)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="SGLang DFlashDraftModel checkpoint dir (converted).")
    ap.add_argument(
        "--backend",
        default="gloo",
        help="torch.distributed backend for single-process init (gloo is safest).",
    )
    args = ap.parse_args()

    model_dir = Path(args.ckpt).expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    # Kaggle often has TensorFlow installed with binary deps that can be
    # incompatible with the runtime numpy. Prevent Transformers from importing TF.
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")

    import torch

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.models.dflash import DFlashDraftModel

    if not torch.distributed.is_initialized():
        init_distributed_environment(world_size=1, rank=0, backend=str(args.backend))
        initialize_model_parallel(1, 1, 1)

    cfg = _load_minimal_hf_config(model_dir)
    model = DFlashDraftModel(cfg)
    model.load_weights(_iter_weights_from_index(model_dir))

    names = [n for n, _ in model.named_parameters()]
    has_mask = "mask_embedding" in names or any(n.endswith(".mask_embedding") for n in names)
    print(
        json.dumps(
            {
                "ok": True,
                "ckpt": str(model_dir),
                "num_params": sum(p.numel() for p in model.parameters()),
                "has_mask_embedding": bool(has_mask),
                "block_size": int(getattr(model, "block_size", 0) or 0),
                "num_context_features": int(getattr(model, "num_context_features", 0) or 0),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
