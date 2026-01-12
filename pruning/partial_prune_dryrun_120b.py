from __future__ import annotations

import argparse
import json
import os
import resource
import time
from math import ceil
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _maxrss_gib() -> float:
    # Linux: ru_maxrss is in KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0**2)


def _keys_for_layer(layer: int) -> list[str]:
    prefix = f"model.layers.{int(layer)}.mlp."
    return [
        f"{prefix}router.weight",
        f"{prefix}router.bias",
        f"{prefix}experts.gate_up_proj_blocks",
        f"{prefix}experts.gate_up_proj_scales",
        f"{prefix}experts.gate_up_proj_bias",
        f"{prefix}experts.down_proj_blocks",
        f"{prefix}experts.down_proj_scales",
        f"{prefix}experts.down_proj_bias",
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="openai/gpt-oss-120b")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--keep-frac", type=float, default=0.5)
    ap.add_argument("--out-dir", default="artifacts/120b_partial_prune_dryrun")
    ap.add_argument("--index-filename", default="model.safetensors.index.json")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cfg_path = hf_hub_download(args.model_id, filename="config.json")
    cfg = _read_json(cfg_path)
    num_experts = int(cfg.get("num_local_experts") or 0)
    num_layers = int(cfg.get("num_hidden_layers") or 0)
    if num_experts <= 0 or num_layers <= 0:
        raise SystemExit("Could not read num_local_experts/num_hidden_layers from config.json")
    if not (0 <= int(args.layer) < num_layers):
        raise SystemExit(f"--layer must be in [0, {num_layers-1}]")

    index_path = hf_hub_download(args.model_id, filename=args.index_filename)
    index = _read_json(index_path)
    weight_map = index.get("weight_map") or {}
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit("Index JSON missing weight_map")

    keys = _keys_for_layer(int(args.layer))
    missing = [k for k in keys if k not in weight_map]
    if missing:
        raise SystemExit(f"Missing keys in weight_map: {missing}")

    keep_n = max(1, min(num_experts, int(ceil(float(args.keep_frac) * num_experts))))
    keep = list(range(keep_n))

    # Download only the shard files that contain the target tensors.
    shard_files = sorted({weight_map[k] for k in keys})
    shard_paths: dict[str, str] = {}
    download_t0 = time.time()
    for fn in shard_files:
        shard_paths[fn] = hf_hub_download(args.model_id, filename=fn)
    download_s = time.time() - download_t0

    # Load and slice tensors for just this layer.
    from safetensors.torch import safe_open
    from safetensors.torch import save_file

    tensors: dict[str, Any] = {}
    load_t0 = time.time()
    for k in keys:
        fn = weight_map[k]
        with safe_open(shard_paths[fn], framework="pt", device="cpu") as f:
            t = f.get_tensor(k)
        if not hasattr(t, "shape") or not t.shape:
            raise SystemExit(f"Unexpected tensor for key {k}: shape={getattr(t,'shape',None)}")
        if int(t.shape[0]) != num_experts:
            raise SystemExit(
                f"Unexpected leading dim for {k}: shape={tuple(t.shape)} expected dim0={num_experts}"
            )
        tensors[k] = t[keep].contiguous()
    load_s = time.time() - load_t0

    out_file = out_dir / f"layer{int(args.layer)}_keep{keep_n}.safetensors"
    write_t0 = time.time()
    save_file(tensors, str(out_file), metadata={"format": "partial_prune_dryrun"})
    write_s = time.time() - write_t0

    total_s = time.time() - t0
    out_size_gib = out_file.stat().st_size / (1024.0**3)
    shard_total_gib = sum(Path(p).stat().st_size for p in shard_paths.values()) / (1024.0**3)

    report = {
        "model_id": args.model_id,
        "layer": int(args.layer),
        "keep_frac": float(args.keep_frac),
        "keep_n": int(keep_n),
        "num_experts": int(num_experts),
        "num_layers": int(num_layers),
        "unique_shards_downloaded": len(shard_files),
        "download_gib": float(shard_total_gib),
        "output_gib": float(out_size_gib),
        "timings_s": {
            "download": float(download_s),
            "load_slice": float(load_s),
            "write": float(write_s),
            "total": float(total_s),
        },
        "peak_rss_gib": float(_maxrss_gib()),
        "shards": shard_files,
        "output_file": str(out_file),
    }

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path = Path("reports/120b_partial_prune_dryrun.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        "\n".join(
            [
                "# 120B partial prune dry-run",
                "",
                f"- Model: `{args.model_id}`",
                f"- Layer: {int(args.layer)}",
                f"- Keep: {keep_n}/{num_experts} ({float(args.keep_frac):.2f})",
                f"- Downloaded shards: {len(shard_files)} ({shard_total_gib:.2f} GiB)",
                f"- Output file: `{out_file}` ({out_size_gib:.2f} GiB)",
                "",
                "## Timings",
                "",
                f"- download: {download_s:.1f}s",
                f"- load+slice: {load_s:.1f}s",
                f"- write: {write_s:.1f}s",
                f"- total: {total_s:.1f}s",
                "",
                "## Peak memory",
                "",
                f"- peak RSS: {report['peak_rss_gib']:.2f} GiB",
                "",
                "## Reproduce",
                "",
                "```bash",
                "python3 -m pruning.partial_prune_dryrun_120b --layer 0 --keep-frac 0.5",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[+] Wrote {md_path}")
    print(f"[+] Wrote {out_dir/'report.json'}")
    print(f"[+] Wrote {out_file}")


if __name__ == "__main__":
    main()

