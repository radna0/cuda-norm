"""
CPU-only 120B partial prune sweep (Modal).

Purpose
-------
Slice MoE expert tensors for *multiple* GPT-OSS-120B layers in a single remote call,
grouped by shard file locality, so we don't repeatedly download/open the same shards.

Outputs
-------
- Writes per-layer safetensors into the persistent Modal volume mounted at `/root/data`
  under: `artifacts/120b_partial_prune_sweep/<run_id>/layer{L}_keep{K}.safetensors`
- Writes a timestamped local report under `reports/` (entrypoint)

Run (always log to unsloth_logs/):
  mkdir -p unsloth_logs
  ts=$(date +%Y%m%d_%H%M%S)
  nohup env MODAL_PROFILE=locthaokien1201 modal run modal/partial_prune_sweep_120b_modal.py \\
    --layers 0-35 --keep-frac 0.5 \\
    > "unsloth_logs/120b_partial_prune_sweep_${ts}.log" 2>&1 &
"""

from __future__ import annotations

import json
import os
import resource
import time
from math import ceil
from pathlib import Path
from typing import Any, Iterable

import modal

APP_NAME = "partial-prune-sweep-120b"


def _maybe_load_repo_dotenv() -> None:
    try:
        dotenv_path = Path(__file__).resolve().parent.parent / ".env"
        if not dotenv_path.exists():
            return
        for raw in dotenv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                val = val[1:-1]
            os.environ.setdefault(key, val)
    except Exception:
        return


_maybe_load_repo_dotenv()

DEFAULT_MODEL_ID_120B = os.environ.get("MODEL_ID_120B", "openai/gpt-oss-120b")

_secrets: list[modal.Secret] = []
if os.environ.get("HF_TOKEN"):
    _secrets.append(modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}))

data_volume = modal.Volume.from_name("pruning-data", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("hf-cache-persistent", create_if_missing=True)

BASE_IMAGE = "nvidia/cuda:12.8.0-devel-ubuntu24.04"
image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.12")
    .apt_install("git", "python3-dev", "build-essential", "curl")
    .run_commands("python -m pip install -U pip setuptools wheel")
    .run_commands(
        "python -m pip install "
        "torch==2.9.0 "
        "--extra-index-url https://download.pytorch.org/whl/cu128"
    )
    .run_commands(
        "python -m pip install "
        "huggingface-hub==0.34.0 hf_transfer "
        "safetensors numpy==2.2.0"
    )
)

app = modal.App(APP_NAME)


def _ensure_hf_env() -> None:
    os.environ.setdefault("HF_HOME", "/root/hf_cache")
    os.environ.setdefault("XDG_CACHE_HOME", "/root/hf_cache/.cache")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for p in ("/root/hf_cache", "/root/hf_cache/.cache", "/root/data"):
        try:
            Path(p).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


def _get_hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    return tok.strip() if tok else None


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _maxrss_gib() -> float:
    # Linux: ru_maxrss is KiB.
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


def _parse_layers(spec: str) -> list[int]:
    out: set[int] = set()
    s = (spec or "").strip()
    if not s:
        return []
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            lo = int(a.strip())
            hi = int(b.strip())
            if hi < lo:
                lo, hi = hi, lo
            for x in range(lo, hi + 1):
                out.add(x)
        else:
            out.add(int(p))
    return sorted(out)


@app.function(
    image=image,
    timeout=21600,
    cpu=16.0,
    memory=262144,
    volumes={
        "/root/data": data_volume,
        "/root/hf_cache": hf_cache_volume,
    },
    secrets=_secrets,
)
def partial_prune_sweep(
    *,
    model_id: str,
    layers: list[int],
    keep_frac: float,
    out_subdir: str,
) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import safe_open
    from safetensors.torch import save_file

    _ensure_hf_env()
    try:
        data_volume.reload()
        hf_cache_volume.reload()
    except Exception:
        pass

    t0 = time.time()
    token = _get_hf_token()

    cfg_path = hf_hub_download(str(model_id), filename="config.json", token=token)
    cfg = _read_json(cfg_path)
    num_experts = int(cfg.get("num_local_experts") or 0)
    num_layers = int(cfg.get("num_hidden_layers") or 0)
    if num_experts <= 0 or num_layers <= 0:
        raise RuntimeError("Could not read num_local_experts/num_hidden_layers from config.json")

    if not layers:
        raise ValueError("layers must be non-empty")
    for li in layers:
        if not (0 <= int(li) < num_layers):
            raise ValueError(f"layer {li} must be in [0, {num_layers-1}]")

    idx_path = hf_hub_download(str(model_id), filename="model.safetensors.index.json", token=token)
    idx = _read_json(idx_path)
    weight_map = idx.get("weight_map") or {}
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("Index JSON missing weight_map")

    keep_n = max(1, min(num_experts, int(ceil(float(keep_frac) * num_experts))))
    keep = list(range(keep_n))

    # Plan: for each layer, list required keys and which shard file they live in.
    layer_plan: dict[int, dict[str, Any]] = {}
    for li in layers:
        keys = _keys_for_layer(int(li))
        missing = [k for k in keys if k not in weight_map]
        if missing:
            raise RuntimeError(f"Missing keys in weight_map for layer={li}: {missing}")
        shard_files = sorted({str(weight_map[k]) for k in keys})
        keys_by_shard: dict[str, list[str]] = {}
        for k in keys:
            fn = str(weight_map[k])
            keys_by_shard.setdefault(fn, []).append(k)
        layer_plan[int(li)] = {
            "keys": keys,
            "shard_files": shard_files,
            "keys_by_shard": keys_by_shard,
        }

    # Group layers by shard locality (same shard file tuple).
    groups: dict[tuple[str, ...], list[int]] = {}
    for li, info in layer_plan.items():
        groups.setdefault(tuple(info["shard_files"]), []).append(li)
    for k in list(groups.keys()):
        groups[k] = sorted(groups[k])

    # Download each shard at most once, even if multiple groups share it.
    shard_paths: dict[str, str] = {}
    shard_sizes: dict[str, float] = {}
    dl_t0 = time.time()
    for shard_files in sorted(groups.keys()):
        for fn in shard_files:
            if fn in shard_paths:
                continue
            path = hf_hub_download(str(model_id), filename=str(fn), token=token)
            shard_paths[fn] = path
            shard_sizes[fn] = Path(path).stat().st_size / (1024.0**3)
    dl_s = time.time() - dl_t0
    dl_gib = float(sum(shard_sizes.values()))

    out_dir = Path("/root/data") / str(out_subdir).strip().strip("/")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_layer: list[dict[str, Any]] = []
    # Process groups so we maximize reuse of downloaded shards.
    for shard_files, group_layers in groups.items():
        for li in group_layers:
            info = layer_plan[li]
            load_t0 = time.time()
            tensors: dict[str, Any] = {}
            for fn in shard_files:
                keys = info["keys_by_shard"].get(fn) or []
                if not keys:
                    continue
                with safe_open(shard_paths[fn], framework="pt", device="cpu") as f:
                    for k in keys:
                        t = f.get_tensor(k)
                        if int(getattr(t, "shape", [0])[0]) != num_experts:
                            raise RuntimeError(
                                f"Unexpected leading dim for {k}: shape={tuple(getattr(t,'shape',()))} "
                                f"expected dim0={num_experts}"
                            )
                        tensors[k] = t[keep].contiguous()
            load_s = time.time() - load_t0

            out_file = out_dir / f"layer{int(li)}_keep{keep_n}.safetensors"
            write_t0 = time.time()
            save_file(
                tensors,
                str(out_file),
                metadata={"format": "pt", "harmony": "partial_prune_sweep"},
            )
            write_s = time.time() - write_t0

            per_layer.append(
                {
                    "layer": int(li),
                    "shards": list(shard_files),
                    "output_file": str(out_file),
                    "output_gib": float(out_file.stat().st_size / (1024.0**3)),
                    "timings_s": {
                        "load_slice": float(load_s),
                        "write": float(write_s),
                        "layer_total": float(load_s + write_s),
                    },
                }
            )

    data_volume.commit()
    hf_cache_volume.commit()

    total_s = time.time() - t0
    unique_shards = sorted(shard_paths.keys())
    return {
        "model_id": str(model_id),
        "layers": [int(x) for x in layers],
        "num_layers": int(num_layers),
        "num_experts": int(num_experts),
        "keep_frac": float(keep_frac),
        "keep_n": int(keep_n),
        "groups": [{"shards": list(k), "layers": v} for k, v in sorted(groups.items(), key=lambda kv: kv[0])],
        "unique_shards_downloaded": int(len(unique_shards)),
        "download_gib": float(dl_gib),
        "download_s": float(dl_s),
        "per_layer": sorted(per_layer, key=lambda r: int(r["layer"])),
        "peak_rss_gib": float(_maxrss_gib()),
        "timings_s": {"total": float(total_s)},
        "out_dir": str(out_dir),
    }


@app.local_entrypoint()
def main(
    model_id: str = DEFAULT_MODEL_ID_120B,
    layers: str = "0-35",
    keep_frac: float = 0.5,
):
    run_id = time.strftime("%Y%m%d_%H%M%S")
    layer_list = _parse_layers(layers)
    res = partial_prune_sweep.remote(
        model_id=str(model_id),
        layers=layer_list,
        keep_frac=float(keep_frac),
        out_subdir=f"artifacts/120b_partial_prune_sweep/{run_id}",
    )

    md_path = Path(f"reports/120b_partial_prune_sweep_{run_id}.md")
    md_latest = Path("reports/120b_partial_prune_sweep_latest.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# 120B partial prune sweep")
    lines.append("")
    lines.append(f"- run_id: `{run_id}`")
    lines.append(f"- Model: `{res['model_id']}`")
    lines.append(f"- layers: {len(res['layers'])} ({min(res['layers'])}..{max(res['layers'])})")
    lines.append(f"- Keep: {res['keep_n']}/{res['num_experts']} ({res['keep_frac']:.2f})")
    lines.append(f"- Unique shards downloaded: {res['unique_shards_downloaded']} ({res['download_gib']:.2f} GiB)")
    lines.append(f"- Download time: {res['download_s']:.1f}s")
    lines.append(f"- Peak RSS: {res['peak_rss_gib']:.2f} GiB")
    lines.append(f"- Output dir (Modal volume): `{res['out_dir']}`")
    lines.append("")
    lines.append("## Layer outputs")
    lines.append("")
    lines.append("| layer | shards | out_gib | load_slice_s | write_s |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in res["per_layer"]:
        lines.append(
            f"| {row['layer']} | {len(row['shards'])} | {row['output_gib']:.2f} | "
            f"{row['timings_s']['load_slice']:.2f} | {row['timings_s']['write']:.2f} |"
        )
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append(
        f"modal run modal/partial_prune_sweep_120b_modal.py --layers {layers!r} --keep-frac {float(keep_frac)}"
    )
    lines.append("```")
    lines.append("")

    md_text = "\n".join(lines) + "\n"
    md_path.write_text(md_text, encoding="utf-8")
    md_latest.write_text(md_text, encoding="utf-8")

    json_path = Path(f"reports/120b_partial_prune_sweep_{run_id}.json")
    json_path.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[+] Wrote {md_path}")
    print(f"[+] Wrote {md_latest}")
    print(f"[+] Wrote {json_path}")
    print("[RESULT]", {k: res[k] for k in ('model_id','keep_frac','keep_n','unique_shards_downloaded','download_gib','download_s','timings_s','out_dir')})
