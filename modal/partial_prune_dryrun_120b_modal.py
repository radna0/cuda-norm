"""
CPU-only 120B partial prune dry-run (Modal).

This is Task 2B (cost probe), not full pruning:
- Load *only* one MoE layer's expert/router tensors for GPT-OSS-120B
- Slice experts to keep_frac (e.g. 0.5)
- Measure download / load+slice / write timings and peak RSS

Outputs:
- Writes `reports/120b_partial_prune_dryrun.md` locally (entrypoint)
- Writes JSON + sliced safetensors into the mounted data volume

Run (always log to unsloth_logs/):
  mkdir -p unsloth_logs
  ts=$(date +%Y%m%d_%H%M%S)
  nohup modal run modal/partial_prune_dryrun_120b_modal.py --layer 0 --keep-frac 0.5 \
    > "unsloth_logs/120b_partial_prune_dryrun_${ts}.log" 2>&1 &
"""

from __future__ import annotations

import json
import os
import resource
import time
from math import ceil
from pathlib import Path
from typing import Any

import modal

APP_NAME = "partial-prune-dryrun-120b"


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
def partial_prune_dryrun(
    *,
    model_id: str,
    layer: int,
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
    if not (0 <= int(layer) < num_layers):
        raise ValueError(f"layer must be in [0, {num_layers-1}]")

    idx_path = hf_hub_download(str(model_id), filename="model.safetensors.index.json", token=token)
    idx = _read_json(idx_path)
    weight_map = idx.get("weight_map") or {}
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("Index JSON missing weight_map")

    keys = _keys_for_layer(int(layer))
    missing = [k for k in keys if k not in weight_map]
    if missing:
        raise RuntimeError(f"Missing keys in weight_map: {missing}")

    keep_n = max(1, min(num_experts, int(ceil(float(keep_frac) * num_experts))))
    keep = list(range(keep_n))

    shard_files = sorted({weight_map[k] for k in keys})
    shard_paths: dict[str, str] = {}
    dl_t0 = time.time()
    for fn in shard_files:
        shard_paths[fn] = hf_hub_download(str(model_id), filename=str(fn), token=token)
    dl_s = time.time() - dl_t0

    tensors: dict[str, Any] = {}
    load_t0 = time.time()
    for k in keys:
        fn = weight_map[k]
        with safe_open(shard_paths[fn], framework="pt", device="cpu") as f:
            t = f.get_tensor(k)
        if int(getattr(t, "shape", [0])[0]) != num_experts:
            raise RuntimeError(
                f"Unexpected leading dim for {k}: shape={tuple(getattr(t,'shape',()))} expected dim0={num_experts}"
            )
        tensors[k] = t[keep].contiguous()
    load_s = time.time() - load_t0

    out_dir = Path("/root/data") / str(out_subdir).strip().strip("/")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"layer{int(layer)}_keep{keep_n}.safetensors"
    write_t0 = time.time()
    save_file(tensors, str(out_file), metadata={"format": "pt", "harmony": "partial_prune_dryrun"})
    write_s = time.time() - write_t0

    data_volume.commit()
    hf_cache_volume.commit()

    total_s = time.time() - t0
    out_size_gib = out_file.stat().st_size / (1024.0**3)
    dl_size_gib = sum(Path(p).stat().st_size for p in shard_paths.values()) / (1024.0**3)

    return {
        "model_id": str(model_id),
        "layer": int(layer),
        "keep_frac": float(keep_frac),
        "keep_n": int(keep_n),
        "num_experts": int(num_experts),
        "num_layers": int(num_layers),
        "unique_shards_downloaded": int(len(shard_files)),
        "download_gib": float(dl_size_gib),
        "output_gib": float(out_size_gib),
        "timings_s": {
            "download": float(dl_s),
            "load_slice": float(load_s),
            "write": float(write_s),
            "total": float(total_s),
        },
        "peak_rss_gib": float(_maxrss_gib()),
        "shards": [str(x) for x in shard_files],
        "output_file": str(out_file),
    }


@app.local_entrypoint()
def main(
    model_id: str = DEFAULT_MODEL_ID_120B,
    layer: int = 0,
    keep_frac: float = 0.5,
):
    run_id = time.strftime("%Y%m%d_%H%M%S")
    res = partial_prune_dryrun.remote(
        model_id=str(model_id),
        layer=int(layer),
        keep_frac=float(keep_frac),
        out_subdir=f"artifacts/120b_partial_prune_dryrun/{run_id}",
    )

    md_path = Path("reports/120b_partial_prune_dryrun.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        "\n".join(
            [
                "# 120B partial prune dry-run",
                "",
                f"- Model: `{res['model_id']}`",
                f"- Layer: {res['layer']}",
                f"- Keep: {res['keep_n']}/{res['num_experts']} ({res['keep_frac']:.2f})",
                f"- Downloaded shards: {res['unique_shards_downloaded']} ({res['download_gib']:.2f} GiB)",
                f"- Output file: `{res['output_file']}` ({res['output_gib']:.2f} GiB)",
                "",
                "## Timings",
                "",
                f"- download: {res['timings_s']['download']:.1f}s",
                f"- load+slice: {res['timings_s']['load_slice']:.1f}s",
                f"- write: {res['timings_s']['write']:.1f}s",
                f"- total: {res['timings_s']['total']:.1f}s",
                "",
                "## Peak memory",
                "",
                f"- peak RSS: {res['peak_rss_gib']:.2f} GiB",
                "",
                "## Reproduce",
                "",
                "```bash",
                "modal run modal/partial_prune_dryrun_120b_modal.py --layer 0 --keep-frac 0.5",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[+] Wrote {md_path}")
    print("[RESULT]", res)
