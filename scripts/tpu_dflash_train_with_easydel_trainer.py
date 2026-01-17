#!/usr/bin/env python3
"""Train a DFlash draft model on TPU using EasyDeL's Trainer loop (cache-first).

This is the intended high-throughput path:
  1) Build a teacher cache dir once (expensive):
     `scripts/tpu_dflash_build_teacher_cache.py --out-dir /dev/shm/dflash_cache/...`
  2) Train draft many steps from the cache (cheap, scalable batch):
     `scripts/tpu_dflash_train_with_easydel_trainer.py --cache-dir ... --teacher-snapshot-dir ...`

Notes
-----
- Training NEVER runs the teacher forward; it only uses the frozen `lm_head.weight`.
- Full-vocab CE is computed with tp-sharded lm_head and dp-sharded batch.
- For best IO: put cache_dir + EasyDeL checkpoints under `/dev/shm`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import json


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, help="Teacher cache directory (meta.json + .npy files).")
    ap.add_argument("--teacher-snapshot-dir", required=True, help="HF snapshot dir (config.json + safetensors).")
    ap.add_argument("--save-directory", default="/dev/shm/dflash-checkpoints", help="EasyDeL checkpoint root dir.")
    ap.add_argument("--model-name", default="gptoss-dflash-draft", help="Checkpoint subdir name under save-directory.")
    ap.add_argument("--resume", type=str, default="true", help="Resume from latest complete run-* checkpoint.")
    ap.add_argument("--resume-strict", type=str, default="false", help="Error if resume requested but no checkpoint.")
    ap.add_argument("--resume-path", default=None, help="Explicit run-* directory to resume from.")
    ap.add_argument("--max-training-steps", type=int, default=2000)
    ap.add_argument("--total-batch-size", type=int, default=128, help="Global batch per step (must be divisible by dp).")
    ap.add_argument("--grad-accum-steps", type=int, default=1)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--do-last-save", type=str, default="true")
    ap.add_argument("--log-steps", type=int, default=10)
    ap.add_argument("--report-steps", type=int, default=10)
    ap.add_argument("--draft-layers", type=int, default=8)
    ap.add_argument("--mlp-ratio", type=float, default=4.0)
    ap.add_argument("--qk-norm", type=str, default="true")
    ap.add_argument("--remat", type=str, default="true")
    ap.add_argument("--vocab-chunk-size", type=int, default=8192, help="Chunk size within each tp shard (VRAM saver).")
    ap.add_argument("--dp", type=int, default=8)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--spmd", type=str, default="false")
    ap.add_argument("--prefetch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device-prefetch", type=int, default=2)
    ap.add_argument("--disable-wandb", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    local_easydel = repo_root / "external" / "EasyDeL"
    if local_easydel.exists():
        sys.path.insert(0, str(local_easydel))
    _load_dotenv(repo_root / ".env")
    # Keep all caches off the root FS (often small) and inside /dev/shm.
    # This TPU box runs older eformer/ejkernel; we carry local shims. Skip the
    # strict EasyDeL version gate so imports work.
    os.environ.setdefault("EASYDEL_SKIP_VERSION_CHECK", "1")
    os.environ.setdefault("HF_HOME", "/dev/shm/hf")
    os.environ.setdefault("HF_HUB_CACHE", "/dev/shm/hf/hub")
    os.environ.setdefault("XDG_CACHE_HOME", "/dev/shm/xdg")
    os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")
    os.environ.setdefault("TMPDIR", "/dev/shm/tmp")
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    # Persistent compilation caches can explode in size and fill /dev/shm.
    # Default: disabled (still uses in-memory compilation cache).
    if os.environ.get("ENABLE_JAX_PERSISTENT_COMPILATION_CACHE", "0").lower() in ("1", "true", "yes", "y", "on"):
        os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/dev/shm/jax_compilation_cache_dflash")
        Path(os.environ["JAX_COMPILATION_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    else:
        os.environ.pop("JAX_COMPILATION_CACHE_DIR", None)

    qk_norm = str(args.qk_norm).lower() in ("1", "true", "yes", "y", "on")
    remat = str(args.remat).lower() in ("1", "true", "yes", "y", "on")
    do_last_save = str(args.do_last_save).lower() in ("1", "true", "yes", "y", "on")
    spmd = str(args.spmd).lower() in ("1", "true", "yes", "y", "on")
    resume = str(args.resume).lower() in ("1", "true", "yes", "y", "on")
    resume_strict = str(args.resume_strict).lower() in ("1", "true", "yes", "y", "on")

    from easydel.trainers.dflash_config import DFlashConfig
    from easydel.trainers.dflash_trainer import DFlashTrainer

    cache_dir = Path(args.cache_dir).resolve()
    meta_path = cache_dir / "meta.json"
    max_length = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # Draft training predicts only the (block_size-1) draft tokens, not the full
        # context window; set this so throughput metrics are meaningful.
        max_length = max(1, int(meta["block_size"]) - 1)

    cfg = DFlashConfig(
        cache_dir=str(cache_dir),
        teacher_snapshot_dir=str(Path(args.teacher_snapshot_dir).resolve()),
        save_directory=str(Path(args.save_directory).resolve()),
        model_name=str(args.model_name),
        max_training_steps=int(args.max_training_steps),
        num_train_epochs=1,
        max_length=int(max_length) if max_length is not None else None,
        total_batch_size=int(args.total_batch_size),
        gradient_accumulation_steps=int(args.grad_accum_steps),
        learning_rate=float(args.learning_rate),
        warmup_steps=int(args.warmup_steps),
        weight_decay=float(args.weight_decay),
        save_steps=int(args.save_steps),
        do_last_save=bool(do_last_save),
        log_steps=int(args.log_steps),
        report_steps=int(args.report_steps),
        do_eval=False,
        use_wandb=not bool(args.disable_wandb),
        draft_layers=int(args.draft_layers),
        mlp_ratio=float(args.mlp_ratio),
        qk_norm=bool(qk_norm),
        remat=bool(remat),
        vocab_chunk_size=int(args.vocab_chunk_size),
        spmd=bool(spmd),
        dp=int(args.dp),
        tp=int(args.tp),
        dataloader_prefetch=int(args.prefetch),
        dataloader_workers=int(args.workers),
        device_prefetch=int(args.device_prefetch),
        resume=bool(resume),
        resume_strict=bool(resume_strict),
        resume_path=str(args.resume_path) if args.resume_path else None,
    )

    # Persist a run manifest next to checkpoints for reproducibility.
    run_dir = Path(cfg.save_directory) / str(cfg.model_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "cache_dir": cfg.cache_dir,
                "teacher_snapshot_dir": cfg.teacher_snapshot_dir,
                "model_name": cfg.model_name,
                "save_directory": cfg.save_directory,
                "max_training_steps": cfg.max_training_steps,
                "total_batch_size": cfg.total_batch_size,
                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                "draft_layers": cfg.draft_layers,
                "mlp_ratio": cfg.mlp_ratio,
                "qk_norm": cfg.qk_norm,
                "remat": getattr(cfg, "remat", None),
                "vocab_chunk_size": cfg.vocab_chunk_size,
                "dp": cfg.dp,
                "tp": cfg.tp,
                "spmd": cfg.spmd,
                "dataloader_prefetch": cfg.dataloader_prefetch,
                "dataloader_workers": getattr(cfg, "dataloader_workers", None),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    trainer = DFlashTrainer(arguments=cfg, processing_class=None)
    trainer.train()


if __name__ == "__main__":
    main()
