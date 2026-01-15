#!/usr/bin/env python3
"""Batch-size sweep for TPU DFlash draft training (subprocess-per-batch).

Why subprocess?
- JAX/XLA retains compiled executables and buffers; sweeping in-process can
  produce misleading OOMs. A clean process per batch gives a real max batch.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Result:
    batch_size: int
    ok: bool
    exit_code: int
    log_path: str


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--teacher-snapshot-dir", required=True)
    ap.add_argument("--dp", type=int, default=8)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--spmd", type=str, default="false")
    ap.add_argument("--draft-layers", type=int, default=8)
    ap.add_argument("--mlp-ratio", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=5, help="Steps per batch size (includes compile).")
    ap.add_argument("--log-dir", default="/dev/shm/tpu_logs")
    ap.add_argument("--batch-sizes", default="32,64,96,128,160,192,224,256")
    ap.add_argument("--vocab-chunk-size", type=int, default=0, help="Set >0 only for tp=1 debug.")
    ap.add_argument("--remat", type=str, default="true")
    ap.add_argument("--prefetch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device-prefetch", type=int, default=2)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    train_py = repo_root / "scripts" / "tpu_dflash_train_with_easydel_trainer.py"
    py = repo_root / ".venv-easydel" / "bin" / "python"
    if not py.exists():
        raise FileNotFoundError(f"Missing venv python at {py}")

    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = [int(x.strip()) for x in str(args.batch_sizes).split(",") if x.strip()]
    results: list[Result] = []

    for bs in batch_sizes:
        run_name = f"dflash_bs{bs}_dp{args.dp}_tp{args.tp}"
        log_path = log_dir / f"{run_name}.log"
        cmd = [
            str(py),
            str(train_py),
            "--cache-dir",
            str(Path(args.cache_dir).resolve()),
            "--teacher-snapshot-dir",
            str(Path(args.teacher_snapshot_dir).resolve()),
            "--max-training-steps",
            str(int(args.steps)),
            "--total-batch-size",
            str(int(bs)),
            "--save-steps",
            "999999999",
            "--do-last-save",
            "false",
            "--log-steps",
            "1",
            "--report-steps",
            "1",
            "--draft-layers",
            str(int(args.draft_layers)),
            "--mlp-ratio",
            str(float(args.mlp_ratio)),
            "--dp",
            str(int(args.dp)),
            "--tp",
            str(int(args.tp)),
            "--spmd",
            str(args.spmd),
            "--vocab-chunk-size",
            str(int(args.vocab_chunk_size)),
            "--remat",
            str(args.remat),
            "--prefetch",
            str(int(args.prefetch)),
            "--workers",
            str(int(args.workers)),
            "--device-prefetch",
            str(int(args.device_prefetch)),
            "--disable-wandb",
        ]
        env = os.environ.copy()
        env.setdefault("TMPDIR", "/dev/shm/tmp")
        env.setdefault("XDG_CACHE_HOME", "/dev/shm/xdg")
        env.setdefault("JAX_COMPILATION_CACHE_DIR", "/dev/shm/jax_compilation_cache_dflash")
        env.setdefault("HF_HOME", "/dev/shm/hf")
        env.setdefault("HF_HUB_CACHE", "/dev/shm/hf/hub")

        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"$ {shlex.join(cmd)}\n\n")
            f.flush()
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

        ok = proc.returncode == 0
        results.append(Result(batch_size=bs, ok=ok, exit_code=proc.returncode, log_path=str(log_path)))

    out = {
        "dp": int(args.dp),
        "tp": int(args.tp),
        "steps": int(args.steps),
        "batch_sizes": batch_sizes,
        "results": [r.__dict__ for r in results],
    }
    summary_path = log_dir / "dflash_batch_sweep_summary.json"
    summary_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[+] Wrote {summary_path}")
    for r in results:
        status = "OK" if r.ok else f"FAIL({r.exit_code})"
        print(f"{status}\tbs={r.batch_size}\t{r.log_path}")


if __name__ == "__main__":
    main()
