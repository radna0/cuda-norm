#!/usr/bin/env python3
"""
Launch REAP-lite saliency profiling on Modal and log to `unsloth_logs/`.

This is a thin wrapper around:
  modal run modal/gpt_oss_pruning_track.py --task reap_saliency_20b ...

Example:
  python scripts/reap_saliency_profile_20b.py --num-rows 400 --max-seq-length 1024 --batch-size 1
  python scripts/reap_saliency_profile_20b.py --domain math --num-rows 400 --max-seq-length 1024 --batch-size 1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=os.environ.get("MODEL_ID_20B", "openai/gpt-oss-20b"))
    ap.add_argument("--dataset-id", default=os.environ.get("DATASET_ID", "radna0/harmony-nemotron-cpu-artifacts"))
    ap.add_argument("--dataset-split", default=os.environ.get("DATASET_SPLIT", "train"))
    ap.add_argument("--text-column", default=os.environ.get("TEXT_COLUMN", "text"))
    ap.add_argument("--domain", default=os.environ.get("DOMAIN", ""))
    ap.add_argument("--num-rows", type=int, default=int(os.environ.get("NUM_ROWS", "500")))
    ap.add_argument("--max-seq-length", type=int, default=int(os.environ.get("MAX_SEQ_LENGTH", "1024")))
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "1")))
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    logs_dir = repo_root / "unsloth_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_domain = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (args.domain or "").strip())
    log_name = f"reap_saliency_20b_{safe_domain or 'all'}_{ts}.log"
    log_path = logs_dir / log_name
    pid_path = log_path.with_suffix(".log.pid")

    cmd = [
        "modal",
        "run",
        "modal/gpt_oss_pruning_track.py",
        "--task",
        "reap_saliency_20b",
        "--model-id-20b",
        str(args.model_id),
        "--dataset-id",
        str(args.dataset_id),
        "--dataset-split",
        str(args.dataset_split),
        "--text-column",
        str(args.text_column),
        "--domain",
        str(args.domain or ""),
        "--num-rows",
        str(int(args.num_rows)),
        "--max-seq-length",
        str(int(args.max_seq_length)),
        "--batch-size",
        str(int(args.batch_size)),
    ]

    with log_path.open("wb") as f:
        p = subprocess.Popen(cmd, cwd=str(repo_root), stdout=f, stderr=subprocess.STDOUT)
    pid_path.write_text(str(p.pid), encoding="utf-8")
    print(f"started pid={p.pid} log={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

