#!/usr/bin/env python3
import os
import glob
import argparse
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import login, HfApi


def find_files(root: Path, split: str):
    # accept both .jsonl and .jsonl.gz
    patt1 = str(root / split / "converted" / "*.jsonl")
    patt2 = str(root / split / "converted" / "*.jsonl.gz")
    files = sorted(glob.glob(patt1)) + sorted(glob.glob(patt2))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default="/dev/shm/nemotron_math_v2_harmony",
        help="Root directory containing <split>/converted/*.jsonl(.gz)",
    )
    ap.add_argument(
        "--repo",
        type=str,
        default="radna0/nemotron-math-v2-harmony",
        help="Target HF dataset repo_id (username/name)",
    )
    ap.add_argument(
        "--private", action="store_true", help="Create as private (default: public)"
    )
    ap.add_argument(
        "--max_files_per_split",
        type=int,
        default=0,
        help="Debug: only push first N files per split (0=all)",
    )
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set. Export HF_TOKEN first.")

    # Faster transfer if hf_transfer is installed
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    # Login (stores token in memory for this run; no git credential needed)
    login(token=token, add_to_git_credential=False)

    api = HfApi()

    # Create repo if needed (public by default)
    api.create_repo(
        repo_id=args.repo,
        repo_type="dataset",
        private=bool(args.private),
        exist_ok=True,
    )

    root = Path(args.root)

    splits = ["low", "medium", "high_part00", "high_part01", "high_part02"]

    for split in splits:
        files = find_files(root, split)
        if not files:
            print(f"[skip] split={split}: no files under {root/split/'converted'}")
            continue

        if args.max_files_per_split and len(files) > args.max_files_per_split:
            files = files[: args.max_files_per_split]

        print(f"[load] split={split}: {len(files)} files")
        # load_dataset can ingest many jsonl/jsonl.gz files
        ds = load_dataset("json", data_files=files, split="train")

        print(f"[push] split={split}: num_rows={ds.num_rows:,} -> {args.repo}")
        # push as a named split
        ds.push_to_hub(
            args.repo,
            split=split,
            private=bool(args.private),
        )

    print("[done] all available splits pushed.")


if __name__ == "__main__":
    main()
