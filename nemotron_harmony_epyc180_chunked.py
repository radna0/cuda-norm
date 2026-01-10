#!/usr/bin/env python3
import os
import glob
import argparse
from pathlib import Path

from huggingface_hub import HfApi, login


def list_local_files(root: Path, split: str):
    # Only local files. No URLs.
    patt = str(root / split / "converted" / "*.jsonl*")
    files = sorted(glob.glob(patt))
    # filter to real files
    return [f for f in files if Path(f).is_file()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root containing <split>/converted/*.jsonl(.gz)",
    )
    ap.add_argument(
        "--repo", type=str, required=True, help="e.g. radna0/nemotron-math-v2-harmony"
    )
    ap.add_argument("--private", action="store_true", help="Default is public")
    ap.add_argument(
        "--max_files_per_split",
        type=int,
        default=0,
        help="Debug: upload only first N files per split",
    )
    ap.add_argument(
        "--commit_every",
        type=int,
        default=50,
        help="Batch commits to avoid thousands of commits",
    )
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN env var not set")

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    login(token=token, add_to_git_credential=False)

    api = HfApi()
    api.create_repo(
        repo_id=args.repo,
        repo_type="dataset",
        private=bool(args.private),
        exist_ok=True,
    )

    root = Path(args.root)
    splits = ["low", "medium", "high_part00", "high_part01", "high_part02"]

    for split in splits:
        files = list_local_files(root, split)
        if not files:
            print(
                f"[skip] split={split}: no local files under {root/split/'converted'}"
            )
            continue

        if args.max_files_per_split and len(files) > args.max_files_per_split:
            files = files[: args.max_files_per_split]

        # Upload in batches using commit_operations to avoid 1 commit per file
        ops = []
        uploaded = 0

        for fp in files:
            p = Path(fp)
            path_in_repo = f"data/{split}/{p.name}"
            ops.append(
                {
                    "op": "upload",
                    "path_or_fileobj": str(p),
                    "path_in_repo": path_in_repo,
                }
            )

            # commit every N files
            if len(ops) >= args.commit_every:
                api.create_commit(
                    repo_id=args.repo,
                    repo_type="dataset",
                    operations=ops,
                    commit_message=f"Upload {split} batch ({uploaded+1}-{uploaded+len(ops)})",
                )
                uploaded += len(ops)
                print(
                    f"[commit] split={split}: uploaded {uploaded}/{len(files)}",
                    flush=True,
                )
                ops = []

        # final commit
        if ops:
            api.create_commit(
                repo_id=args.repo,
                repo_type="dataset",
                operations=ops,
                commit_message=f"Upload {split} final batch",
            )
            uploaded += len(ops)
            print(
                f"[commit] split={split}: uploaded {uploaded}/{len(files)}", flush=True
            )

        # Optionally upload samples_text if present
        sample_dir = root / split / "converted" / "samples_text"
        if sample_dir.exists():
            sample_files = sorted(sample_dir.glob("*.txt"))
            if sample_files:
                ops = []
                for sp in sample_files[:200]:
                    ops.append(
                        {
                            "op": "upload",
                            "path_or_fileobj": str(sp),
                            "path_in_repo": f"samples/{split}/{sp.name}",
                        }
                    )
                api.create_commit(
                    repo_id=args.repo,
                    repo_type="dataset",
                    operations=ops,
                    commit_message=f"Upload samples for {split}",
                )
                print(
                    f"[commit] split={split}: uploaded samples ({min(200, len(sample_files))})",
                    flush=True,
                )

    # README
    readme = (
        "# Nemotron-Math-v2 Harmony formatted (raw shards)\n\n"
        "This dataset repo contains raw JSONL / JSONL.GZ shards.\n"
        "Each line is a JSON object with a single key: `text`.\n\n"
        "Files are organized under:\n"
        "- `data/<split>/...`\n\n"
        "To load a subset:\n"
        "```python\n"
        "from datasets import load_dataset\n"
        "ds = load_dataset('json', data_files=['data/low/<file>.jsonl.gz'], split='train')\n"
        "```\n"
    )
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
        commit_message="Add README",
    )

    print("[done] uploaded raw shards only (no Arrow build, no redownload).")


if __name__ == "__main__":
    main()
