from __future__ import annotations

import argparse
import os
from pathlib import Path


def _iter_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        # Never upload partial downloads / resume artifacts.
        if rel.endswith(".part") or rel.endswith(".part.json") or rel.endswith(".tmp"):
            continue
        out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--repo-id", required=True, help="e.g. radna0/gptoss20b-keep24of32-k75-eaftreap")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--repo-type", default="model", choices=["model"])
    ap.add_argument("--commit-message", default="Upload pruned GPT-OSS-20B keep24/32 EAFT-REAP checkpoint")
    ap.add_argument("--revision", default="main")
    args = ap.parse_args()

    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
    if not token:
        raise SystemExit("[err] HF_TOKEN/HUGGINGFACE_HUB_TOKEN is not set in env")

    src_dir = Path(args.src_dir).resolve()
    if not src_dir.exists():
        raise SystemExit(f"[err] src dir does not exist: {src_dir}")
    if not (src_dir / "config.json").exists():
        raise SystemExit(f"[err] src dir missing config.json: {src_dir}")

    files = _iter_files(src_dir)
    if not files:
        raise SystemExit(f"[err] no files to upload under: {src_dir}")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=str(args.repo_id),
        repo_type=str(args.repo_type),
        private=bool(args.private),
        exist_ok=True,
    )

    # Use HF Transfer for large files if installed.
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    # Upload everything as a single commit.
    # (huggingface_hub will LFS large files automatically.)
    api.upload_folder(
        folder_path=str(src_dir),
        repo_id=str(args.repo_id),
        repo_type=str(args.repo_type),
        revision=str(args.revision),
        commit_message=str(args.commit_message),
        ignore_patterns=["*.part", "*.part.json", "*.tmp"],
    )

    print(f"[+] uploaded {len(files)} files to {args.repo_id} ({'private' if args.private else 'public'})")


if __name__ == "__main__":
    main()

