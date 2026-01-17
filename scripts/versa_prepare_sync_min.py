from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory for minimal Versa sync tree.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out = Path(args.out).expanduser().resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    include_files = [
        repo_root / "README.md",
        repo_root / ".env",
        # EAFT / pruning evaluation (SGLang logprob collection).
        repo_root / "modal" / "collect_calib_packs_eaft_single.py",
        # Pruning track (runs in Kaggle via PRUNING_LOCAL_MODE=1).
        repo_root / "modal" / "gpt_oss_pruning_track.py",
        repo_root / "scripts" / "kaggle_dflash_smoke.sh",
        repo_root / "scripts" / "kaggle_dflash_gptoss20b_pipeline.sh",
        repo_root / "scripts" / "kaggle_dflash_bench_from_easydel_run.sh",
        repo_root / "scripts" / "kaggle_dflash_bench_sweep.sh",
        repo_root / "scripts" / "kaggle_sglang_dflash_smoke_from_tar.sh",
        repo_root / "scripts" / "kaggle_smoke_pruned_eaftreap.py",
        repo_root / "scripts" / "kaggle_decode_bench_sglang.py",
        repo_root / "scripts" / "sglang_overlay_install.py",
        repo_root / "scripts" / "convert_hf_dflash_ckpt_to_sglang.py",
        repo_root / "scripts" / "convert_easydel_dflash_ckpt_to_sglang.py",
        repo_root / "scripts" / "sglang_dflash_draft_load_smoke.py",
        repo_root / "scripts" / "dflash_gptoss20b_train_kaggle.py",
        repo_root / "scripts" / "dflash_gptoss20b_bench_sglang_kaggle.py",
        repo_root / "scripts" / "test_convert_hf_dflash_ckpt_to_sglang_smoke.py",
    ]
    for f in include_files:
        if not f.exists():
            raise FileNotFoundError(f"Missing required file: {f}")

    # Layout mirrors the repo for relative paths referenced by remote scripts.
    (out / "scripts").mkdir(parents=True, exist_ok=True)
    (out / "modal").mkdir(parents=True, exist_ok=True)
    for f in include_files:
        if f.name in ("README.md", ".env"):
            shutil.copy2(f, out / f.name)
        elif f.parent.name == "modal":
            shutil.copy2(f, out / "modal" / f.name)
        else:
            shutil.copy2(f, out / "scripts" / f.name)

    # Python packages needed by the trainer.
    _copytree(repo_root / "dflash_gptoss", out / "dflash_gptoss")

    # Patched SGLang python sources used by the overlay installer.
    _copytree(
        repo_root / "sglang-flashinfer" / "python" / "sglang",
        out / "sglang-flashinfer" / "python" / "sglang",
    )

    manifest = {
        "root": str(repo_root),
        "synced": [
            "README.md",
            ".env",
            "modal/collect_calib_packs_eaft_single.py",
            "modal/gpt_oss_pruning_track.py",
            "scripts/* (selected)",
            "dflash_gptoss/",
            "sglang-flashinfer/python/sglang/",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(str(out), flush=True)


if __name__ == "__main__":
    main()
