from __future__ import annotations

import os
import shutil
from pathlib import Path


def _copy_tree(src: Path, dst: Path) -> None:
    for root, dirs, files in os.walk(src):
        root_p = Path(root)
        rel = root_p.relative_to(src)
        dst_root = dst / rel
        dst_root.mkdir(parents=True, exist_ok=True)
        for d in dirs:
            (dst_root / d).mkdir(parents=True, exist_ok=True)
        for f in files:
            s = root_p / f
            t = dst_root / f
            try:
                # The SGLang source tree contains a few symlinks (e.g. formatting configs)
                # that may be broken once synced onto Kaggle. Preserve symlinks rather
                # than following them.
                try:
                    if t.exists() or t.is_symlink():
                        t.unlink()
                except Exception:
                    pass
                shutil.copy2(s, t, follow_symlinks=False)
            except FileNotFoundError:
                # Broken symlink or transient file; skip.
                continue


def main() -> None:
    # This script is intended to run on the remote Jupyter machine (Kaggle) after:
    #   pip install 'sglang[all]'
    #
    # It overlays our local patched SGLang python sources into site-packages so we can
    # use the DFLASH PR changes without rebuilding wheels.
    import sglang  # noqa: F401

    import sglang as sglang_pkg

    repo_root = Path(__file__).resolve().parents[1]
    override = os.environ.get("SGLANG_OVERLAY_SRC", "").strip()
    src = Path(override).expanduser().resolve() if override else (repo_root / "sglang-flashinfer" / "python" / "sglang")
    if not src.exists():
        raise FileNotFoundError(f"Expected SGLang source tree at {src}")

    pkg_dir = Path(sglang_pkg.__file__).resolve().parent

    _copy_tree(src, pkg_dir)

    # Clean pycache to avoid stale bytecode issues.
    for p in pkg_dir.rglob("__pycache__"):
        try:
            shutil.rmtree(p)
        except Exception:
            pass

    print(f"[+] Overlayed {src} -> {pkg_dir}", flush=True)


if __name__ == "__main__":
    main()
