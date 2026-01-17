#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must be a YAML mapping at top-level.")
    return obj


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k == "extends":
            continue
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve(path: Path, *, seen: set[Path] | None = None) -> dict[str, Any]:
    if seen is None:
        seen = set()
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Cycle detected in extends chain at {path}")
    seen.add(path)

    obj = _load_yaml(path)
    ext = obj.get("extends")
    if not ext:
        return obj

    if isinstance(ext, str):
        parents = [ext]
    elif isinstance(ext, list) and all(isinstance(x, str) for x in ext):
        parents = list(ext)
    else:
        raise ValueError(f"{path}: extends must be a string or list of strings.")

    merged: dict[str, Any] = {}
    for rel in parents:
        p = (path.parent / rel).resolve()
        parent_obj = resolve(p, seen=seen)
        merged = _deep_merge(merged, parent_obj)

    return _deep_merge(merged, obj)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="YAML file to resolve (supports `extends:`)")
    ap.add_argument("--out", default="", help="Optional output path; prints to stdout if omitted")
    args = ap.parse_args()

    resolved = resolve(Path(args.path))
    text = yaml.safe_dump(resolved, sort_keys=False)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"[+] Wrote {out_path}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()

