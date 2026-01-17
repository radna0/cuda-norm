#!/usr/bin/env python3
"""Patch legacy EasyDeL DFlash run-*/config.json with missing parity fields.

Early TPU DFlash runs stored only the draft-architecture core in `run-*/config.json`
and omitted:
  - target_layer_ids
  - add_one_for_pre_layer_capture

Those fields are required for correct verify hidden-feature extraction parity
between:
  - teacher-cache build (verify executor capture points)
  - inference-time DFlash verification (eSurge verify executor)

This utility fills the missing fields from the associated cache meta.json, which
is referenced by <model_root>/run_config.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to run-*/ directory (contains config.json).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}")

    cfg = _read_json(cfg_path)
    changed = False

    if not cfg.get("target_layer_ids"):
        root_cfg_path = run_dir.parent / "run_config.json"
        if not root_cfg_path.exists():
            raise FileNotFoundError(
                f"Missing {root_cfg_path}; cannot infer cache_dir to locate target_layer_ids."
            )
        root_cfg = _read_json(root_cfg_path)
        cache_dir = root_cfg.get("cache_dir")
        if not cache_dir:
            raise ValueError(f"{root_cfg_path} does not define cache_dir")
        meta_path = Path(str(cache_dir)).expanduser().resolve() / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing {meta_path}")
        meta = _read_json(meta_path)
        layer_ids = meta.get("target_layer_ids")
        if not layer_ids:
            raise ValueError(f"{meta_path} missing target_layer_ids")
        cfg["target_layer_ids"] = [int(x) for x in layer_ids]
        changed = True

    if "add_one_for_pre_layer_capture" not in cfg or cfg.get("add_one_for_pre_layer_capture") is None:
        cfg["add_one_for_pre_layer_capture"] = True
        changed = True

    if not changed:
        print(f"[*] No changes needed: {cfg_path}")
        return

    if args.dry_run:
        print(f"[*] Would patch: {cfg_path}")
        print(json.dumps(cfg, indent=2, sort_keys=True))
        return

    _write_json(cfg_path, cfg)
    print(f"[+] Patched: {cfg_path}")


if __name__ == "__main__":
    main()

