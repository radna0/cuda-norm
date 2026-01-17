#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pack_index(packs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in packs:
        name = str(p.get("pack") or "")
        if not name:
            continue
        out[name] = p
    return out


def merge_eaft_jsons(paths: list[Path], *, label: str | None = None) -> dict[str, Any]:
    if not paths:
        raise ValueError("No inputs.")

    merged: dict[str, Any] = {}
    merged_meta: dict[str, Any] = {}
    merged_packs: dict[str, dict[str, Any]] = {}
    seq_lens: set[int] = set()

    for i, p in enumerate(paths):
        obj = _load(p)
        meta = dict(obj.get("meta") or {})
        packs = list(obj.get("packs") or [])
        if not isinstance(packs, list):
            raise ValueError(f"{p} packs is not a list.")

        if i == 0:
            merged_meta = meta
        else:
            for k in ("model_id", "model_path", "top_k", "entropy_topk"):
                if str(merged_meta.get(k, "")) and str(meta.get(k, "")) and merged_meta.get(k) != meta.get(k):
                    raise ValueError(f"Mismatch meta[{k!r}] across inputs: {merged_meta.get(k)!r} vs {meta.get(k)!r}")

        for s in meta.get("seq_lens") or []:
            try:
                seq_lens.add(int(s))
            except Exception:
                pass

        for pack_name, pack_obj in _pack_index(packs).items():
            seq = dict(pack_obj.get("seq") or {})
            if pack_name not in merged_packs:
                merged_packs[pack_name] = {"pack": pack_name, "seq": {}}
            dst_seq: dict[str, Any] = merged_packs[pack_name]["seq"]
            for seq_len, metrics in seq.items():
                if seq_len in dst_seq:
                    continue
                dst_seq[str(seq_len)] = metrics

    merged_meta["seq_lens"] = sorted(seq_lens)
    if label:
        merged_meta["label"] = str(label)
    merged["meta"] = merged_meta
    merged["packs"] = [merged_packs[k] for k in sorted(merged_packs.keys())]
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--label", default="", help="Optional label to stamp into meta")
    ap.add_argument("inputs", nargs="+", help="Input EAFT JSONs to merge")
    args = ap.parse_args()

    out_path = Path(args.out)
    inputs = [Path(x) for x in args.inputs]
    out = merge_eaft_jsons(inputs, label=(args.label or None))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[+] Wrote {out_path}")


if __name__ == "__main__":
    main()

