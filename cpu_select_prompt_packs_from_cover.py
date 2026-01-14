#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def _parse_sizes(spec: str) -> list[int]:
    out: list[int] = []
    for part in (spec or "").split(","):
        s = part.strip()
        if not s:
            continue
        out.append(int(s))
    out = sorted(set(out))
    if not out:
        raise SystemExit("--sizes must be non-empty, e.g. 1000,10000,100000")
    if any(x <= 0 for x in out):
        raise SystemExit("--sizes must be > 0")
    return out


def _parse_ratios(spec: str) -> dict[str, float]:
    if not spec.strip():
        return {"tool": 0.40, "reasoning": 0.40, "general": 0.20}
    try:
        obj = json.loads(spec)
    except Exception as e:
        raise SystemExit(f"--mix_ratios_json must be valid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise SystemExit("--mix_ratios_json must be a JSON object")
    out: dict[str, float] = {}
    for k, v in obj.items():
        kk = str(k).strip()
        if not kk:
            continue
        out[kk] = float(v)
    for k in ("tool", "reasoning", "general"):
        out.setdefault(k, 0.0)
    s = sum(out.values())
    if s <= 0:
        raise SystemExit("--mix_ratios_json must sum to > 0")
    return {k: float(v / s) for k, v in out.items()}


def _alloc_counts(total: int, ratios: dict[str, float], keys: list[str]) -> dict[str, int]:
    raw = {k: float(total) * float(ratios.get(k, 0.0)) for k in keys}
    base = {k: int(np.floor(v)) for k, v in raw.items()}
    rem = total - sum(base.values())
    if rem <= 0:
        return base
    # Deterministic remainder distribution: largest fractional parts first.
    fracs = sorted(((k, raw[k] - base[k]) for k in keys), key=lambda kv: (-kv[1], kv[0]))
    for i in range(rem):
        base[fracs[i % len(fracs)][0]] += 1
    return base


def _select_within_group(tbl: pa.Table, k: int, *, rng: np.random.Generator, dense_fraction: float) -> pa.Table:
    if k <= 0 or tbl.num_rows == 0:
        return tbl.slice(0, 0)
    if k >= tbl.num_rows:
        return tbl

    dense_n = int(round(k * dense_fraction))
    dense_n = max(0, min(dense_n, k))
    tail_n = k - dense_n

    # Prefer representative buckets first (largest bucket_count).
    if "bucket_count" in tbl.column_names:
        sort_idx = pc.sort_indices(tbl, sort_keys=[("bucket_count", "descending")])
        tbl_sorted = tbl.take(sort_idx)
    else:
        tbl_sorted = tbl

    dense_part = tbl_sorted.slice(0, dense_n) if dense_n else tbl_sorted.slice(0, 0)
    if tail_n <= 0:
        return dense_part

    tail_tbl = tbl_sorted.slice(dense_n)
    if tail_tbl.num_rows <= tail_n:
        return pa.concat_tables([dense_part, tail_tbl], promote=True)

    idx = rng.choice(np.arange(tail_tbl.num_rows), size=tail_n, replace=False)
    tail_part = tail_tbl.take(pa.array(idx, type=pa.int32()))
    return pa.concat_tables([dense_part, tail_part], promote=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Derive calib_prompt_{1k,10k,100k} selected_meta subsets from a prompt_cover_200k_selected_meta.parquet"
    )
    ap.add_argument("--cover_meta", type=str, required=True, help="Parquet produced by prompt-cover LSH selection (200k)")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--sizes", type=str, default="1000,10000,100000", help="Comma-separated pack sizes")
    ap.add_argument(
        "--mix_ratios_json",
        type=str,
        default="",
        help='JSON dict mapping mix_group to ratio, e.g. {"tool":0.4,"reasoning":0.4,"general":0.2}. Empty=default.',
    )
    ap.add_argument("--dense_fraction", type=float, default=0.80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name_prefix", type=str, default="calib_prompt")
    ap.add_argument(
        "--filter_dataset",
        type=str,
        default="",
        help="Optional exact dataset filter (e.g. nvidia/Nemotron-Agentic-v1).",
    )
    ap.add_argument(
        "--prompt_tokens_min",
        type=int,
        default=0,
        help="Optional minimum prompt_tokens filter (inclusive).",
    )
    ap.add_argument(
        "--prompt_tokens_max",
        type=int,
        default=0,
        help="Optional maximum prompt_tokens filter (inclusive). 0 disables.",
    )
    args = ap.parse_args()

    cover_path = Path(args.cover_meta)
    if not cover_path.exists():
        raise SystemExit(f"missing cover_meta: {cover_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = _parse_sizes(args.sizes)
    ratios = _parse_ratios(args.mix_ratios_json)
    dense_fraction = float(args.dense_fraction)
    if dense_fraction < 0.0 or dense_fraction > 1.0:
        raise SystemExit("--dense_fraction must be within [0,1]")

    tbl = pq.read_table(cover_path)
    required = {"id", "dataset", "split", "mix_group"}
    missing = required - set(tbl.column_names)
    if missing:
        raise SystemExit(f"cover_meta missing required columns: {sorted(missing)}")

    if args.filter_dataset:
        tbl = tbl.filter(pc.equal(tbl["dataset"], str(args.filter_dataset)))
    if int(args.prompt_tokens_min) > 0 and "prompt_tokens" in tbl.column_names:
        tbl = tbl.filter(pc.greater_equal(tbl["prompt_tokens"], int(args.prompt_tokens_min)))
    if int(args.prompt_tokens_max) > 0 and "prompt_tokens" in tbl.column_names:
        tbl = tbl.filter(pc.less_equal(tbl["prompt_tokens"], int(args.prompt_tokens_max)))

    if tbl.num_rows == 0:
        raise SystemExit("no rows remain after filters (dataset/prompt_tokens)")

    # Keep deterministic ordering for outputs.
    tbl = tbl.select([c for c in tbl.column_names if c in {"id", "dataset", "split", "mix_group", "prompt_tokens", "bucket_key", "bucket_count", "score"}])

    rng = np.random.default_rng(int(args.seed))
    groups = {mg: tbl.filter(pc.equal(tbl["mix_group"], mg)) for mg in ("tool", "reasoning", "general")}

    manifest = {
        "generated_at": _now(),
        "cover_meta": str(cover_path),
        "rows_in": int(tbl.num_rows),
        "sizes": sizes,
        "mix_ratios": ratios,
        "dense_fraction": dense_fraction,
        "seed": int(args.seed),
        "name_prefix": str(args.name_prefix),
        "outputs": {},
    }

    for size in sizes:
        want = _alloc_counts(int(size), ratios, ["tool", "reasoning", "general"])
        parts: list[pa.Table] = []
        for mg, k in want.items():
            parts.append(_select_within_group(groups.get(mg, groups["general"]), int(k), rng=rng, dense_fraction=dense_fraction))

        sel = pa.concat_tables(parts, promote=True) if len(parts) > 1 else parts[0]
        # Final dedup by id (should already be unique).
        ids = [str(x or "") for x in sel["id"].to_pylist()]
        seen: set[str] = set()
        keep: list[int] = []
        for i, rid in enumerate(ids):
            if not rid or rid in seen:
                continue
            seen.add(rid)
            keep.append(i)
        sel = sel.take(pa.array(keep, type=pa.int32()))

        if sel.num_rows != int(size):
            raise SystemExit(f"selection size mismatch for {size}: got {sel.num_rows}")

        name = f"{args.name_prefix}_{size}"
        out_parquet = out_dir / f"{name}_selected_meta.parquet"
        out_ids = out_dir / f"{name}_ids.txt"
        pq.write_table(sel, out_parquet, compression="zstd")
        out_ids.write_text("\n".join(sel["id"].to_pylist()) + "\n", encoding="utf-8")

        manifest["outputs"][str(size)] = {"selected_meta": str(out_parquet), "ids_txt": str(out_ids), "mix_counts": want}

        print(f"[ok] {name}: rows={sel.num_rows} mix_counts={want}", flush=True)

    (out_dir / "prompt_pack_selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[ok] wrote {out_dir}/prompt_pack_selection_manifest.json", flush=True)


if __name__ == "__main__":
    main()
