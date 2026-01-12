#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract ids + mix_group mapping from a candidate Parquet dataset")
    ap.add_argument("--dataset_id", type=str, required=True, help="HF dataset repo_id (candidates)")
    ap.add_argument("--subdir", type=str, default="", help="Optional subdir within the dataset")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--max_rows", type=int, default=0, help="Debug cap (0=all)")
    ap.add_argument("--batch_rows", type=int, default=65536)
    ap.add_argument("--write_counts", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    subdir = (args.subdir or "").strip() or None
    allow = ["**/*.parquet", "bucket_manifest.json", "README.md"]
    if subdir:
        allow = [f"{subdir}/**/*.parquet", f"{subdir}/*.parquet", "bucket_manifest.json", "README.md"]

    t0 = time.time()
    print(f"[*] snapshot_download: {args.dataset_id} subdir={subdir!r}", flush=True)
    snap = snapshot_download(repo_id=args.dataset_id, repo_type="dataset", allow_patterns=allow)
    snap_path = Path(snap)
    scan_root = (snap_path / subdir) if subdir else snap_path

    parquet_files = sorted(scan_root.rglob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"no parquet files under {scan_root}")
    print(f"[ok] downloaded {len(parquet_files)} parquet files dt={time.time()-t0:.1f}s", flush=True)

    dataset = ds.dataset([str(p) for p in parquet_files], format="parquet")
    cols = set(dataset.schema.names)
    need = {"id", "mix_group"}
    if not need.issubset(cols):
        raise SystemExit(f"dataset missing required columns: {sorted(need - cols)}")

    read_cols = ["id", "mix_group"]
    if "dataset" in cols:
        read_cols.append("dataset")
    if "split" in cols:
        read_cols.append("split")

    ids_path = out_dir / "ids.txt"
    map_path = out_dir / "id_mix_group.parquet"
    meta_path = out_dir / "extract_manifest.json"

    counts: dict[str, int] = {}
    row_count = 0

    writer: pq.ParquetWriter | None = None
    try:
        scanner = dataset.scanner(columns=read_cols, batch_size=int(args.batch_rows), use_threads=True)
        with ids_path.open("w", encoding="utf-8") as f_ids:
            for batch in scanner.to_batches():
                if batch.num_rows == 0:
                    continue
                if args.max_rows and row_count >= args.max_rows:
                    break
                tbl = pa.Table.from_batches([batch])
                if args.max_rows and (row_count + tbl.num_rows) > args.max_rows:
                    tbl = tbl.slice(0, args.max_rows - row_count)
                row_count += tbl.num_rows

                # ids.txt
                for s in tbl["id"].to_pylist():
                    sid = str(s or "")
                    if sid:
                        f_ids.write(sid + "\n")

                if args.write_counts and "dataset" in tbl.column_names and "split" in tbl.column_names:
                    ds_vals = tbl["dataset"].to_pylist()
                    sp_vals = tbl["split"].to_pylist()
                    for d, sp in zip(ds_vals, sp_vals):
                        key = f"{d}:{sp}"
                        counts[key] = counts.get(key, 0) + 1

                if writer is None:
                    writer = pq.ParquetWriter(str(map_path), tbl.schema, compression="zstd")
                writer.write_table(tbl)
                if row_count % 500_000 == 0:
                    print(f"[prog] rows={row_count}", flush=True)
    finally:
        if writer is not None:
            writer.close()

    manifest = {
        "generated_at": _now(),
        "dataset_id": args.dataset_id,
        "subdir": subdir or "",
        "snap_path": str(snap_path),
        "scan_root": str(scan_root),
        "parquet_files": len(parquet_files),
        "rows": int(row_count),
        "ids_path": str(ids_path),
        "map_path": str(map_path),
        "counts": counts if args.write_counts else None,
        "elapsed_s": time.time() - t0,
    }
    meta_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ok] wrote {ids_path} and {map_path}", flush=True)
    print(f"[ok] wrote {meta_path}", flush=True)


if __name__ == "__main__":
    main()

