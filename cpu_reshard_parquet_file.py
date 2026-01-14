#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser(description="Reshard a Parquet file into smaller Parquet files (row-count based).")
    ap.add_argument("--in_file", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--rows_per_shard", type=int, default=25_000)
    ap.add_argument("--compression", type=str, default="zstd")
    ap.add_argument("--prefix", type=str, default="part")
    args = ap.parse_args()

    in_file = Path(args.in_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_per_shard = int(args.rows_per_shard)
    if rows_per_shard <= 0:
        raise SystemExit("--rows_per_shard must be > 0")

    pf = pq.ParquetFile(in_file)
    total_rows = pf.metadata.num_rows
    est_shards = max(1, int(math.ceil(total_rows / rows_per_shard)))
    print(f"[*] {in_file} rows={total_rows} rows_per_shard={rows_per_shard} est_shards={est_shards}", flush=True)

    shard_idx = 0
    shard_rows = 0
    writer: pq.ParquetWriter | None = None

    def _open_writer(example_table: pa.Table) -> pq.ParquetWriter:
        nonlocal shard_idx
        out_path = out_dir / f"{args.prefix}-{shard_idx:05d}.parquet"
        print(f"[*] open {out_path}", flush=True)
        shard_idx += 1
        return pq.ParquetWriter(out_path, example_table.schema, compression=args.compression)

    try:
        for batch in pf.iter_batches(batch_size=min(rows_per_shard, 131_072)):
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = _open_writer(table)
                shard_rows = 0
            writer.write_table(table)
            shard_rows += table.num_rows
            if shard_rows >= rows_per_shard:
                writer.close()
                writer = None
    finally:
        if writer is not None:
            writer.close()

    out_files = sorted(out_dir.glob(f"{args.prefix}-*.parquet"))
    out_rows = 0
    for p in out_files:
        out_rows += pq.ParquetFile(p).metadata.num_rows
    print(f"[ok] wrote {len(out_files)} shards rows={out_rows}", flush=True)


if __name__ == "__main__":
    main()

