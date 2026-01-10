#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


@dataclass
class PoolWriter:
    out_dir: Path
    pool: str
    rows_per_shard: int
    compression: str

    shard_index: int = 0
    rows_in_shard: int = 0
    total_rows_written: int = 0
    schema: pa.Schema | None = None
    _writer: pq.ParquetWriter | None = None

    def _path_for_shard(self) -> Path:
        name = f"{self.pool}__{self.shard_index:05d}.parquet"
        return self.out_dir / self.pool / name

    def _open_if_needed(self, schema: pa.Schema) -> None:
        if self._writer is not None:
            return
        path = self._path_for_shard()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self._writer = pq.ParquetWriter(str(path), schema, compression=self.compression)

    def _close(self) -> None:
        if self._writer is None:
            return
        self._writer.close()
        self._writer = None
        self.shard_index += 1
        self.rows_in_shard = 0

    def write_table(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        if self.schema is None:
            self._open_if_needed(table.schema)
        assert self.schema is not None
        if table.schema != self.schema:
            table = table.cast(self.schema)

        remaining_table = table
        while remaining_table.num_rows:
            if self._writer is None:
                self._open_if_needed(self.schema)
            assert self._writer is not None
            remaining = self.rows_per_shard - self.rows_in_shard
            if remaining <= 0:
                self._close()
                continue
            if remaining_table.num_rows <= remaining:
                self._writer.write_table(remaining_table)
                n = remaining_table.num_rows
                self.rows_in_shard += n
                self.total_rows_written += n
                if self.rows_in_shard >= self.rows_per_shard:
                    self._close()
                break
            chunk = remaining_table.slice(0, remaining)
            self._writer.write_table(chunk)
            self.rows_in_shard += remaining
            self.total_rows_written += remaining
            self._close()
            remaining_table = remaining_table.slice(remaining)

    def flush(self) -> None:
        # Close any open shard (even if partially filled).
        self._close()


def _fill_null_false(mask: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    return pc.fill_null(mask, False)


def _col(table: pa.Table, name: str) -> pa.ChunkedArray | None:
    if name not in table.column_names:
        return None
    return table[name]


def _bool_mask_all_true(length: int) -> pa.BooleanArray:
    # Only used when we truly need an explicit all-true mask.
    return pa.array([True] * length, type=pa.bool_())


def main() -> None:
    ap = argparse.ArgumentParser(description="Build simple candidate pools from normalized Parquet shards")
    ap.add_argument("--in_dir", type=str, required=True, help="Directory with normalized *.parquet")
    ap.add_argument("--out_dir", type=str, default="cpu_pools", help="Output directory")
    ap.add_argument("--rows_per_shard", type=int, default=100_000)
    ap.add_argument("--compression", type=str, default="zstd")
    ap.add_argument("--correctness_hi", type=float, default=0.875)
    ap.add_argument("--correctness_lo", type=float, default=0.25)
    ap.add_argument(
        "--num_threads",
        type=int,
        default=0,
        help="PyArrow compute threads (0=auto)",
    )
    ap.add_argument(
        "--progress_every",
        type=int,
        default=1_000_000,
        help="Log progress every N input rows (0=off)",
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_threads = int(args.num_threads)
    if num_threads == 0:
        num_threads = os.cpu_count() or 1
    num_threads = max(1, num_threads)
    try:
        pa.set_cpu_count(num_threads)
    except Exception:
        pass

    files = sorted(str(p) for p in in_dir.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files found in {in_dir}")

    dataset = ds.dataset(files, format="parquet")

    summary: dict[str, Any] = {
        "generated_at": _now(),
        "in_dir": str(in_dir),
        "files": len(files),
        "pools": {},
    }

    # Build pool names based on available columns.
    schema_names = set(dataset.schema.names)
    pools: list[str] = []

    if "quality_has_tool" in schema_names:
        pools.append("tool_use")

    if "meta_correctness" in schema_names:
        pools.extend(
            [
                "has_correctness_meta",
                "missing_correctness_meta",
                "high_correctness",
                "low_correctness",
                "mid_correctness",
            ]
        )
    else:
        pools.append("missing_correctness_meta")

    if {"meta_domain", "meta_difficulty_bin", "meta_correctness_high"} <= schema_names:
        pools.append("math_high_verified_correct")

    if "source_reasoning" in schema_names:
        pools.extend(["reasoning_on", "reasoning_off"])

    if "source_capability_target" in schema_names:
        pools.extend(["capability_chat", "capability_structured_outputs"])

    writers: dict[str, PoolWriter] = {
        name: PoolWriter(
            out_dir=out_dir,
            pool=name,
            rows_per_shard=args.rows_per_shard,
            compression=args.compression,
        )
        for name in pools
    }

    base_cols = [
        c
        for c in ["quality_valid_harmony", "quality_completion_nonempty", "quality_valid_tool_schema"]
        if c in schema_names
    ]

    scanner = dataset.scanner(batch_size=8192, use_threads=True)
    total_in = 0
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        table = pa.Table.from_batches([batch])

        base_mask = None
        for c in base_cols:
            col = _col(table, c)
            if col is None:
                continue
            m = _fill_null_false(col)
            base_mask = m if base_mask is None else pc.and_(base_mask, m)
        if base_mask is None:
            base_mask = _bool_mask_all_true(table.num_rows)

        # Pool: tool_use
        if "tool_use" in writers:
            col = _col(table, "quality_has_tool")
            if col is not None:
                mask = pc.and_(base_mask, _fill_null_false(col))
                writers["tool_use"].write_table(table.filter(mask))

        # Pools: meta_correctness based
        if "meta_correctness" in schema_names:
            mc = _col(table, "meta_correctness")
            if mc is not None:
                mask_valid = pc.is_valid(mc)
                mask_null = pc.is_null(mc)
                if "has_correctness_meta" in writers:
                    writers["has_correctness_meta"].write_table(table.filter(pc.and_(base_mask, mask_valid)))
                if "missing_correctness_meta" in writers:
                    writers["missing_correctness_meta"].write_table(table.filter(pc.and_(base_mask, mask_null)))

                if "high_correctness" in writers:
                    m = _fill_null_false(pc.greater_equal(mc, args.correctness_hi))
                    writers["high_correctness"].write_table(table.filter(pc.and_(base_mask, m)))
                if "low_correctness" in writers:
                    m = _fill_null_false(pc.less_equal(mc, args.correctness_lo))
                    writers["low_correctness"].write_table(table.filter(pc.and_(base_mask, m)))
                if "mid_correctness" in writers:
                    m_lo = _fill_null_false(pc.greater(mc, args.correctness_lo))
                    m_hi = _fill_null_false(pc.less(mc, args.correctness_hi))
                    m = pc.and_(m_lo, m_hi)
                    writers["mid_correctness"].write_table(table.filter(pc.and_(base_mask, m)))
        else:
            # No correctness meta column: treat everything as missing meta.
            if "missing_correctness_meta" in writers:
                writers["missing_correctness_meta"].write_table(table.filter(base_mask))

        # Pool: math_high_verified_correct
        if "math_high_verified_correct" in writers:
            md = _col(table, "meta_domain")
            db = _col(table, "meta_difficulty_bin")
            mch = _col(table, "meta_correctness_high")
            if md is not None and db is not None and mch is not None:
                m1 = _fill_null_false(pc.equal(md, "math"))
                m2 = _fill_null_false(pc.equal(db, "high"))
                m3 = _fill_null_false(pc.greater_equal(mch, args.correctness_hi))
                m = pc.and_(pc.and_(m1, m2), m3)
                writers["math_high_verified_correct"].write_table(table.filter(pc.and_(base_mask, m)))

        # Pools: reasoning on/off
        if "reasoning_on" in writers or "reasoning_off" in writers:
            sr = _col(table, "source_reasoning")
            if sr is not None:
                if "reasoning_on" in writers:
                    m = _fill_null_false(pc.equal(sr, "on"))
                    writers["reasoning_on"].write_table(table.filter(pc.and_(base_mask, m)))
                if "reasoning_off" in writers:
                    m = _fill_null_false(pc.equal(sr, "off"))
                    writers["reasoning_off"].write_table(table.filter(pc.and_(base_mask, m)))

        # Pools: capability targets
        if "capability_chat" in writers or "capability_structured_outputs" in writers:
            ct = _col(table, "source_capability_target")
            if ct is not None:
                if "capability_chat" in writers:
                    m = _fill_null_false(pc.equal(ct, "chat"))
                    writers["capability_chat"].write_table(table.filter(pc.and_(base_mask, m)))
                if "capability_structured_outputs" in writers:
                    m = _fill_null_false(pc.equal(ct, "structured_outputs"))
                    writers["capability_structured_outputs"].write_table(table.filter(pc.and_(base_mask, m)))

        total_in += table.num_rows
        if args.progress_every and total_in % int(args.progress_every) < table.num_rows:
            print(f"[prog] rows_scanned={total_in:,}", flush=True)

    for name, writer in writers.items():
        writer.flush()
        summary["pools"][name] = {"rows": writer.total_rows_written, "shards": writer.shard_index}
        print(f"[pool] {name}: rows={writer.total_rows_written} shards={writer.shard_index}", flush=True)

    out_path = out_dir / "pools_manifest.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
