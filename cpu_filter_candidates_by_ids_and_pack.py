#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def _parse_edges(spec: str) -> list[int]:
    edges: list[int] = []
    for part in (spec or "").split(","):
        s = part.strip()
        if not s:
            continue
        if s.lower() == "inf":
            continue
        edges.append(int(s))
    if not edges or edges[0] != 0:
        raise SystemExit("--bucket_edges must start with 0, e.g. 0,256,512,1024,2048,inf")
    if edges != sorted(edges):
        raise SystemExit("--bucket_edges must be sorted ascending")
    return edges


def _bucket_label(lo: int, hi: int | None) -> str:
    if hi is None:
        return f"len_{lo:05d}_inf"
    return f"len_{lo:05d}_{hi:05d}"


@dataclass
class ShardWriter:
    base_dir: Path
    bucket: str
    rows_per_shard: int
    compression: str = "zstd"
    shard_index: int = 0
    rows_in_shard: int = 0
    schema: pa.Schema | None = None
    _writer: pq.ParquetWriter | None = None

    def _path(self) -> Path:
        return self.base_dir / self.bucket / f"part-{self.shard_index:05d}.parquet"

    def _open(self, schema: pa.Schema) -> None:
        out_path = self._path()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self._writer = pq.ParquetWriter(str(out_path), schema, compression=self.compression)

    def _close(self) -> None:
        if self._writer is None:
            return
        path = self._path()
        self._writer.close()
        self._writer = None
        print(f"[write] {path} rows={self.rows_in_shard}", flush=True)
        self.shard_index += 1
        self.rows_in_shard = 0

    def write(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        if self._writer is None:
            self._open(table.schema)
        assert self.schema is not None
        if table.schema != self.schema:
            table = table.cast(self.schema)
        remaining = table
        while remaining.num_rows:
            if self._writer is None:
                self._open(self.schema)
            assert self._writer is not None
            cap = self.rows_per_shard - self.rows_in_shard
            if cap <= 0:
                self._close()
                continue
            if remaining.num_rows <= cap:
                self._writer.write_table(remaining)
                self.rows_in_shard += remaining.num_rows
                if self.rows_in_shard >= self.rows_per_shard:
                    self._close()
                break
            head = remaining.slice(0, cap)
            self._writer.write_table(head)
            self.rows_in_shard += head.num_rows
            self._close()
            remaining = remaining.slice(cap)

    def flush(self) -> None:
        self._close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Filter a candidates dataset by an ids.txt file and pack into length-bucketed large Parquet shards."
    )
    ap.add_argument("--in_dir", type=str, required=True)
    ap.add_argument("--ids_txt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--rows_per_shard", type=int, default=200_000)
    ap.add_argument("--bucket_edges", type=str, default="0,512,1024,1536,2048,inf")
    ap.add_argument("--batch_size", type=int, default=131_072)
    ap.add_argument("--manifest", type=str, default="packed_manifest.json")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids_path = Path(args.ids_txt)
    if not ids_path.exists():
        raise SystemExit(f"missing ids_txt: {ids_path}")
    ids = [ln.strip() for ln in ids_path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    if not ids:
        raise SystemExit("ids_txt is empty")
    ids_set = set(ids)
    ids_arr = pa.array(sorted(ids_set), type=pa.string())

    files = sorted(in_dir.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {in_dir}")
    dataset = ds.dataset([str(p) for p in files], format="parquet")
    cols = set(dataset.schema.names)
    for c in ("id", "stats_tok_len"):
        if c not in cols:
            raise SystemExit(f"dataset missing required column {c!r}")

    edges_raw: list[str] = [p.strip() for p in (args.bucket_edges or "").split(",") if p.strip()]
    if not edges_raw or edges_raw[0] != "0" or edges_raw[-1].lower() != "inf":
        raise SystemExit("--bucket_edges must look like: 0,512,1024,2048,inf")
    edges = _parse_edges(",".join(edges_raw))
    bucket_pairs: list[tuple[int, int | None]] = []
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else None
        bucket_pairs.append((lo, hi))

    writers: dict[str, ShardWriter] = {}
    counts: dict[str, int] = {}

    total_in = 0
    total_out = 0
    seen_out: set[str] = set()
    t0 = time.time()

    read_cols = list(dataset.schema.names)
    scanner = dataset.scanner(columns=read_cols, batch_size=int(args.batch_size), use_threads=True)
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        table = pa.Table.from_batches([batch])
        total_in += int(table.num_rows)

        m = pc.is_in(table["id"], value_set=ids_arr)
        m = pc.fill_null(m, False)
        table = table.filter(m)
        if table.num_rows == 0:
            continue

        # Dedup by id across the whole stream (keeps first occurrence).
        ids_py = [str(x or "") for x in table["id"].to_pylist()]
        keep_mask: list[bool] = []
        for rid in ids_py:
            if not rid:
                keep_mask.append(False)
                continue
            if rid in seen_out:
                keep_mask.append(False)
                continue
            seen_out.add(rid)
            keep_mask.append(True)
        table = table.filter(pa.array(keep_mask))
        if table.num_rows == 0:
            continue

        tok_len = pc.fill_null(table["stats_tok_len"].cast(pa.int32()), 0)
        for lo, hi in bucket_pairs:
            label = _bucket_label(lo, hi)
            if hi is None:
                mm = pc.greater_equal(tok_len, lo)
            else:
                mm = pc.and_(pc.greater_equal(tok_len, lo), pc.less(tok_len, hi))
            mm = pc.fill_null(mm, False)
            sub = table.filter(mm)
            if sub.num_rows == 0:
                continue
            w = writers.get(label)
            if w is None:
                w = ShardWriter(base_dir=out_dir, bucket=label, rows_per_shard=int(args.rows_per_shard))
                writers[label] = w
                counts[label] = 0
            w.write(sub)
            counts[label] += int(sub.num_rows)
            total_out += int(sub.num_rows)

        if total_in and (total_in % 1_000_000 == 0):
            dt = time.time() - t0
            print(f"[prog] rows_in={total_in} rows_out={total_out} dt={dt:.1f}s", flush=True)

    for w in writers.values():
        w.flush()

    dt = time.time() - t0
    manifest = {
        "generated_at": _now(),
        "in_dir": str(in_dir),
        "ids_txt": str(ids_path),
        "out_dir": str(out_dir),
        "bucket_edges": edges_raw,
        "rows_per_shard": int(args.rows_per_shard),
        "totals": {"rows_in_scanned": int(total_in), "rows_out": int(total_out), "elapsed_s": float(dt)},
        "buckets": {k: int(v) for k, v in sorted(counts.items())},
    }
    (out_dir / args.manifest).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if total_out != len(ids_set):
        raise SystemExit(f"filtered rows_out={total_out} != ids={len(ids_set)} (missing ids?)")
    print(f"[ok] wrote {out_dir}/{args.manifest}", flush=True)


if __name__ == "__main__":
    main()
