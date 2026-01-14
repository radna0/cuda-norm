#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def _parse_bucket_edges(spec: str) -> list[int]:
    parts = [p.strip() for p in (spec or "").split(",") if p.strip()]
    if not parts or parts[0] != "0" or parts[-1].lower() != "inf":
        raise SystemExit("--len_bucket_edges must look like: 0,64,128,256,512,1024,inf")
    edges: list[int] = []
    for p in parts[:-1]:
        try:
            edges.append(int(p))
        except Exception as e:
            raise SystemExit(f"bad --len_bucket_edges part {p!r}: {e}") from e
    if edges != sorted(edges):
        raise SystemExit("--len_bucket_edges must be sorted ascending")
    if edges[0] != 0:
        raise SystemExit("--len_bucket_edges must start with 0")
    return edges


def _len_bucket_label(lo: int, hi: int | None) -> str:
    if hi is None:
        return f"{lo:04d}_inf"
    return f"{lo:04d}_{hi:04d}"


def _assign_len_bucket_vec(tok: np.ndarray, edges: list[int]) -> list[str]:
    # tok: int array shape (n,)
    t = tok.astype(np.int64, copy=False)
    t = np.maximum(t, 0)
    # edges are inclusive lower bounds; last is inf.
    # Use np.searchsorted to find right bucket.
    # Example edges [0,64,128,...]
    idx = np.searchsorted(np.asarray(edges[1:], dtype=np.int64), t, side="right")
    # idx in [0..len(edges)-1]
    labels: list[str] = []
    for i in idx.tolist():
        lo = int(edges[i])
        hi = int(edges[i + 1]) if i + 1 < len(edges) else None
        labels.append(_len_bucket_label(lo, hi))
    return labels


def _canonical_difficulty_vec(vals: list[Any]) -> list[str]:
    out: list[str] = []
    for v in vals:
        s = str(v or "").strip().lower()
        if s in {"low", "medium", "high"}:
            out.append(s)
        else:
            out.append("unknown")
    return out


def _embedding_array_to_numpy(arr: pa.Array) -> tuple[np.ndarray, int]:
    if not pa.types.is_fixed_size_list(arr.type):
        raise TypeError(f"expected FixedSizeListArray embedding, got {arr.type}")
    dim = int(arr.type.list_size)
    values = arr.values.to_numpy(zero_copy_only=False)
    if values.size % dim != 0:
        raise RuntimeError(f"embedding values size {values.size} not divisible by dim {dim}")
    return values.reshape(-1, dim), dim


def _iter_parquet_files(in_dir: Path) -> list[Path]:
    files = sorted(in_dir.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {in_dir}")
    return files


def _iter_sample_rows(
    parquet_files: list[Path],
    *,
    sample_per_file: int,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    emb_list: list[np.ndarray] = []
    n = 0
    for pf in parquet_files:
        if n >= max_samples:
            break
        parquet = pq.ParquetFile(pf)
        cols = set(parquet.schema_arrow.names)
        if "embedding" not in cols:
            continue
        taken = 0
        for batch in parquet.iter_batches(columns=["embedding"], batch_size=65_536):
            if taken >= sample_per_file or n >= max_samples:
                break
            tbl = pa.Table.from_batches([batch])
            if tbl.num_rows == 0:
                continue
            need = min(sample_per_file - taken, max_samples - n, int(tbl.num_rows))
            if need <= 0:
                break
            if need < tbl.num_rows:
                idx = rng.choice(np.arange(tbl.num_rows), size=need, replace=False)
                tbl = tbl.take(pa.array(idx, type=pa.int32()))
            else:
                tbl = tbl.slice(0, need)
            emb_np, _ = _embedding_array_to_numpy(tbl["embedding"].chunk(0))
            emb_list.append(emb_np.astype(np.float32, copy=False))
            taken += int(tbl.num_rows)
            n += int(tbl.num_rows)
        if taken:
            print(f"[train-sample] {pf} took={taken}", flush=True)
    if not emb_list:
        raise SystemExit("no embedding rows found to train PCA")
    X = np.vstack(emb_list)
    if X.shape[0] > max_samples:
        X = X[:max_samples]
    return X


def _write_batches(
    *,
    out_parquet: Path,
    schema: pa.Schema,
    batches: Iterable[pa.RecordBatch],
) -> None:
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(out_parquet, schema=schema, compression="zstd")
    try:
        for b in batches:
            writer.write_batch(b)
    finally:
        writer.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Project an embedding dataset to PCA (2D/3D) for all rows.")
    ap.add_argument("--in_dir", type=str, required=True, help="Directory containing embedding *.parquet")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--pca_dims", type=int, default=3, help="2 or 3")
    ap.add_argument("--num_threads", type=int, default=0, help="0=auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_sample_per_file", type=int, default=25_000)
    ap.add_argument("--train_max_samples", type=int, default=200_000)
    ap.add_argument("--batch_size", type=int, default=65_536)
    ap.add_argument(
        "--len_bucket_edges",
        type=str,
        default="0,64,128,256,512,1024,inf",
        help="Token bucket edges for coloring/analysis.",
    )
    ap.add_argument(
        "--out_name",
        type=str,
        default="pca_3d_full.parquet",
        help="Output parquet filename within --out_dir.",
    )
    ap.add_argument("--manifest_name", type=str, default="pca_3d_full_manifest.json")
    args = ap.parse_args()

    pca_dims = int(args.pca_dims)
    if pca_dims not in {2, 3}:
        raise SystemExit("--pca_dims must be 2 or 3")

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    len_edges = _parse_bucket_edges(args.len_bucket_edges)

    parquet_files = _iter_parquet_files(in_dir)
    train_sample_per_file = int(args.train_sample_per_file)
    train_max_samples = int(args.train_max_samples)

    try:
        import faiss  # type: ignore
    except Exception as e:
        raise SystemExit(f"faiss not available: {e}") from e

    num_threads = int(args.num_threads) if args.num_threads else (os.cpu_count() or 1)
    faiss.omp_set_num_threads(int(num_threads))

    print(f"[cfg] in_dir={in_dir}", flush=True)
    print(f"[cfg] out_dir={out_dir}", flush=True)
    print(f"[cfg] pca_dims={pca_dims} threads={num_threads}", flush=True)
    print(f"[cfg] train_sample_per_file={train_sample_per_file} train_max_samples={train_max_samples}", flush=True)

    t0 = time.time()
    X_train = _iter_sample_rows(
        parquet_files,
        sample_per_file=train_sample_per_file,
        max_samples=train_max_samples,
        seed=int(args.seed),
    )
    dim = int(X_train.shape[1])

    pca = faiss.PCAMatrix(dim, pca_dims)
    t_train0 = time.time()
    pca.train(X_train)
    train_elapsed = time.time() - t_train0

    ev = getattr(pca, "eigenvalues", None)
    if ev is not None:
        ev_arr = faiss.vector_to_array(ev)
        ev_list = [float(x) for x in ev_arr.tolist()]
    else:
        ev_list = []

    out_parquet = out_dir / args.out_name
    manifest_path = out_dir / args.manifest_name

    want_cols = [
        "id",
        "dataset",
        "split",
        "mix_group",
        "meta_domain",
        "meta_difficulty_bin",
        "meta_correctness",
        "prompt_tokens",
    ]

    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise SystemExit("--batch_size must be > 0")

    total_rows = 0
    xyz_min = np.full((pca_dims,), np.inf, dtype=np.float64)
    xyz_max = np.full((pca_dims,), -np.inf, dtype=np.float64)

    schema: pa.Schema | None = None

    def batches() -> Iterable[pa.RecordBatch]:
        nonlocal total_rows, schema, xyz_min, xyz_max
        for pf in parquet_files:
            parquet = pq.ParquetFile(pf)
            cols = set(parquet.schema_arrow.names)
            if "embedding" not in cols or "id" not in cols:
                continue
            read_cols = [c for c in want_cols if c in cols] + ["embedding"]
            for batch in parquet.iter_batches(columns=read_cols, batch_size=batch_size):
                tbl = pa.Table.from_batches([batch])
                if tbl.num_rows == 0:
                    continue
                emb_np, _ = _embedding_array_to_numpy(tbl["embedding"].chunk(0))
                emb_np = emb_np.astype(np.float32, copy=False)
                Y = pca.apply_py(emb_np)
                # Track global min/max (for downstream density binning).
                xyz_min = np.minimum(xyz_min, np.min(Y, axis=0))
                xyz_max = np.maximum(xyz_max, np.max(Y, axis=0))

                # Derived columns.
                mdiff = tbl["meta_difficulty_bin"].to_pylist() if "meta_difficulty_bin" in tbl.column_names else []
                difficulty = _canonical_difficulty_vec(mdiff) if mdiff else ["unknown"] * tbl.num_rows
                ptok = (
                    np.asarray(tbl["prompt_tokens"].to_numpy(zero_copy_only=False), dtype=np.int64)
                    if "prompt_tokens" in tbl.column_names
                    else np.zeros((tbl.num_rows,), dtype=np.int64)
                )
                len_bucket = _assign_len_bucket_vec(ptok, len_edges)

                arrays: dict[str, pa.Array] = {}
                for c in want_cols:
                    if c in tbl.column_names:
                        arrays[c] = tbl[c].combine_chunks()
                arrays["difficulty_bin"] = pa.array(difficulty, type=pa.string())
                arrays["len_bucket"] = pa.array(len_bucket, type=pa.string())
                arrays["pca_x"] = pa.array(Y[:, 0].astype(np.float32))
                arrays["pca_y"] = pa.array(Y[:, 1].astype(np.float32))
                if pca_dims >= 3:
                    arrays["pca_z"] = pa.array(Y[:, 2].astype(np.float32))

                out_tbl = pa.table(arrays)
                if schema is None:
                    schema = out_tbl.schema
                total_rows += int(out_tbl.num_rows)
                if total_rows % 250_000 < out_tbl.num_rows:
                    print(f"[prog] rows={total_rows:,}", flush=True)
                yield out_tbl.to_batches(max_chunksize=out_tbl.num_rows)[0]

    # Materialize schema by consuming first batch.
    it = iter(batches())
    first = next(it, None)
    if first is None:
        raise SystemExit("no rows to project (no embedding+id columns found)")
    assert schema is not None

    def chain_first() -> Iterable[pa.RecordBatch]:
        yield first
        for b in it:
            yield b

    _write_batches(out_parquet=out_parquet, schema=schema, batches=chain_first())
    proj_elapsed = time.time() - t0

    manifest = {
        "generated_at": _now(),
        "in_dir": str(in_dir),
        "out_parquet": str(out_parquet),
        "rows": int(total_rows),
        "embedding_dim": int(dim),
        "pca_dims": int(pca_dims),
        "train_sample_per_file": int(train_sample_per_file),
        "train_max_samples": int(train_max_samples),
        "faiss_threads": int(num_threads),
        "train_elapsed_s": float(train_elapsed),
        "project_elapsed_s": float(proj_elapsed),
        "eigenvalues": ev_list[: min(16, len(ev_list))],
        "eigenvalues_sum": float(sum(ev_list)) if ev_list else math.nan,
        "coord_min": [float(x) for x in xyz_min.tolist()],
        "coord_max": [float(x) for x in xyz_max.tolist()],
        "len_bucket_edges": [0] + list(len_edges[1:]) + ["inf"],
        "batch_size": int(batch_size),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_parquet}", flush=True)
    print(f"[ok] wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()

