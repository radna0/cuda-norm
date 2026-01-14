#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def _quantile_edges(values: np.ndarray, qlo: float, qhi: float) -> tuple[float, float]:
    lo = float(np.quantile(values, qlo))
    hi = float(np.quantile(values, qhi))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo = float(np.min(values))
        hi = float(np.max(values))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _bin_1d(x: np.ndarray, lo: float, hi: float, n: int) -> np.ndarray:
    # Returns int32 bin indices in [0, n-1], clipping out of range.
    x = x.astype(np.float32, copy=False)
    t = (x - lo) / (hi - lo)
    b = np.floor(t * n).astype(np.int32)
    return np.clip(b, 0, n - 1)


def _dense_to_sparse_2d(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.nonzero(arr)
    c = arr[y, x].astype(np.int64)
    return x.astype(np.int32), y.astype(np.int32), c


def _dense_to_sparse_3d(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z, y, x = np.nonzero(arr)
    c = arr[z, y, x].astype(np.int64)
    return x.astype(np.int32), y.astype(np.int32), z.astype(np.int32), c


def main() -> None:
    ap = argparse.ArgumentParser(description="Build coarse density bins from a full PCA parquet.")
    ap.add_argument("--in_parquet", type=str, required=True, help="Full PCA parquet (id + pca_x/y/z + keys)")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--x_col", type=str, default="pca_x")
    ap.add_argument("--y_col", type=str, default="pca_y")
    ap.add_argument("--z_col", type=str, default="pca_z")
    ap.add_argument("--grid_2d", type=int, default=256)
    ap.add_argument("--grid_3d", type=int, default=96)
    ap.add_argument("--qlo", type=float, default=0.005, help="Lower quantile for robust coord range")
    ap.add_argument("--qhi", type=float, default=0.995, help="Upper quantile for robust coord range")
    ap.add_argument("--group_cols", type=str, default="dataset,mix_group", help="Comma-separated columns to group by")
    ap.add_argument("--out_name", type=str, default="density_bins.parquet")
    ap.add_argument("--manifest_name", type=str, default="density_manifest.json")
    args = ap.parse_args()

    in_parquet = Path(args.in_parquet)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_2d = int(args.grid_2d)
    grid_3d = int(args.grid_3d)
    if grid_2d <= 8 or grid_3d <= 8:
        raise SystemExit("--grid_2d and --grid_3d must be reasonably large")

    group_cols = [c.strip() for c in str(args.group_cols).split(",") if c.strip()]
    if not group_cols:
        raise SystemExit("--group_cols must be non-empty")

    pf = pq.ParquetFile(in_parquet)
    cols = set(pf.schema_arrow.names)
    x_col = str(args.x_col)
    y_col = str(args.y_col)
    z_col = str(args.z_col)
    need = {x_col, y_col, z_col}.union(group_cols)
    missing = [c for c in sorted(need) if c not in cols]
    if missing:
        raise SystemExit(f"in_parquet missing columns: {missing}")

    # Load the required columns once (2M rows is manageable; we avoid per-row Python loops this way).
    tbl = pf.read(columns=list(need), use_threads=True)
    total = int(tbl.num_rows)
    print(f"[load] rows={total:,} cols={len(need)}", flush=True)

    x = tbl[x_col].to_numpy(zero_copy_only=False).astype(np.float32)
    y = tbl[y_col].to_numpy(zero_copy_only=False).astype(np.float32)
    z = tbl[z_col].to_numpy(zero_copy_only=False).astype(np.float32)

    xlo, xhi = _quantile_edges(x, float(args.qlo), float(args.qhi))
    ylo, yhi = _quantile_edges(y, float(args.qlo), float(args.qhi))
    zlo, zhi = _quantile_edges(z, float(args.qlo), float(args.qhi))

    bx2 = _bin_1d(x, xlo, xhi, grid_2d)
    by2 = _bin_1d(y, ylo, yhi, grid_2d)
    bz2 = _bin_1d(z, zlo, zhi, grid_2d)

    bx3 = _bin_1d(x, xlo, xhi, grid_3d)
    by3 = _bin_1d(y, ylo, yhi, grid_3d)
    bz3 = _bin_1d(z, zlo, zhi, grid_3d)

    # Build group codes (small cardinality) in a stable way.
    group_arrays: list[list[str]] = []
    for c in group_cols:
        col = tbl[c]
        group_arrays.append([str(v or "unknown") for v in col.to_pylist()])

    # Encode each group col to categorical codes.
    col_uniques: list[list[str]] = []
    col_codes: list[np.ndarray] = []
    for vals in group_arrays:
        uniq = sorted(set(vals))
        inv = {v: i for i, v in enumerate(uniq)}
        codes = np.fromiter((inv[v] for v in vals), dtype=np.int32, count=len(vals))
        col_uniques.append(uniq)
        col_codes.append(codes)

    # Combine codes into a single group_code via mixed radix.
    group_code = np.zeros((total,), dtype=np.int32)
    radix = 1
    for codes, uniq in zip(col_codes, col_uniques):
        group_code += codes * radix
        radix *= len(uniq)
    num_groups = int(radix)
    print(f"[groups] cols={group_cols} groups={num_groups}", flush=True)

    # Decode group_code -> group tuple for record emission.
    group_keys: list[tuple[str, ...]] = []
    for g in range(num_groups):
        rem = g
        parts: list[str] = []
        for uniq in col_uniques:
            parts.append(uniq[rem % len(uniq)])
            rem //= len(uniq)
        group_keys.append(tuple(parts))

    # Vectorized bincount for each density kind.
    def bincount_grouped(idx: np.ndarray, num_bins: int) -> np.ndarray:
        combined = group_code.astype(np.int64) * int(num_bins) + idx.astype(np.int64)
        return np.bincount(combined, minlength=num_groups * num_bins).reshape(num_groups, num_bins)

    lin_xy = (by2.astype(np.int64) * grid_2d + bx2.astype(np.int64)).astype(np.int64)
    lin_xz = (bz2.astype(np.int64) * grid_2d + bx2.astype(np.int64)).astype(np.int64)
    lin_yz = (bz2.astype(np.int64) * grid_2d + by2.astype(np.int64)).astype(np.int64)
    lin_xyz = (bz3.astype(np.int64) * (grid_3d * grid_3d) + by3.astype(np.int64) * grid_3d + bx3.astype(np.int64)).astype(
        np.int64
    )

    counts_xy = bincount_grouped(lin_xy, grid_2d * grid_2d)
    counts_xz = bincount_grouped(lin_xz, grid_2d * grid_2d)
    counts_yz = bincount_grouped(lin_yz, grid_2d * grid_2d)
    counts_xyz = bincount_grouped(lin_xyz, grid_3d * grid_3d * grid_3d)

    # Convert dense hists to sparse long-form records.
    records: list[dict[str, object]] = []

    def emit(kind: str, grid: int, key: tuple[str, ...], bx: np.ndarray, by: np.ndarray, bz: np.ndarray, c: np.ndarray) -> None:
        for j in range(len(c)):
            r: dict[str, object] = {"kind": kind, "grid": int(grid), "count": int(c[j])}
            for col, val in zip(group_cols, key):
                r[col] = val
            r["bx"] = int(bx[j])
            r["by"] = int(by[j])
            r["bz"] = int(bz[j])
            records.append(r)

    for gi, key in enumerate(group_keys):
        arr_xy = counts_xy[gi].reshape(grid_2d, grid_2d).astype(np.int32, copy=False)
        bx, by, c = _dense_to_sparse_2d(arr_xy)
        emit("xy", grid_2d, key, bx, by, np.full_like(bx, -1), c)

        arr_xz = counts_xz[gi].reshape(grid_2d, grid_2d).astype(np.int32, copy=False)
        bx, bz, c = _dense_to_sparse_2d(arr_xz)
        emit("xz", grid_2d, key, bx, np.full_like(bx, -1), bz, c)

        arr_yz = counts_yz[gi].reshape(grid_2d, grid_2d).astype(np.int32, copy=False)
        by, bz, c = _dense_to_sparse_2d(arr_yz)
        emit("yz", grid_2d, key, np.full_like(by, -1), by, bz, c)

        arr_xyz = counts_xyz[gi].reshape(grid_3d, grid_3d, grid_3d).astype(np.int32, copy=False)
        bx, by, bz, c = _dense_to_sparse_3d(arr_xyz)
        emit("xyz", grid_3d, key, bx, by, bz, c)

    out_parquet = out_dir / args.out_name
    pq.write_table(pa.Table.from_pylist(records), out_parquet, compression="zstd")

    manifest = {
        "generated_at": _now(),
        "in_parquet": str(in_parquet),
        "out_parquet": str(out_parquet),
        "rows": int(total),
        "coord_cols": {"x": x_col, "y": y_col, "z": z_col},
        "group_cols": group_cols,
        "groups": int(num_groups),
        "grid_2d": int(grid_2d),
        "grid_3d": int(grid_3d),
        "qlo": float(args.qlo),
        "qhi": float(args.qhi),
        "coord_range": {"x": [xlo, xhi], "y": [ylo, yhi], "z": [zlo, zhi]},
        "records": int(len(records)),
    }
    (out_dir / args.manifest_name).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_parquet}", flush=True)
    print(f"[ok] wrote {out_dir / args.manifest_name}", flush=True)


if __name__ == "__main__":
    main()
