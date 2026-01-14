#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _quantiles(x: np.ndarray, qs: list[float]) -> dict[str, float]:
    x = np.asarray(x)
    if x.size == 0:
        return {str(q): float("nan") for q in qs}
    out = np.quantile(x.astype(np.float64, copy=False), qs)
    return {str(q): float(v) for q, v in zip(qs, out.tolist(), strict=True)}


def _iter_parquet_files(in_dir: Path) -> list[Path]:
    files = sorted(in_dir.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {in_dir}")
    return files


def _embedding_array_to_numpy(arr: pa.Array) -> tuple[np.ndarray, int]:
    if not pa.types.is_fixed_size_list(arr.type):
        raise TypeError(f"expected FixedSizeListArray embedding, got {arr.type}")
    dim = int(arr.type.list_size)
    values = arr.values.to_numpy(zero_copy_only=False)
    if values.size % dim != 0:
        raise RuntimeError(f"embedding values size {values.size} not divisible by dim {dim}")
    return values.reshape(-1, dim), dim


def _table_value_counts(tbl: pa.Table, col: str) -> dict[str, int]:
    if col not in tbl.column_names:
        return {}
    arr = tbl[col]
    if arr.null_count:
        arr = pc.fill_null(arr, "unknown")
    vc = pc.value_counts(arr)
    out: dict[str, int] = {}

    # Arrow versions differ: value_counts may return a StructArray or a Table-like object.
    values_arr: pa.Array
    counts_arr: pa.Array
    if isinstance(vc, pa.Array) and pa.types.is_struct(vc.type):
        # StructArray with fields (values, counts).
        field_names = [f.name for f in vc.type]
        if "values" in field_names:
            values_arr = vc.field("values")
        else:
            values_arr = vc.field(0)
        if "counts" in field_names:
            counts_arr = vc.field("counts")
        else:
            counts_arr = vc.field(1)
    else:
        # Assume Table / RecordBatch-like.
        values_arr = vc.column(0)  # type: ignore[attr-defined]
        counts_arr = vc.column(1)  # type: ignore[attr-defined]

    for v, c in zip(values_arr.to_pylist(), counts_arr.to_pylist(), strict=True):
        out[str(v)] = int(c)
    return out


def _parse_umap_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    parsed: dict[str, Any] = {}

    in_table = False
    for ln in lines:
        s = ln.strip()
        if s.startswith("| k | overlap"):
            in_table = True
            continue
        if not in_table:
            continue
        if not s.startswith("|"):
            break
        parts = [p.strip() for p in s.strip("|").split("|")]
        if len(parts) < 7:
            continue
        try:
            k = int(parts[0])
        except Exception:
            continue
        parsed[str(k)] = {
            "overlap": float(parts[1]),
            "purity_dataset": float(parts[2]),
            "purity_domain": float(parts[3]),
            "purity_mix_group": float(parts[4]),
            "purity_difficulty": float(parts[5]),
            "purity_len_bucket": float(parts[6]),
        }
    return parsed


def _gini(counts: np.ndarray) -> float:
    x = np.asarray(counts, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    x = np.maximum(x, 0.0)
    s = float(x.sum())
    if s <= 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((np.sum((2 * idx - n - 1) * x)) / (n * s))


def _effective_clusters(counts: np.ndarray) -> float:
    x = np.asarray(counts, dtype=np.float64)
    x = np.maximum(x, 0.0)
    s = float(x.sum())
    if s <= 0:
        return 0.0
    p = x / s
    p = p[p > 0]
    h = -float(np.sum(p * np.log(p)))
    return float(np.exp(h))


def _kmeans_cluster_metrics(*, X: np.ndarray, k: int, seed: int, niter: int, threads: int) -> dict[str, Any]:
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise SystemExit(f"faiss not available: {e}") from e

    X = np.asarray(X, dtype=np.float32)
    n, dim = X.shape
    if n <= 1 or k <= 1:
        return {}

    faiss.omp_set_num_threads(int(threads))
    km = faiss.Kmeans(dim, int(k), niter=int(niter), verbose=False, seed=int(seed), spherical=True)
    km.train(X)
    D, I = km.index.search(X, 1)
    assign = I.reshape(-1).astype(np.int64, copy=False)
    counts = np.bincount(assign, minlength=int(k)).astype(np.int64, copy=False)
    counts_sorted = np.sort(counts)[::-1]

    eff = _effective_clusters(counts.astype(np.float64))
    g = _gini(counts.astype(np.float64))
    top1 = float(counts_sorted[0] / max(1, counts_sorted.sum()))
    top10 = float(counts_sorted[:10].sum() / max(1, counts_sorted.sum())) if counts_sorted.size >= 10 else top1
    nonempty = int((counts > 0).sum())

    return {
        "k": int(k),
        "n": int(n),
        "niter": int(niter),
        "seed": int(seed),
        "nonempty_clusters": nonempty,
        "gini": float(g),
        "effective_clusters_exp_entropy": float(eff),
        "top1_frac": float(top1),
        "top10_frac": float(top10),
        "counts_quantiles": _quantiles(counts.astype(np.float64), [0.0, 0.5, 0.9, 0.99, 1.0]),
    }


def _density_xyz_group_stats(
    *, density_bins_parquet: Path, density_manifest: dict[str, Any]
) -> dict[str, Any]:
    group_cols = density_manifest.get("group_cols") or ["dataset", "mix_group"]
    if not isinstance(group_cols, list) or not group_cols:
        group_cols = ["dataset", "mix_group"]
    group_cols = [str(c) for c in group_cols if str(c).strip()]

    schema = pq.ParquetFile(density_bins_parquet).schema_arrow
    available = set(schema.names)
    group_cols = [c for c in group_cols if c in available]
    if not group_cols:
        return {"group_cols": [], "groups": [], "totals": {"error": "density_bins has no recognized group columns"}}

    tbl = pq.read_table(
        density_bins_parquet,
        columns=["kind", "count", *group_cols],
        use_threads=True,
    )
    tbl = tbl.filter(pc.equal(tbl["kind"], "xyz"))
    if tbl.num_rows == 0:
        return {"group_cols": group_cols, "groups": [], "totals": {}}

    g = tbl.group_by(group_cols).aggregate([("count", "sum"), ("count", "count")])
    g = g.rename_columns([*group_cols, "points", "voxels"])

    grid3 = int(density_manifest.get("grid_3d", 0) or 0)
    max_vox = grid3**3 if grid3 else None

    points_sum = 0
    voxels_sum = 0
    groups: list[dict[str, Any]] = []
    for row in g.to_pylist():
        points = int(row["points"])
        voxels = int(row["voxels"])
        points_sum += points
        voxels_sum += voxels
        rec: dict[str, Any] = {c: str(row.get(c)) for c in group_cols}
        rec.update(
            {
                "points": points,
                "voxels": voxels,
                "avg_pts_per_voxel": float(points / voxels) if voxels else float("nan"),
                "occupancy": float(voxels / max_vox) if max_vox else None,
            }
        )
        groups.append(rec)
    groups.sort(key=lambda r: r["points"], reverse=True)

    totals: dict[str, Any] = {
        "grid3": grid3,
        "max_voxels": max_vox,
        "points_sum": points_sum,
        "voxels_sum": voxels_sum,
        "mean_occupancy": float(voxels_sum / (max_vox * len(groups))) if (max_vox and groups) else None,
    }
    return {"group_cols": group_cols, "groups": groups, "totals": totals}


def _sample_embeddings(
    *,
    parquet_files: list[Path],
    sample_size: int,
    seed: int,
    want_cols: list[str],
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    files = list(parquet_files)
    per_file = int(math.ceil(sample_size / max(1, len(files))))

    X_list: list[np.ndarray] = []
    cols_data: dict[str, list[Any]] = {c: [] for c in want_cols if c != "embedding"}
    total = 0
    dim: int | None = None

    for pf in files:
        parquet = pq.ParquetFile(pf)
        schema_cols = set(parquet.schema_arrow.names)
        cols = [c for c in want_cols if c in schema_cols]
        if "embedding" not in cols:
            continue

        n = int(parquet.metadata.num_rows) if parquet.metadata else 0
        if n <= 0:
            continue

        take_n = min(per_file, n)
        idx = rng.choice(n, size=take_n, replace=False).astype(np.int64, copy=False)
        idx_arr = pa.array(idx, type=pa.int64())
        tbl = parquet.read(columns=cols).take(idx_arr)

        emb_np, d = _embedding_array_to_numpy(tbl["embedding"].combine_chunks())
        dim = d if dim is None else dim
        if dim != d:
            raise RuntimeError(f"inconsistent embedding dims: saw {dim} then {d} in {pf}")
        X_list.append(emb_np.astype(np.float32, copy=False))

        for c in cols_data:
            if c not in tbl.column_names:
                cols_data[c].extend([None] * int(tbl.num_rows))
            else:
                cols_data[c].extend(tbl[c].combine_chunks().to_pylist())

        total += int(tbl.num_rows)
        print(f"[sample] {pf} rows={tbl.num_rows} total={total}", flush=True)
        if total >= sample_size:
            break

    if not X_list or dim is None:
        raise SystemExit("failed to sample any embeddings")

    X = np.vstack(X_list)
    if X.shape[0] > sample_size:
        X = X[:sample_size]
        for c in cols_data:
            cols_data[c] = cols_data[c][:sample_size]

    return {"X": X, "dim": int(dim), "cols": cols_data}


def _knn_metrics(
    *,
    X: np.ndarray,
    ids: list[str] | None,
    labels: dict[str, list[Any]],
    k: int,
    threads: int,
) -> dict[str, Any]:
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise SystemExit(f"faiss not available: {e}") from e

    X = np.asarray(X, dtype=np.float32)
    n, dim = X.shape
    if n <= 1:
        return {}

    faiss.omp_set_num_threads(int(threads))

    norms = np.linalg.norm(X, axis=1).astype(np.float32, copy=False)
    norm_stats = {
        "mean": float(norms.mean()),
        "std": float(norms.std()),
        "quantiles": _quantiles(norms, [0.0, 0.01, 0.5, 0.9, 0.99, 1.0]),
    }

    # Normalize and search in L2 space. For unit vectors, cosine = 1 - (L2^2)/2.
    Xn = X / np.maximum(norms[:, None], 1e-12)

    M = 32
    index = faiss.IndexHNSWFlat(dim, M)  # L2
    index.hnsw.efConstruction = 80
    index.hnsw.efSearch = 80
    index.add(Xn)

    D, I = index.search(Xn, int(k) + 1)
    # D is squared L2 distance.
    cos = 1.0 - 0.5 * D

    nn1 = cos[:, 1].astype(np.float32, copy=False)
    nnk = cos[:, -1].astype(np.float32, copy=False)
    nn_mean = cos[:, 1:].mean(axis=1).astype(np.float32, copy=False)

    dup_thresholds = [0.99, 0.995, 0.999, 0.9995]
    dup_rates = {str(t): float((nn1 >= t).mean()) for t in dup_thresholds}

    purities: dict[str, float] = {}
    for name, vals in labels.items():
        arr = np.asarray([str(v) for v in vals], dtype=object)
        if arr.shape[0] != n:
            continue
        _, codes = np.unique(arr, return_inverse=True)
        codes = codes.astype(np.int32, copy=False)
        neigh = codes[I[:, 1:]]  # (n,k)
        same = (neigh == codes[:, None]).mean(axis=1)
        purities[name] = float(same.mean())

    out_n = max(10, int(0.001 * n))
    out_idx = np.argsort(nn1)[:out_n]

    outliers: list[dict[str, Any]] = []
    if ids is not None and len(ids) == n:
        for i in out_idx[:200]:
            outliers.append({"id": ids[int(i)], "nn1_cos": float(nn1[int(i)])})

    out_label_dist: dict[str, Any] = {}
    for name, vals in labels.items():
        arr = [str(vals[int(i)]) for i in out_idx]
        out_label_dist[name] = dict(Counter(arr).most_common(10))

    return {
        "n": int(n),
        "k": int(k),
        "norms": norm_stats,
        "nn1_cos": {"quantiles": _quantiles(nn1, [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0]), "dup_rates": dup_rates},
        "nnk_cos": {"quantiles": _quantiles(nnk, [0.0, 0.5, 0.9, 0.99, 1.0])},
        "nn_mean_cos": {"quantiles": _quantiles(nn_mean, [0.0, 0.5, 0.9, 0.99, 1.0])},
        "label_purity_mean": purities,
        "outliers": outliers,
        "outlier_label_top10": out_label_dist,
    }


def _render_md(*, title: str, metrics: dict[str, Any]) -> str:
    pca = metrics.get("pca") or {}
    dens = metrics.get("density_xyz") or {}
    dens_by = metrics.get("density_xyz_by") or {}
    umap = metrics.get("umap") or {}
    knn = metrics.get("knn") or {}
    kmeans = metrics.get("kmeans") or {}
    counts = metrics.get("full_counts") or {}

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- generated_at: {metrics.get('generated_at')}")
    lines.append(f"- embedding_dir: {metrics.get('embedding_dir')}")
    lines.append("")

    lines.append("## PCA (full)")
    lines.append("")
    lines.append(f"- rows: {pca.get('rows')}")
    lines.append(f"- explained_var_pc1: {pca.get('explained_var_pc1')}")
    lines.append(f"- explained_var_pc1_3: {pca.get('explained_var_pc1_3')}")
    lines.append("")

    lines.append("## Density (PCA space, xyz voxels)")
    lines.append("")
    totals = dens.get("totals") or {}
    lines.append(f"- grid3: {totals.get('grid3')} (max_voxels={totals.get('max_voxels')})")
    lines.append(f"- mean_occupancy: {totals.get('mean_occupancy')}")
    lines.append("")
    group_cols = dens.get("group_cols") or ["dataset", "mix_group"]
    group_cols = [str(c) for c in group_cols if str(c).strip()]
    header_cols = [*group_cols, "points", "voxels", "avg_pts/voxel", "occupancy"]
    align = ["---"] * len(group_cols) + ["---:"] * 4
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "|".join(align) + "|")
    for g in (dens.get("groups") or [])[:10]:
        row_vals: list[Any] = [g.get(c, "") for c in group_cols]
        row_vals.extend(
            [
                int(g.get("points") or 0),
                int(g.get("voxels") or 0),
                f"{float(g.get('avg_pts_per_voxel') or 0.0):.2f}",
                g.get("occupancy"),
            ]
        )
        lines.append("| " + " | ".join(str(v) for v in row_vals) + " |")
    lines.append("")

    if dens_by:
        lines.append("**Density breakdowns (xyz occupancy)**")
        lines.append("")
        for name, d in dens_by.items():
            tot = (d or {}).get("totals") or {}
            lines.append(f"- {name}: mean_occupancy={tot.get('mean_occupancy')}")
        lines.append("")

    if umap:
        lines.append("## UMAP local-structure report (200k fit, 50k eval)")
        lines.append("")
        lines.append("| k | overlap | purity(dataset) | purity(domain) | purity(mix_group) | purity(difficulty) | purity(len_bucket) |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for k in sorted(int(x) for x in umap.keys()):
            r = umap[str(k)]
            lines.append(
                f"| {k} | {r['overlap']:.4f} | {r['purity_dataset']:.4f} | {r['purity_domain']:.4f} | {r['purity_mix_group']:.4f} | {r['purity_difficulty']:.4f} | {r['purity_len_bucket']:.4f} |"
            )
        lines.append("")

    if knn:
        lines.append("## kNN redundancy + purity (embedding space, sampled)")
        lines.append("")
        lines.append(f"- n(sample): {knn.get('n')} k={knn.get('k')}")
        lines.append(f"- nn1 cosine quantiles: {knn.get('nn1_cos', {}).get('quantiles')}")
        lines.append(f"- nn1 near-dup rates: {knn.get('nn1_cos', {}).get('dup_rates')}")
        lines.append("")
        lines.append("Mean neighbor-label purity (kNN@k, exclude self):")
        for k, v in sorted((knn.get("label_purity_mean") or {}).items()):
            lines.append(f"- {k}: {v:.4f}")
        lines.append("")

    if kmeans:
        lines.append("## Cluster KPIs (embedding space, sampled)")
        lines.append("")
        for name, km in sorted(kmeans.items()):
            lines.append(f"**k={name}**")
            lines.append(f"- effective_clusters(exp_entropy): {km.get('effective_clusters_exp_entropy')}")
            lines.append(f"- gini: {km.get('gini')}")
            lines.append(f"- top1_frac: {km.get('top1_frac')} top10_frac: {km.get('top10_frac')}")
            lines.append("")

    lines.append("## Full 2M label counts (top10)")
    lines.append("")
    for col, dist in counts.items():
        items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
        top = ", ".join([f"{k}={v}" for k, v in items[:10]])
        lines.append(f"- {col}: {top}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute quantitative embedding-manifold diagnostics (kNN redundancy/purity + PCA/density/UMAP summaries)."
    )
    ap.add_argument("--title", type=str, required=True)
    ap.add_argument("--embedding_dir", type=str, required=True)
    ap.add_argument("--pca_manifest", type=str, required=True)
    ap.add_argument("--pca_parquet", type=str, required=True)
    ap.add_argument("--density_manifest", type=str, required=True)
    ap.add_argument("--density_bins", type=str, required=True)
    ap.add_argument("--umap_report", type=str, default="")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--sample_size", type=int, default=200_000)
    ap.add_argument("--knn_k", type=int, default=10)
    ap.add_argument("--kmeans_ks", type=str, default="1000,5000", help="Comma-separated k values for k-means KPIs (sampled)")
    ap.add_argument("--kmeans_niter", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="0=auto")
    args = ap.parse_args()

    threads = int(args.threads) if int(args.threads) > 0 else (os.cpu_count() or 1)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pca_manifest = _read_json(Path(args.pca_manifest))
    density_manifest = _read_json(Path(args.density_manifest))

    eigenvalues = [float(x) for x in pca_manifest.get("eigenvalues", [])]
    ev_sum = float(pca_manifest.get("eigenvalues_sum") or 0.0)
    ev_pc1 = eigenvalues[0] if eigenvalues else 0.0
    ev_pc1_3 = sum(eigenvalues[:3]) if eigenvalues else 0.0
    pca_summary = {
        "rows": int(pca_manifest.get("rows") or 0),
        "embedding_dim": int(pca_manifest.get("embedding_dim") or 0),
        "eigenvalues_sum": ev_sum,
        "explained_var_pc1": float(ev_pc1 / ev_sum) if ev_sum else float("nan"),
        "explained_var_pc1_3": float(ev_pc1_3 / ev_sum) if ev_sum else float("nan"),
    }

    density_stats = _density_xyz_group_stats(
        density_bins_parquet=Path(args.density_bins),
        density_manifest=density_manifest,
    )
    density_xyz_by: dict[str, Any] = {}
    # Extra groupings for KPI deltas (cheap: operates on density_bins parquet only).
    for grp in (["meta_domain"], ["mix_group"], ["dataset"]):
        density_xyz_by["_".join(grp)] = _density_xyz_group_stats(
            density_bins_parquet=Path(args.density_bins),
            density_manifest={**density_manifest, "group_cols": grp},
        )

    umap = _parse_umap_report(Path(args.umap_report)) if args.umap_report else {}

    pca_tbl = pq.read_table(
        Path(args.pca_parquet),
        columns=["dataset", "mix_group", "meta_domain", "difficulty_bin", "len_bucket"],
        use_threads=True,
    )
    full_counts = {
        "dataset": _table_value_counts(pca_tbl, "dataset"),
        "mix_group": _table_value_counts(pca_tbl, "mix_group"),
        "meta_domain": _table_value_counts(pca_tbl, "meta_domain"),
        "difficulty_bin": _table_value_counts(pca_tbl, "difficulty_bin"),
        "len_bucket": _table_value_counts(pca_tbl, "len_bucket"),
    }

    parquet_files = _iter_parquet_files(Path(args.embedding_dir))
    want_cols = [
        "embedding",
        "id",
        "dataset",
        "mix_group",
        "meta_domain",
        "meta_difficulty_bin",
        "prompt_tokens",
    ]
    sample = _sample_embeddings(
        parquet_files=parquet_files,
        sample_size=int(args.sample_size),
        seed=int(args.seed),
        want_cols=want_cols,
    )
    cols = sample["cols"]
    ids = [str(x) for x in cols.get("id", [])]
    label_cols = {
        "dataset": cols.get("dataset", []),
        "mix_group": cols.get("mix_group", []),
        "meta_domain": cols.get("meta_domain", []),
        "meta_difficulty_bin": cols.get("meta_difficulty_bin", []),
    }
    knn = _knn_metrics(
        X=sample["X"],
        ids=ids if len(ids) == sample["X"].shape[0] else None,
        labels=label_cols,
        k=int(args.knn_k),
        threads=threads,
    )

    kmeans: dict[str, Any] = {}
    ks: list[int] = []
    for part in str(args.kmeans_ks).split(","):
        s = part.strip()
        if not s:
            continue
        ks.append(int(s))
    ks = [k for k in sorted(set(ks)) if k > 1]
    for k in ks:
        kmeans[str(k)] = _kmeans_cluster_metrics(
            X=sample["X"],
            k=int(k),
            seed=int(args.seed),
            niter=int(args.kmeans_niter),
            threads=threads,
        )

    metrics: dict[str, Any] = {
        "generated_at": _now(),
        "title": str(args.title),
        "embedding_dir": str(Path(args.embedding_dir)),
        "pca": pca_summary,
        "density_xyz": density_stats,
        "density_xyz_by": density_xyz_by,
        "umap": umap,
        "knn": knn,
        "kmeans": kmeans,
        "full_counts": full_counts,
    }

    (out_dir / "diagnostics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "diagnostics.md").write_text(
        _render_md(title=str(args.title), metrics=metrics),
        encoding="utf-8",
    )
    print(f"[ok] wrote {out_dir}/diagnostics.md and diagnostics.json", flush=True)


if __name__ == "__main__":
    main()
