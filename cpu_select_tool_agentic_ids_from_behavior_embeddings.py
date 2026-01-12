#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def _uint64_weights(bits: int) -> np.ndarray:
    if bits <= 0 or bits > 63:
        raise ValueError("bits must be in [1,63]")
    return (np.uint64(1) << np.arange(bits, dtype=np.uint64)).astype(np.uint64)


def _embedding_array_to_numpy(arr: pa.Array) -> tuple[np.ndarray, int]:
    if not pa.types.is_fixed_size_list(arr.type):
        raise TypeError(f"expected FixedSizeListArray embedding, got {arr.type}")
    d = int(arr.type.list_size)
    values = arr.values.to_numpy(zero_copy_only=False)
    if values.size % d != 0:
        raise RuntimeError(f"embedding values size {values.size} not divisible by dim {d}")
    return values.reshape(-1, d), d


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Select diverse tool/agentic IDs from behavior embeddings using LSH buckets"
    )
    ap.add_argument("--in_dir", type=str, required=True, help="Directory containing embedding *.parquet")
    ap.add_argument(
        "--target_k",
        type=int,
        default=10_000,
        help="How many samples to select",
    )
    ap.add_argument(
        "--mix_group",
        type=str,
        default="tool",
        help="Filter rows by mix_group (default: tool)",
    )
    ap.add_argument(
        "--bits",
        type=int,
        default=24,
        help="LSH bits (random hyperplanes). 16–32 recommended. Must be <=63.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--dense_fraction",
        type=float,
        default=0.80,
        help="Fraction of selections from largest buckets (rest sampled from tail)",
    )
    ap.add_argument(
        "--score_field",
        type=str,
        default="stats_embed_word_count",
        help="Pick the max-scoring row within a bucket by this numeric field",
    )
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--max_rows", type=int, default=0, help="Debug cap (0=all)")
    ap.add_argument("--batch_size", type=int, default=8192)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(in_dir.rglob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"no parquet files under {in_dir}")

    dataset = ds.dataset([str(p) for p in parquet_files], format="parquet")
    cols = set(dataset.schema.names)
    need = {"id", "embedding", "mix_group"}
    missing = need - cols
    if missing:
        raise SystemExit(f"dataset missing required columns: {sorted(missing)}")

    score_field = args.score_field
    if score_field not in cols:
        raise SystemExit(f"missing score field {score_field!r} in embeddings dataset")
    for c in ("dataset", "split"):
        if c not in cols:
            raise SystemExit(f"missing required passthrough column {c!r} in embeddings dataset")

    filter_expr = ds.field("mix_group") == args.mix_group
    read_cols = ["id", "embedding", "mix_group", "dataset", "split", score_field]
    if "prompt_tokens" in cols:
        read_cols.append("prompt_tokens")

    rng = np.random.default_rng(int(args.seed))
    weights = _uint64_weights(int(args.bits))
    proj_mat: np.ndarray | None = None
    dim: int | None = None

    # key -> [count, best_score, best_id, best_dataset, best_split, best_prompt_tokens]
    agg: dict[int, list[Any]] = {}
    total = 0
    kept = 0
    t0 = time.time()

    scanner = dataset.scanner(columns=read_cols, filter=filter_expr, batch_size=int(args.batch_size), use_threads=True)
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        if args.max_rows and total >= args.max_rows:
            break

        tbl = pa.Table.from_batches([batch])
        if args.max_rows and (total + tbl.num_rows) > args.max_rows:
            tbl = tbl.slice(0, args.max_rows - total)

        ids = [str(x or "") for x in tbl["id"].to_pylist()]
        ds_vals = [str(x or "") for x in tbl["dataset"].to_pylist()]
        sp_vals = [str(x or "") for x in tbl["split"].to_pylist()]
        scores = tbl[score_field].to_pylist()
        ptoks = tbl["prompt_tokens"].to_pylist() if "prompt_tokens" in tbl.column_names else None

        emb_vals, d = _embedding_array_to_numpy(tbl["embedding"].chunk(0))
        if dim is None:
            dim = d
        elif dim != d:
            raise RuntimeError(f"embedding dim changed from {dim} to {d}")

        if proj_mat is None:
            proj_mat = rng.standard_normal(size=(d, int(args.bits))).astype(np.float32)

        proj = emb_vals.astype(np.float32) @ proj_mat
        bits = proj > 0
        keys = (bits.astype(np.uint64) * weights).sum(axis=1).astype(np.uint64)

        for i, k in enumerate(keys.tolist()):
            rid = ids[i]
            if not rid:
                continue
            score = scores[i]
            try:
                score_f = float(score) if score is not None else math.nan
            except Exception:
                score_f = math.nan
            if math.isnan(score_f):
                score_f = -1.0
            prompt_tokens_i = int(ptoks[i] or 0) if ptoks is not None else 0

            rec = agg.get(k)
            if rec is None:
                agg[k] = [1, score_f, rid, ds_vals[i], sp_vals[i], prompt_tokens_i]
                continue

            rec[0] += 1
            if score_f > float(rec[1]):
                rec[1] = score_f
                rec[2] = rid
                rec[3] = ds_vals[i]
                rec[4] = sp_vals[i]
                rec[5] = prompt_tokens_i

        total += tbl.num_rows

    if not agg:
        raise SystemExit("no rows after filtering; check --mix_group and input data")

    # Select buckets
    items = [(k, int(v[0])) for k, v in agg.items()]
    items.sort(key=lambda kv: kv[1], reverse=True)
    unique_buckets = len(items)

    target_k = int(args.target_k)
    if target_k <= 0:
        raise SystemExit("--target_k must be > 0")
    target_k = min(target_k, unique_buckets)

    dense_n = int(round(target_k * float(args.dense_fraction)))
    dense_n = max(0, min(dense_n, target_k))
    tail_n = target_k - dense_n

    dense_keys = [k for k, _ in items[:dense_n]]
    tail_pool = [k for k, _ in items[dense_n:]]
    if tail_n and tail_pool:
        tail_keys = rng.choice(np.array(tail_pool, dtype=np.uint64), size=tail_n, replace=False).tolist()
    else:
        tail_keys = []

    selected_keys = dense_keys + tail_keys
    if len(selected_keys) != target_k:
        raise RuntimeError(f"selected {len(selected_keys)} keys, expected {target_k}")

    # Emit selected rows (best per bucket).
    out_rows: list[dict[str, Any]] = []
    ids_out: list[str] = []
    for k in selected_keys:
        rec = agg[int(k)]
        out_rows.append(
            {
                "bucket_key": int(k),
                "bucket_count": int(rec[0]),
                "score": float(rec[1]),
                "id": str(rec[2]),
                "dataset": str(rec[3]),
                "split": str(rec[4]),
                "prompt_tokens": int(rec[5]),
                "mix_group": str(args.mix_group),
            }
        )
        ids_out.append(str(rec[2]))

    # Dedup (should already be unique).
    ids_unique = list(dict.fromkeys([x for x in ids_out if x]))
    if len(ids_unique) != len(ids_out):
        raise RuntimeError(f"duplicate ids detected: {len(ids_out)} -> {len(ids_unique)}")

    out_ids_path = out_dir / f"{args.mix_group}_agentic_ids_{len(ids_unique)}.txt"
    out_meta_path = out_dir / f"{args.mix_group}_agentic_selected_{len(ids_unique)}.parquet"
    out_report_path = out_dir / f"{args.mix_group}_agentic_selection_report.md"
    out_manifest_path = out_dir / "selection_manifest.json"

    out_ids_path.write_text("\n".join(ids_unique) + "\n", encoding="utf-8")
    pq.write_table(pa.Table.from_pylist(out_rows), out_meta_path, compression="zstd")

    ds_counts = Counter([r["dataset"] for r in out_rows])
    top_ds = "\n".join([f"- {k}: `{v}`" for k, v in ds_counts.most_common(25)])
    report = [
        "# Tool/Agentic Selection Report",
        f"- generated_at: `{_now()}`",
        f"- in_dir: `{in_dir}`",
        f"- parquet_files: `{len(parquet_files)}`",
        f"- mix_group: `{args.mix_group}`",
        f"- rows_scanned: `{total}`",
        f"- unique_buckets: `{unique_buckets}`",
        f"- target_k: `{target_k}`",
        f"- dense_fraction: `{float(args.dense_fraction):.3f}`",
        f"- bits: `{int(args.bits)}`",
        f"- seed: `{int(args.seed)}`",
        f"- score_field: `{score_field}`",
        "",
        "## Top Datasets (selected)",
        top_ds or "- (none)",
        "",
    ]
    out_report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    manifest = {
        "generated_at": _now(),
        "in_dir": str(in_dir),
        "parquet_files": len(parquet_files),
        "mix_group": args.mix_group,
        "rows_scanned": int(total),
        "unique_buckets": int(unique_buckets),
        "target_k": int(target_k),
        "dense_fraction": float(args.dense_fraction),
        "bits": int(args.bits),
        "seed": int(args.seed),
        "score_field": score_field,
        "out_ids_path": str(out_ids_path),
        "out_meta_path": str(out_meta_path),
        "out_report_path": str(out_report_path),
        "elapsed_s": time.time() - t0,
    }
    out_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_ids_path}", flush=True)
    print(f"[ok] wrote {out_meta_path}", flush=True)
    print(f"[ok] wrote {out_report_path}", flush=True)
    print(f"[ok] wrote {out_manifest_path}", flush=True)


if __name__ == "__main__":
    main()

