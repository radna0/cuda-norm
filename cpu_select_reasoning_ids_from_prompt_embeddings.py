#!/usr/bin/env python3
"""Select a difficulty/length-stratified reasoning pack from prompt-view embeddings.

This is used to rebuild `reasoning_10k` from the 2M prompt-view embedding run.
"""
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
    dim = int(arr.type.list_size)
    values = arr.values.to_numpy(zero_copy_only=False)
    if values.size % dim != 0:
        raise RuntimeError(f"embedding values size {values.size} not divisible by dim {dim}")
    return values.reshape(-1, dim), dim


def _canonical_difficulty(x: Any) -> str:
    s = str(x or "").strip().lower()
    if s in {"low", "medium", "high"}:
        return s
    return "unknown"


def _parse_quantiles(spec: str) -> list[float]:
    out: list[float] = []
    for part in (spec or "").split(","):
        s = part.strip()
        if not s:
            continue
        out.append(float(s))
    if not out or out[0] != 0.0 or out[-1] != 1.0:
        raise SystemExit("--len_quantiles must start with 0 and end with 1 (e.g. 0,0.2,0.4,0.6,0.8,1)")
    if out != sorted(out):
        raise SystemExit("--len_quantiles must be sorted ascending")
    if any(q < 0.0 or q > 1.0 for q in out):
        raise SystemExit("--len_quantiles must be within [0,1]")
    return out


def _quantile_edges(sample: np.ndarray, quantiles: list[float]) -> list[int]:
    if sample.size == 0:
        raise SystemExit("no prompt_tokens available for quantile computation")
    qs = np.quantile(sample.astype(np.float32), quantiles)
    edges = [int(round(float(x))) for x in qs]
    if edges[0] != int(sample.min()):
        edges[0] = int(sample.min())
    if edges[-1] != int(sample.max()):
        edges[-1] = int(sample.max())
    # Convert quantile points to bucket edges: start at 0, internal cuts, end at max+1.
    cuts = [int(x) for x in edges[1:-1]]
    max_tok = int(edges[-1])
    out = [0] + cuts + [max_tok + 1]
    # Ensure strictly increasing edges.
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + 1
    if out[-1] <= out[-2]:
        out[-1] = out[-2] + 1
    return out


def _len_bucket(tok: int, edges: list[int]) -> int:
    # edges define half-open buckets [edges[i], edges[i+1]) for i in [0..n-2]
    t = int(tok)
    if t < 0:
        t = 0
    # edges are small (<= ~1025); linear scan is fine.
    for i in range(len(edges) - 1):
        if edges[i] <= t < edges[i + 1]:
            return i
    return len(edges) - 2


def _bucket_label(lo: int, hi: int) -> str:
    return f"{lo:04d}_{hi:04d}"


def _parse_difficulty_weights(spec: str) -> dict[str, float]:
    if not spec.strip():
        return {"high": 0.40, "medium": 0.30, "low": 0.30, "unknown": 0.00}
    try:
        obj = json.loads(spec)
    except Exception as e:
        raise SystemExit(f"--difficulty_weights_json must be valid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise SystemExit("--difficulty_weights_json must be a JSON object")
    out: dict[str, float] = {}
    for k, v in obj.items():
        kk = _canonical_difficulty(k)
        try:
            out[kk] = float(v)
        except Exception as e:
            raise SystemExit(f"bad weight for {k!r}: {e}") from e
    for k in ("high", "medium", "low", "unknown"):
        out.setdefault(k, 0.0)
    return out


def _round_robin_fill(quota: dict[int, int], capacity: dict[int, int], *, target_total: int) -> None:
    # Increase quota up to capacity until target_total is reached (deterministic).
    while sum(quota.values()) < target_total:
        progressed = False
        # Prefer strata with the most remaining capacity.
        remaining = sorted(
            capacity.items(), key=lambda kv: (-(kv[1] - quota.get(kv[0], 0)), kv[0])
        )
        for sid, cap in remaining:
            q = quota.get(sid, 0)
            if q >= cap:
                continue
            quota[sid] = q + 1
            progressed = True
            if sum(quota.values()) >= target_total:
                return
        if not progressed:
            return


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Redo reasoning_10k selection from prompt-view embeddings (difficulty+length stratified, LSH-diverse)"
    )
    ap.add_argument("--in_dir", type=str, required=True, help="Directory containing embedding *.parquet")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--target_k", type=int, default=10_000)
    ap.add_argument("--mix_group", type=str, default="reasoning")
    ap.add_argument("--bits", type=int, default=24, help="LSH bits (<=63). 20–32 recommended.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--dense_fraction",
        type=float,
        default=0.80,
        help="Within each stratum, fraction of selections from largest buckets (rest from tail).",
    )
    ap.add_argument(
        "--len_quantiles",
        type=str,
        default="0,0.2,0.4,0.6,0.8,1",
        help="Quantiles used to build prompt_tokens buckets (must start 0 and end 1).",
    )
    ap.add_argument("--tok_sample_cap", type=int, default=200_000)
    ap.add_argument(
        "--difficulty_weights_json",
        type=str,
        default="",
        help='JSON dict, e.g. {"high":0.4,"medium":0.3,"low":0.3,"unknown":0.0}. Empty=default.',
    )
    ap.add_argument(
        "--score_field",
        type=str,
        default="prompt_tokens",
        help="Field to maximize per LSH bucket (prompt_tokens|stats_embed_word_count|meta_correctness).",
    )
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--max_rows", type=int, default=0, help="Debug cap (0=all)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(in_dir.rglob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"no parquet files under {in_dir}")

    dataset = ds.dataset([str(p) for p in parquet_files], format="parquet")
    cols = set(dataset.schema.names)
    required = {"id", "embedding", "mix_group", "dataset", "split", "prompt_tokens"}
    missing = required - cols
    if missing:
        raise SystemExit(f"dataset missing required columns: {sorted(missing)}")

    score_field = args.score_field
    if score_field not in cols:
        raise SystemExit(f"missing score field {score_field!r} in embeddings dataset")

    bits = int(args.bits)
    if bits <= 0 or bits > 63:
        raise SystemExit("--bits must be in [1,63]")
    bucket_mask = (1 << bits) - 1

    rng = np.random.default_rng(int(args.seed))
    rng_sample = np.random.default_rng(int(args.seed) + 12345)

    # Pass 1: build prompt_tokens length buckets from a sample (fast path: no embeddings read).
    tok_sample: list[int] = []
    tok_sample_cap = int(args.tok_sample_cap)
    diff_counts = Counter()

    filter_expr = ds.field("mix_group") == args.mix_group
    read_cols_fast = ["prompt_tokens", "mix_group"]
    if "meta_difficulty_bin" in cols:
        read_cols_fast.append("meta_difficulty_bin")
    scanner = dataset.scanner(columns=read_cols_fast, filter=filter_expr, batch_size=65_536, use_threads=True)
    rows_seen = 0
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        table = pa.Table.from_batches([batch])
        if args.max_rows and rows_seen >= args.max_rows:
            break
        if args.max_rows and (rows_seen + table.num_rows) > args.max_rows:
            table = table.slice(0, args.max_rows - rows_seen)

        pt = table["prompt_tokens"].combine_chunks().to_numpy(zero_copy_only=False)
        pt = np.asarray(pt, dtype=np.int32)
        remaining = tok_sample_cap - len(tok_sample)
        if remaining > 0 and pt.size > 0:
            take = min(int(remaining), int(pt.size))
            # Random sample without replacement to avoid bias from file order.
            idx = rng_sample.choice(np.arange(int(pt.size)), size=take, replace=False)
            tok_sample.extend([int(x) for x in pt[idx].tolist()])
        if "meta_difficulty_bin" in table.column_names:
            diffs = table["meta_difficulty_bin"].to_pylist()
            diff_counts.update(_canonical_difficulty(x) for x in diffs)
        else:
            diff_counts.update(["unknown"] * int(table.num_rows))

        rows_seen += int(table.num_rows)
        if len(tok_sample) >= tok_sample_cap:
            break

    tok_arr = np.array(tok_sample, dtype=np.int32)
    quantiles = _parse_quantiles(args.len_quantiles)
    edges = _quantile_edges(tok_arr, quantiles)
    n_len_buckets = len(edges) - 1

    print(f"[*] length buckets (prompt_tokens) edges={edges}", flush=True)
    print(f"[*] difficulty counts (sampled): {dict(diff_counts)}", flush=True)

    # Pass 2: scan embeddings and build LSH buckets per (difficulty, len_bucket) stratum.
    diff_order = ["high", "medium", "low", "unknown"]
    diff_to_idx = {d: i for i, d in enumerate(diff_order)}
    n_strata = len(diff_order) * n_len_buckets

    weights = _uint64_weights(bits)
    proj_mat: np.ndarray | None = None
    dim: int | None = None

    # composite_key -> [count, best_score, best_id, best_dataset, best_split, best_prompt_tokens, best_diff, best_len_bucket]
    agg: dict[int, list[Any]] = {}
    stratum_row_counts = np.zeros(n_strata, dtype=np.int64)

    read_cols = ["id", "embedding", "dataset", "split", "mix_group", "prompt_tokens"]
    if score_field not in read_cols:
        read_cols.append(score_field)
    if "meta_difficulty_bin" in cols:
        read_cols.append("meta_difficulty_bin")

    t0 = time.time()
    total_rows = 0
    processed_rows = 0
    scanner = dataset.scanner(columns=read_cols, filter=filter_expr, batch_size=max(16_384, int(args.batch_size) * 4), use_threads=True)
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        table = pa.Table.from_batches([batch])
        if args.max_rows and processed_rows >= args.max_rows:
            break
        if args.max_rows and (processed_rows + table.num_rows) > args.max_rows:
            table = table.slice(0, args.max_rows - processed_rows)

        ids = [str(x or "") for x in table["id"].to_pylist()]
        ds_vals = [str(x or "") for x in table["dataset"].to_pylist()]
        sp_vals = [str(x or "") for x in table["split"].to_pylist()]
        ptoks = table["prompt_tokens"].combine_chunks().to_numpy(zero_copy_only=False)
        ptoks = np.asarray(ptoks, dtype=np.int32)

        if "meta_difficulty_bin" in table.column_names:
            diffs_raw = table["meta_difficulty_bin"].to_pylist()
            diffs = [_canonical_difficulty(x) for x in diffs_raw]
        else:
            diffs = ["unknown"] * len(ids)

        diff_idx = np.array([diff_to_idx[d] for d in diffs], dtype=np.int32)
        len_bucket = np.searchsorted(np.array(edges, dtype=np.int32), ptoks, side="right") - 1
        len_bucket = np.clip(len_bucket, 0, n_len_buckets - 1).astype(np.int32)
        stratum_id = diff_idx * n_len_buckets + len_bucket

        # Record row counts per stratum.
        stratum_row_counts += np.bincount(stratum_id, minlength=n_strata).astype(np.int64)

        # Scores.
        if score_field == "prompt_tokens":
            scores = ptoks.astype(np.float32)
        else:
            score_col = table[score_field].combine_chunks()
            if pa.types.is_floating(score_col.type) or pa.types.is_integer(score_col.type):
                scores = np.asarray(score_col.to_numpy(zero_copy_only=False), dtype=np.float32)
            else:
                # Fallback: treat non-numeric as 0
                scores = np.zeros(len(ids), dtype=np.float32)
            scores = np.nan_to_num(scores, nan=-1.0, neginf=-1.0, posinf=-1.0)

        emb_vals, d = _embedding_array_to_numpy(table["embedding"].chunk(0))
        if dim is None:
            dim = d
        elif dim != d:
            raise RuntimeError(f"embedding dim changed from {dim} to {d}")
        if proj_mat is None:
            proj_mat = rng.standard_normal(size=(d, bits)).astype(np.float32)

        proj = emb_vals.astype(np.float32) @ proj_mat
        bitmask = proj > 0
        keys = (bitmask.astype(np.uint64) * weights).sum(axis=1).astype(np.uint64)
        comp = (stratum_id.astype(np.uint64) << np.uint64(bits)) | keys

        order = np.argsort(comp)
        comp_s = comp[order]
        scores_s = scores[order]

        change = np.ones(comp_s.shape[0], dtype=bool)
        if comp_s.shape[0] > 1:
            change[1:] = comp_s[1:] != comp_s[:-1]
        starts = np.flatnonzero(change)
        ends = np.concatenate([starts[1:], np.array([comp_s.shape[0]], dtype=np.int64)])

        for s, e in zip(starts.tolist(), ends.tolist()):
            k = int(comp_s[s])
            count = int(e - s)
            best_rel = int(np.argmax(scores_s[s:e])) + s
            best_i = int(order[best_rel])
            rid = ids[best_i]
            if not rid:
                continue
            best_score = float(scores[best_i])
            best_prompt_tokens = int(ptoks[best_i])
            best_diff = diffs[best_i]
            best_len_bucket = int(len_bucket[best_i])

            rec = agg.get(k)
            if rec is None:
                agg[k] = [
                    count,
                    best_score,
                    rid,
                    ds_vals[best_i],
                    sp_vals[best_i],
                    best_prompt_tokens,
                    best_diff,
                    best_len_bucket,
                ]
            else:
                rec[0] += count
                if best_score > float(rec[1]):
                    rec[1] = best_score
                    rec[2] = rid
                    rec[3] = ds_vals[best_i]
                    rec[4] = sp_vals[best_i]
                    rec[5] = best_prompt_tokens
                    rec[6] = best_diff
                    rec[7] = best_len_bucket

        processed_rows += int(table.num_rows)
        total_rows += int(table.num_rows)
        if processed_rows and (processed_rows % 500_000 == 0):
            dt = time.time() - t0
            print(
                f"[prog] rows={processed_rows} unique_buckets={len(agg)} dt={dt:.1f}s",
                flush=True,
            )

    if not agg:
        raise SystemExit("no rows after filtering; check --mix_group and input data")

    # Group buckets by stratum_id.
    stratum_items: dict[int, list[tuple[int, list[Any]]]] = {i: [] for i in range(n_strata)}
    for comp_key, rec in agg.items():
        sid = int(np.uint64(comp_key) >> np.uint64(bits))
        stratum_items[sid].append((comp_key, rec))

    capacity = {sid: len(items) for sid, items in stratum_items.items() if items}
    total_capacity = sum(capacity.values())
    target_k = int(args.target_k)
    if target_k <= 0:
        raise SystemExit("--target_k must be > 0")
    if total_capacity < target_k:
        print(
            f"[warn] only {total_capacity} unique buckets available; capping target_k to {total_capacity}",
            flush=True,
        )
        target_k = total_capacity

    # Compute per-difficulty quota targets.
    diff_weights = _parse_difficulty_weights(args.difficulty_weights_json)
    diff_cap = {d: 0 for d in diff_order}
    for sid, cap in capacity.items():
        d = diff_order[sid // n_len_buckets]
        diff_cap[d] += cap

    active_diffs = [d for d in diff_order if diff_cap[d] > 0 and diff_weights.get(d, 0.0) > 0.0]
    if not active_diffs:
        active_diffs = [d for d in diff_order if diff_cap[d] > 0]
        for d in diff_order:
            diff_weights[d] = 1.0 if d in active_diffs else 0.0

    weight_sum = sum(diff_weights[d] for d in active_diffs)
    diff_quota = {d: 0 for d in diff_order}
    for d in active_diffs:
        diff_quota[d] = int(round(target_k * (diff_weights[d] / weight_sum)))

    # Fix rounding to match target_k (deterministic).
    while sum(diff_quota.values()) < target_k:
        d = max(active_diffs, key=lambda x: diff_cap[x] - diff_quota[x])
        diff_quota[d] += 1
    while sum(diff_quota.values()) > target_k:
        d = max(active_diffs, key=lambda x: diff_quota[x])
        if diff_quota[d] <= 0:
            break
        diff_quota[d] -= 1

    # Allocate per stratum within each difficulty by capacity, then globally fill any slack.
    quota: dict[int, int] = {sid: 0 for sid in range(n_strata)}
    diff_idx_map = {d: i for i, d in enumerate(diff_order)}
    global_slack = 0
    for diff in diff_order:
        qd_target = int(diff_quota.get(diff, 0))
        if qd_target <= 0:
            continue
        d_idx = diff_idx_map[diff]
        sids = [d_idx * n_len_buckets + b for b in range(n_len_buckets)]
        caps = {sid: int(capacity.get(sid, 0)) for sid in sids}
        cap_sum = sum(caps.values())
        if cap_sum <= 0:
            global_slack += qd_target
            continue
        qd = min(qd_target, cap_sum)
        global_slack += qd_target - qd

        # Initial proportional allocation.
        alloc: dict[int, int] = {sid: 0 for sid in sids}
        for sid in sids:
            c = caps[sid]
            if c:
                alloc[sid] = int(round(qd * (c / cap_sum)))

        # Fix rounding within difficulty.
        while sum(alloc.values()) < qd:
            sid = max(sids, key=lambda s: (caps[s] - alloc[s], caps[s]))
            if alloc[sid] >= caps[sid]:
                break
            alloc[sid] += 1
        while sum(alloc.values()) > qd:
            sid = max(sids, key=lambda s: alloc[s])
            if alloc[sid] <= 0:
                break
            alloc[sid] -= 1

        # Cap to capacity and collect local slack.
        local_slack = 0
        for sid in sids:
            if alloc[sid] > caps[sid]:
                local_slack += alloc[sid] - caps[sid]
                alloc[sid] = caps[sid]

        # Redistribute local slack within this difficulty.
        while local_slack > 0:
            sid = max(sids, key=lambda s: (caps[s] - alloc[s], caps[s]))
            if alloc[sid] >= caps[sid]:
                break
            alloc[sid] += 1
            local_slack -= 1

        global_slack += local_slack
        for sid in sids:
            quota[sid] = int(alloc[sid])

    # Global fill to exact target_k (use any remaining capacity, including unknown if needed).
    _round_robin_fill(quota, capacity, target_total=target_k)
    if sum(quota.values()) != target_k:
        raise RuntimeError(f"quota sum {sum(quota.values())} != target_k {target_k}")

    # Select within each stratum.
    dense_fraction = float(args.dense_fraction)
    if not (0.0 <= dense_fraction <= 1.0):
        raise SystemExit("--dense_fraction must be in [0,1]")

    selected_rows: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for sid, k in sorted(quota.items()):
        if k <= 0:
            continue
        items = stratum_items.get(sid) or []
        if not items:
            continue
        # items: (comp_key, rec). rec[0]=bucket_count
        items.sort(key=lambda x: int(x[1][0]), reverse=True)
        k = min(k, len(items))
        dense_n = int(round(k * dense_fraction))
        dense_n = max(0, min(dense_n, k))
        tail_n = k - dense_n
        chosen = [it for it in items[:dense_n]]
        tail_pool = items[dense_n:]
        if tail_n and tail_pool:
            idx = rng.choice(np.arange(len(tail_pool)), size=tail_n, replace=False)
            chosen.extend([tail_pool[int(i)] for i in idx.tolist()])
        if len(chosen) != k:
            raise RuntimeError(f"stratum {sid} selected {len(chosen)} != {k}")

        diff = diff_order[sid // n_len_buckets]
        b = sid % n_len_buckets
        lo, hi = edges[b], edges[b + 1]
        label = _bucket_label(lo, hi)
        for comp_key, rec in chosen:
            rid = str(rec[2])
            if not rid or rid in selected_ids:
                continue
            selected_ids.add(rid)
            selected_rows.append(
                {
                    "id": rid,
                    "dataset": str(rec[3]),
                    "split": str(rec[4]),
                    "mix_group": str(args.mix_group),
                    "difficulty_bin": diff,
                    "meta_difficulty_bin": str(rec[6]),
                    "prompt_tokens": int(rec[5]),
                    "len_bucket": label,
                    "bucket_key": int(int(comp_key) & bucket_mask),
                    "bucket_count": int(rec[0]),
                    "score": float(rec[1]),
                }
            )

    # Safety: ensure exact target_k unique ids (top up if duplicates reduced count).
    if len(selected_rows) < target_k:
        need = target_k - len(selected_rows)
        print(f"[warn] after dedup, short by {need}; topping up from remaining buckets", flush=True)
        # Flatten remaining candidates and top up.
        pool: list[tuple[int, list[Any], int]] = []
        for sid, items in stratum_items.items():
            for comp_key, rec in items:
                rid = str(rec[2])
                if rid and rid not in selected_ids:
                    pool.append((sid, rec, comp_key))
        rng.shuffle(pool)
        for sid, rec, comp_key in pool:
            if len(selected_rows) >= target_k:
                break
            rid = str(rec[2])
            if not rid or rid in selected_ids:
                continue
            selected_ids.add(rid)
            diff = diff_order[sid // n_len_buckets]
            b = sid % n_len_buckets
            lo, hi = edges[b], edges[b + 1]
            label = _bucket_label(lo, hi)
            selected_rows.append(
                {
                    "id": rid,
                    "dataset": str(rec[3]),
                    "split": str(rec[4]),
                    "mix_group": str(args.mix_group),
                    "difficulty_bin": diff,
                    "meta_difficulty_bin": str(rec[6]),
                    "prompt_tokens": int(rec[5]),
                    "len_bucket": label,
                    "bucket_key": int(int(comp_key) & bucket_mask),
                    "bucket_count": int(rec[0]),
                    "score": float(rec[1]),
                }
            )

    if len(selected_rows) != target_k:
        raise RuntimeError(f"selected {len(selected_rows)} rows, expected {target_k}")

    out_parquet = out_dir / "reasoning_selected_meta.parquet"
    pq.write_table(pa.Table.from_pylist(selected_rows), out_parquet, compression="zstd")

    out_ids = out_dir / "reasoning_ids_10000.txt"
    out_ids.write_text("\n".join([r["id"] for r in selected_rows]) + "\n", encoding="utf-8")

    # Summaries.
    stratum_summary: list[dict[str, Any]] = []
    for sid in range(n_strata):
        items = stratum_items.get(sid) or []
        cap = len(items)
        if cap == 0 and quota.get(sid, 0) == 0:
            continue
        diff = diff_order[sid // n_len_buckets]
        b = sid % n_len_buckets
        lo, hi = edges[b], edges[b + 1]
        label = _bucket_label(lo, hi)
        stratum_summary.append(
            {
                "difficulty_bin": diff,
                "len_bucket": label,
                "rows": int(stratum_row_counts[sid].item()),
                "unique_buckets": int(cap),
                "selected": int(quota.get(sid, 0)),
            }
        )

    manifest = {
        "generated_at": _now(),
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "mix_group": str(args.mix_group),
        "target_k": int(target_k),
        "bits": int(bits),
        "seed": int(args.seed),
        "dense_fraction": float(dense_fraction),
        "len_quantiles": quantiles,
        "len_bucket_edges": edges,
        "difficulty_weights": diff_weights,
        "difficulty_capacity": diff_cap,
        "difficulty_quota": diff_quota,
        "score_field": score_field,
        "rows_scanned": int(total_rows),
        "unique_buckets_total": int(total_capacity),
        "elapsed_s": float(time.time() - t0),
        "strata": stratum_summary,
        "outputs": {
            "selected_meta_parquet": str(out_parquet),
            "ids_txt": str(out_ids),
        },
    }
    (out_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[ok] wrote {out_parquet}", flush=True)
    print(f"[ok] wrote {out_ids}", flush=True)
    print(f"[ok] wrote {out_dir / 'selection_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
