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


def _alloc_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    keys = ["tool", "reasoning", "general"]
    raw = {k: float(total) * float(ratios.get(k, 0.0)) for k in keys}
    base = {k: int(np.floor(v)) for k, v in raw.items()}
    rem = total - sum(base.values())
    if rem > 0:
        fracs = sorted(((k, raw[k] - base[k]) for k in keys), key=lambda kv: (-kv[1], kv[0]))
        for i in range(rem):
            base[fracs[i % len(fracs)][0]] += 1
    return base


def _parse_mix_ratios(spec: str) -> dict[str, float]:
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


def _gini_from_counts(counts: list[int]) -> float:
    if not counts:
        return float("nan")
    x = np.array(counts, dtype=np.float64)
    if np.any(x < 0):
        x = np.maximum(x, 0.0)
    if x.sum() <= 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((np.sum((2 * idx - n - 1) * x)) / (n * np.sum(x)))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a prompt cover (LSH-diverse) with per-dataset quotas to reduce dominance."
    )
    ap.add_argument("--embedding_dir", type=str, required=True, help="Directory containing prompt embedding *.parquet")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--target_n", type=int, default=200_000)
    ap.add_argument("--mix_ratios_json", type=str, default="")
    ap.add_argument("--bits", type=int, default=22)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dense_fraction", type=float, default=0.80)
    ap.add_argument(
        "--score_field",
        type=str,
        default="prompt_tokens",
        help="Choose best row per (dataset, bucket) by this numeric field.",
    )
    ap.add_argument(
        "--max_dataset_frac",
        type=float,
        default=0.35,
        help="Cap each dataset to <= this fraction of each mix_group selection (0 disables).",
    )
    ap.add_argument(
        "--per_bucket_keep",
        type=int,
        default=3,
        help="Keep up to N candidates per LSH bucket (distinct datasets) to satisfy dataset caps.",
    )
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--max_rows", type=int, default=0, help="Debug cap (0=all)")
    ap.add_argument("--prompt_tokens_min", type=int, default=0, help="Optional filter: prompt_tokens >= this")
    ap.add_argument("--prompt_tokens_max", type=int, default=0, help="Optional filter: prompt_tokens <= this (0 disables)")
    ap.add_argument("--filter_dataset", type=str, default="", help="Optional exact dataset filter")
    args = ap.parse_args()

    in_dir = Path(args.embedding_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_n = int(args.target_n)
    if target_n <= 0:
        raise SystemExit("--target_n must be > 0")

    mix_ratios = _parse_mix_ratios(args.mix_ratios_json)
    mix_targets = _alloc_counts(target_n, mix_ratios)
    bits = int(args.bits)
    if bits <= 0 or bits > 63:
        raise SystemExit("--bits must be in [1,63]")
    dense_fraction = float(args.dense_fraction)
    if dense_fraction < 0.0 or dense_fraction > 1.0:
        raise SystemExit("--dense_fraction must be within [0,1]")

    parquet_files = sorted(in_dir.rglob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"no parquet files under {in_dir}")

    dataset = ds.dataset([str(p) for p in parquet_files], format="parquet")
    cols = set(dataset.schema.names)
    required = {"id", "embedding", "mix_group", "dataset", "split"}
    missing = required - cols
    if missing:
        raise SystemExit(f"dataset missing required columns: {sorted(missing)}")
    score_field = str(args.score_field)
    if score_field not in cols:
        raise SystemExit(f"missing score_field {score_field!r} in embeddings dataset")

    rng = np.random.default_rng(int(args.seed))
    weights = _uint64_weights(bits)
    proj_mat: np.ndarray | None = None
    dim: int | None = None

    # For each mix_group, we aggregate bucket -> list[candidate], where candidate holds (score, id, dataset, split, prompt_tokens).
    # We keep up to per_bucket_keep distinct datasets per bucket, to support dataset caps later.
    per_bucket_keep = int(args.per_bucket_keep)
    if per_bucket_keep <= 0:
        raise SystemExit("--per_bucket_keep must be > 0")

    agg: dict[str, dict[int, list[list[Any]]]] = {k: {} for k in ("tool", "reasoning", "general")}
    bucket_counts: dict[str, Counter[int]] = {k: Counter() for k in ("tool", "reasoning", "general")}
    rows_scanned = 0

    read_cols = ["id", "embedding", "mix_group", "dataset", "split", score_field]
    if "prompt_tokens" in cols and score_field != "prompt_tokens":
        read_cols.append("prompt_tokens")

    t0 = time.time()
    filt = None
    ds_filter = str(args.filter_dataset).strip()
    if ds_filter:
        filt = ds.field("dataset") == ds_filter
    if int(args.prompt_tokens_min) > 0:
        f = ds.field("prompt_tokens") >= int(args.prompt_tokens_min)
        filt = f if filt is None else (filt & f)
    if int(args.prompt_tokens_max) > 0:
        f = ds.field("prompt_tokens") <= int(args.prompt_tokens_max)
        filt = f if filt is None else (filt & f)

    scanner = dataset.scanner(
        columns=read_cols, filter=filt, batch_size=int(args.batch_size), use_threads=True
    )
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        if args.max_rows and rows_scanned >= int(args.max_rows):
            break
        tbl = pa.Table.from_batches([batch])
        if args.max_rows and (rows_scanned + tbl.num_rows) > int(args.max_rows):
            tbl = tbl.slice(0, int(args.max_rows) - rows_scanned)

        mix = [str(x or "") for x in tbl["mix_group"].to_pylist()]
        ids = [str(x or "") for x in tbl["id"].to_pylist()]
        dss = [str(x or "") for x in tbl["dataset"].to_pylist()]
        sps = [str(x or "") for x in tbl["split"].to_pylist()]
        scores_raw = tbl[score_field].to_pylist()
        ptoks = tbl["prompt_tokens"].to_pylist() if "prompt_tokens" in tbl.column_names else None

        emb_vals, d = _embedding_array_to_numpy(tbl["embedding"].chunk(0))
        if dim is None:
            dim = d
        elif dim != d:
            raise RuntimeError(f"embedding dim changed from {dim} to {d}")
        if proj_mat is None:
            proj_mat = rng.standard_normal(size=(d, bits)).astype(np.float32)

        keys = ((emb_vals.astype(np.float32) @ proj_mat) > 0).astype(np.uint64)
        keys = (keys * weights).sum(axis=1).astype(np.uint64)

        for i, k in enumerate(keys.tolist()):
            mg = mix[i]
            if mg not in agg:
                continue
            rid = ids[i]
            if not rid:
                continue
            ds_name = dss[i]
            sp_name = sps[i]
            try:
                score_f = float(scores_raw[i]) if scores_raw[i] is not None else -1.0
            except Exception:
                score_f = -1.0
            prompt_tokens_i = int(ptoks[i] or 0) if ptoks is not None else 0

            bucket_counts[mg][int(k)] += 1
            bucket = agg[mg].get(int(k))
            if bucket is None:
                agg[mg][int(k)] = [[score_f, rid, ds_name, sp_name, prompt_tokens_i]]
                continue

            # Keep top per_bucket_keep, prefer distinct datasets.
            # If same dataset exists, update if score is higher.
            updated = False
            for rec in bucket:
                if rec[2] == ds_name:
                    if score_f > float(rec[0]):
                        rec[0] = score_f
                        rec[1] = rid
                        rec[3] = sp_name
                        rec[4] = prompt_tokens_i
                    updated = True
                    break
            if updated:
                continue
            if len(bucket) < per_bucket_keep:
                bucket.append([score_f, rid, ds_name, sp_name, prompt_tokens_i])
                continue
            # Replace worst if better and introduces new dataset.
            worst_i = min(range(len(bucket)), key=lambda j: float(bucket[j][0]))
            if score_f > float(bucket[worst_i][0]):
                bucket[worst_i] = [score_f, rid, ds_name, sp_name, prompt_tokens_i]

        rows_scanned += int(tbl.num_rows)
        if rows_scanned and (rows_scanned % 1_000_000 == 0):
            dt = time.time() - t0
            print(f"[prog] rows_scanned={rows_scanned} dt={dt:.1f}s", flush=True)

    if dim is None:
        raise SystemExit("no embeddings scanned")

    # Selection per mix_group with dataset caps.
    all_selected: list[dict[str, Any]] = []
    for mg in ("tool", "reasoning", "general"):
        target_k = int(mix_targets.get(mg, 0))
        if target_k <= 0:
            continue
        buckets = list(agg[mg].items())  # (bucket_key, candidates[])
        if not buckets:
            raise SystemExit(f"no rows for mix_group={mg}")
        # Sort buckets by observed density (count descending) for representativeness.
        buckets.sort(key=lambda kv: bucket_counts[mg][int(kv[0])], reverse=True)

        dense_n = int(round(target_k * dense_fraction))
        dense_n = max(0, min(dense_n, target_k))
        tail_n = target_k - dense_n

        dense_buckets = buckets[:dense_n]
        tail_pool = buckets[dense_n:]
        if tail_n > 0 and tail_pool:
            tail_idx = rng.choice(np.arange(len(tail_pool)), size=min(tail_n, len(tail_pool)), replace=False)
            tail_buckets = [tail_pool[int(i)] for i in tail_idx.tolist()]
        else:
            tail_buckets = []

        bucket_order = dense_buckets + tail_buckets

        ds_cap = None
        if float(args.max_dataset_frac) > 0:
            ds_cap = int(math.floor(float(args.max_dataset_frac) * target_k))
            ds_cap = max(1, ds_cap)
        ds_counts: Counter[str] = Counter()
        selected_ids: set[str] = set()

        # Pass 1: enforce dataset cap strictly.
        for bucket_key, cand_list in bucket_order:
            if len(all_selected) >= target_n and mg == "general":
                break
            # Prefer highest score candidate that doesn't violate dataset cap.
            cand_list_sorted = sorted(cand_list, key=lambda r: float(r[0]), reverse=True)
            chosen: list[Any] | None = None
            for rec in cand_list_sorted:
                rid = str(rec[1])
                ds_name = str(rec[2])
                if not rid or rid in selected_ids:
                    continue
                if ds_cap is not None and ds_counts[ds_name] >= ds_cap:
                    continue
                chosen = rec
                break
            if chosen is None:
                continue
            rid = str(chosen[1])
            ds_name = str(chosen[2])
            selected_ids.add(rid)
            ds_counts[ds_name] += 1
            all_selected.append(
                {
                    "bucket_key": int(bucket_key),
                    "bucket_count": int(bucket_counts[mg][int(bucket_key)]),
                    "score": float(chosen[0]),
                    "id": rid,
                    "dataset": ds_name,
                    "split": str(chosen[3]),
                    "prompt_tokens": int(chosen[4]),
                    "mix_group": mg,
                }
            )
            if len([r for r in all_selected if r["mix_group"] == mg]) >= target_k:
                break

        # Pass 2: backfill any missing with relaxed caps (still dedup by id).
        current_mg = [r for r in all_selected if r["mix_group"] == mg]
        if len(current_mg) < target_k:
            need = target_k - len(current_mg)
            for bucket_key, cand_list in bucket_order:
                if need <= 0:
                    break
                cand_list_sorted = sorted(cand_list, key=lambda r: float(r[0]), reverse=True)
                for rec in cand_list_sorted:
                    rid = str(rec[1])
                    if not rid or rid in selected_ids:
                        continue
                    selected_ids.add(rid)
                    all_selected.append(
                        {
                            "bucket_key": int(bucket_key),
                            "bucket_count": int(bucket_counts[mg][int(bucket_key)]),
                            "score": float(rec[0]),
                            "id": rid,
                            "dataset": str(rec[2]),
                            "split": str(rec[3]),
                            "prompt_tokens": int(rec[4]),
                            "mix_group": mg,
                        }
                    )
                    need -= 1
                    break
            current_mg = [r for r in all_selected if r["mix_group"] == mg]
            if len(current_mg) != target_k:
                raise SystemExit(f"failed to fill mix_group={mg}: got {len(current_mg)} want {target_k}")

    # Final size check / dedup.
    ids = [r["id"] for r in all_selected]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate ids detected in cover selection")
    if len(all_selected) != target_n:
        raise SystemExit(f"cover size mismatch: expected {target_n} got {len(all_selected)}")

    out_meta = out_dir / "prompt_cover_200k_selected_meta.parquet"
    out_ids = out_dir / "prompt_cover_200k_ids.txt"
    out_report = out_dir / "prompt_cover_200k_report.md"
    out_manifest = out_dir / "prompt_cover_200k_manifest.json"

    pq.write_table(pa.Table.from_pylist(all_selected), out_meta, compression="zstd")
    out_ids.write_text("\n".join(ids) + "\n", encoding="utf-8")

    ds_counts_all = Counter([r["dataset"] for r in all_selected])
    mg_counts = Counter([r["mix_group"] for r in all_selected])
    ds_gini = _gini_from_counts([int(v) for _, v in ds_counts_all.most_common()])

    report_lines = [
        "# Prompt Cover v2 (dataset-quota-aware) Report",
        f"- generated_at: `{_now()}`",
        f"- embedding_dir: `{in_dir}`",
        f"- target_n: `{target_n}`",
        f"- mix_targets: `{mix_targets}`",
        f"- bits: `{bits}` seed: `{int(args.seed)}` dense_fraction: `{dense_fraction}`",
        f"- max_dataset_frac(per mix_group): `{float(args.max_dataset_frac)}`",
        f"- per_bucket_keep: `{per_bucket_keep}`",
        f"- rows_scanned: `{rows_scanned}`",
        "",
        "## Mix-group counts",
        "",
        "\n".join([f"- {k}: `{v}`" for k, v in mg_counts.items()]) or "- (none)",
        "",
        "## Dataset counts (top 20)",
        "",
        "\n".join([f"- {k}: `{v}`" for k, v in ds_counts_all.most_common(20)]) or "- (none)",
        "",
        f"- dataset_count_gini: `{ds_gini:.4f}` (lower is more balanced)",
        "",
    ]
    out_report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest = {
        "generated_at": _now(),
        "embedding_dir": str(in_dir),
        "target_n": int(target_n),
        "mix_ratios": mix_ratios,
        "mix_targets": mix_targets,
        "bits": bits,
        "seed": int(args.seed),
        "dense_fraction": dense_fraction,
        "score_field": score_field,
        "max_dataset_frac": float(args.max_dataset_frac),
        "per_bucket_keep": per_bucket_keep,
        "rows_scanned": int(rows_scanned),
        "out_meta": str(out_meta),
        "out_ids": str(out_ids),
        "out_report": str(out_report),
        "dataset_counts": dict(ds_counts_all),
        "mix_group_counts": dict(mg_counts),
        "dataset_count_gini": float(ds_gini),
    }
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_meta}", flush=True)
    print(f"[ok] wrote {out_ids}", flush=True)
    print(f"[ok] wrote {out_report}", flush=True)
    print(f"[ok] wrote {out_manifest}", flush=True)


if __name__ == "__main__":
    main()
