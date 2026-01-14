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
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import zlib


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def _parse_quota_json(spec: str) -> dict[str, int]:
    try:
        obj = json.loads(spec)
    except Exception as e:
        raise SystemExit(f"--domain_quota_json must be valid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise SystemExit("--domain_quota_json must be a JSON object mapping domain -> count")
    out: dict[str, int] = {}
    for k, v in obj.items():
        kk = str(k).strip()
        if not kk:
            continue
        out[kk] = int(v)
    if not out or any(x <= 0 for x in out.values()):
        raise SystemExit("--domain_quota_json must contain positive counts")
    return out


def _alloc_default(total: int) -> dict[str, int]:
    # Default: enforce multi-domain coverage for deep reasoning excerpt view.
    # This fixes the common skew where long excerpts are dominated by math.
    base = {
        "math": int(round(total * 0.35)),
        "proof": int(round(total * 0.15)),
        "science": int(round(total * 0.15)),
        "agentic": int(round(total * 0.175)),
        "chat_if": int(round(total * 0.175)),
    }
    s = sum(base.values())
    if s != total:
        # Deterministic adjust remainder onto math.
        base["math"] += (total - s)
    return base


def _domain_key(x: Any) -> str:
    s = str(x or "").strip()
    return s if s else "unknown"


def _read_dataset(in_dir: Path) -> ds.Dataset:
    files = sorted(in_dir.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {in_dir}")
    return ds.dataset([str(p) for p in files], format="parquet")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Select a deep-reasoning excerpt pool with explicit per-domain quotas (from pre-tokenized candidates)."
    )
    ap.add_argument("--in_dir", type=str, required=True, help="Directory containing reasoning_excerpt candidates *.parquet")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--target_n", type=int, default=300_000)
    ap.add_argument(
        "--domain_quota_json",
        type=str,
        default="",
        help="Optional JSON mapping domain->count (sum must equal target_n). If empty, uses a conservative default mix.",
    )
    ap.add_argument("--min_tok_len", type=int, default=1024, help="Keep only rows with stats_tok_len >= this")
    ap.add_argument("--max_tok_len", type=int, default=2048, help="Keep only rows with stats_tok_len <= this (0 disables)")
    ap.add_argument(
        "--score_field",
        type=str,
        default="stats_tok_len",
        help="Field to rank within each domain when over-subscribed (default: stats_tok_len, i.e., prefer longer).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=131_072)
    ap.add_argument("--max_rows_scan", type=int, default=0, help="Debug cap (0=all)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[*] select excerpt pool: in_dir={in_dir} target_n={args.target_n} "
        f"min_tok_len={args.min_tok_len} max_tok_len={args.max_tok_len} score_field={args.score_field!r} "
        f"batch_size={args.batch_size} seed={args.seed}",
        flush=True,
    )

    target_n = int(args.target_n)
    if target_n <= 0:
        raise SystemExit("--target_n must be > 0")

    if args.domain_quota_json.strip():
        quota = _parse_quota_json(args.domain_quota_json)
        if sum(quota.values()) != target_n:
            raise SystemExit(
                f"--domain_quota_json sum must equal target_n={target_n}, got {sum(quota.values())}"
            )
    else:
        quota = _alloc_default(target_n)

    dataset = _read_dataset(in_dir)
    cols = set(dataset.schema.names)
    required = {"id", "dataset", "split", "meta_domain", "stats_tok_len"}
    missing = required - cols
    if missing:
        raise SystemExit(f"dataset missing required columns: {sorted(missing)}")
    score_field = str(args.score_field)
    if score_field not in cols:
        raise SystemExit(f"missing score_field {score_field!r} in dataset")

    # Pass 1: compute availability per domain under length bounds.
    lo = int(args.min_tok_len)
    hi = int(args.max_tok_len)
    tok = ds.field("stats_tok_len")
    filt = tok >= lo
    if hi > 0:
        filt = filt & (tok <= hi)

    counts: Counter[str] = Counter()
    rows_scanned = 0
    scanner = dataset.scanner(columns=["meta_domain"], filter=filt, batch_size=int(args.batch_size), use_threads=True)
    for batch in scanner.to_batches():
        rows_scanned += int(batch.num_rows)
        dom = batch.column(0)
        if dom.null_count:
            dom = pc.fill_null(dom, "unknown")
        vc = pc.value_counts(dom)
        values = vc.field("values").to_pylist()
        cs = vc.field("counts").to_pylist()
        for v, c in zip(values, cs, strict=True):
            counts[_domain_key(v)] += int(c)
        if args.max_rows_scan and rows_scanned >= int(args.max_rows_scan):
            break

    top_counts = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
    print(f"[*] availability under length bounds: rows_scanned={rows_scanned} top_domains={top_counts}", flush=True)

    # Reconcile quotas with availability (borrow shortfalls deterministically).
    quota_adj = dict(quota)
    shortfall = 0
    for d, want in list(quota_adj.items()):
        have = int(counts.get(d, 0))
        if have < want:
            shortfall += (want - have)
            quota_adj[d] = have
    if shortfall:
        # Borrow from math first, then agentic, then chat_if, then science, then proof.
        borrow_order = ["math", "agentic", "chat_if", "science", "proof"]
        for d in borrow_order:
            have = int(counts.get(d, 0))
            cur = int(quota_adj.get(d, 0))
            spare = max(0, have - cur)
            if spare <= 0:
                continue
            take = min(spare, shortfall)
            quota_adj[d] = cur + take
            shortfall -= take
            if shortfall <= 0:
                break

    # Only do a tiny, safe fix-up for rounding mismatches; never introduce new domains here.
    delta = target_n - sum(quota_adj.values())
    if delta != 0:
        if "math" in quota_adj:
            quota_adj["math"] = int(quota_adj.get("math", 0)) + int(delta)
        elif quota_adj:
            # Adjust the largest available domain among those already present.
            adjust_dom = max(quota_adj.keys(), key=lambda d: int(counts.get(d, 0)))
            quota_adj[adjust_dom] = int(quota_adj.get(adjust_dom, 0)) + int(delta)

    print(f"[*] quota_adj(row-based)={quota_adj} shortfall_remaining={shortfall}", flush=True)

    # Pass 2: select IDs per domain, preferring higher score_field and then random tie-break.
    seed_int = int(args.seed)

    def _stable_tie(sample_id: str) -> int:
        # Deterministic tie-break for reproducibility across file order / duplicates.
        return int(zlib.crc32(f"{seed_int}:{sample_id}".encode("utf-8")) & 0x7FFFFFFF)

    # Keep the best (highest-score) row per id, per domain. This avoids long-text duplicates
    # crowding out unique samples when ranking by length (common in proof/science).
    per_domain_best: dict[str, dict[str, tuple[float, int, str, str]]] = {
        d: {} for d, k in quota_adj.items() if k > 0
    }

    # Avoid duplicate field names in projected schema (e.g., when score_field == "stats_tok_len").
    read_cols = list(dict.fromkeys(["id", "dataset", "split", "meta_domain", "stats_tok_len", score_field]))
    scanner = dataset.scanner(columns=read_cols, filter=filt, batch_size=int(args.batch_size), use_threads=True)
    processed = 0
    t0 = time.time()
    for batch in scanner.to_batches():
        table = pa.Table.from_batches([batch])

        def _col_values(col_name: str) -> list[Any]:
            names = table.schema.names
            try:
                idx = names.index(col_name)
            except ValueError:
                raise KeyError(f"missing required column {col_name!r} in scan batch (cols={names})")
            return table.column(idx).to_pylist()

        ids = [str(x or "") for x in _col_values("id")]
        dss = [str(x or "") for x in _col_values("dataset")]
        sps = [str(x or "") for x in _col_values("split")]
        doms = [_domain_key(x) for x in _col_values("meta_domain")]
        tok_len = [int(x or 0) for x in _col_values("stats_tok_len")]
        scores_raw = _col_values(score_field)

        for rid, ds_name, sp_name, dom, tl, sc in zip(ids, dss, sps, doms, tok_len, scores_raw, strict=True):
            if not rid or dom not in per_domain_best:
                continue
            try:
                score_f = float(sc) if sc is not None else float(tl)
            except Exception:
                score_f = float(tl)
            tie = _stable_tie(rid)
            cur = per_domain_best[dom].get(rid)
            if cur is None or (score_f > cur[0]) or (score_f == cur[0] and tie < cur[1]):
                per_domain_best[dom][rid] = (score_f, tie, ds_name, sp_name)

        processed += int(table.num_rows)
        if processed and (processed % 2_000_000 == 0):
            dt = time.time() - t0
            print(f"[prog] scanned={processed} dt={dt:.1f}s", flush=True)

    # Reconcile quotas against UNIQUE-ID availability (duplicates can be heavy in long-text subsets).
    uniq_counts = {dom: len(best) for dom, best in per_domain_best.items()}
    quota_final = dict(quota_adj)
    uniq_shortfall = 0
    for dom, want in list(quota_final.items()):
        have = int(uniq_counts.get(dom, 0))
        if have < int(want):
            uniq_shortfall += int(want) - have
            quota_final[dom] = have
    if uniq_shortfall:
        borrow_order = ["math", "agentic", "chat_if", "science", "proof"]
        for dom in borrow_order:
            have = int(uniq_counts.get(dom, 0))
            cur = int(quota_final.get(dom, 0))
            spare = max(0, have - cur)
            if spare <= 0:
                continue
            take = min(spare, uniq_shortfall)
            quota_final[dom] = cur + take
            uniq_shortfall -= take
            if uniq_shortfall <= 0:
                break

    # Safety: never introduce new domains at this stage; if we still can't fill target_n, fail fast.
    if sum(quota_final.values()) != target_n:
        raise SystemExit(
            f"quota_final sum mismatch: expected {target_n}, got {sum(quota_final.values())}. "
            f"uniq_counts={uniq_counts} quota_final={quota_final}"
        )

    print(f"[*] uniq_counts={uniq_counts} quota_final={quota_final} uniq_shortfall_remaining={uniq_shortfall}", flush=True)

    selected_rows: list[dict[str, Any]] = []
    selected_ids: list[str] = []
    for dom, want in sorted(quota_final.items(), key=lambda kv: kv[0]):
        if want <= 0:
            continue
        best = per_domain_best.get(dom) or {}
        if not best:
            continue
        items = [(v[0], v[1], rid, v[2], v[3]) for rid, v in best.items()]
        # Sort descending by score, then tie.
        items.sort(key=lambda x: (-x[0], x[1]))
        take = items[: int(want)]
        for score_f, tie, rid, ds_name, sp_name in take:
            selected_rows.append(
                {
                    "id": rid,
                    "dataset": ds_name,
                    "split": sp_name,
                    "meta_domain": dom,
                    "stats_tok_len": int(math.floor(score_f)) if score_field == "stats_tok_len" else None,
                    "score_field": score_field,
                    "score": float(score_f),
                }
            )
            selected_ids.append(rid)

    # Dedup / size check.
    seen: set[str] = set()
    dedup_rows: list[dict[str, Any]] = []
    for r in selected_rows:
        rid = str(r["id"] or "")
        if not rid or rid in seen:
            continue
        seen.add(rid)
        dedup_rows.append(r)

    if len(dedup_rows) != target_n:
        raise SystemExit(
            f"selection size mismatch: expected {target_n}, got {len(dedup_rows)}. "
            "This indicates insufficient candidates under the length bounds for one or more domains; "
            "lower --min_tok_len or increase candidate universe."
        )

    out_ids = out_dir / f"reasoning_excerpt_pool_ids_{target_n}.txt"
    out_meta = out_dir / f"reasoning_excerpt_pool_selected_meta_{target_n}.parquet"
    out_manifest = out_dir / "pool_manifest.json"

    out_ids.write_text("\n".join(selected_ids) + "\n", encoding="utf-8")
    pq.write_table(pa.Table.from_pylist(dedup_rows), out_meta, compression="zstd")

    manifest = {
        "generated_at": _now(),
        "in_dir": str(in_dir),
        "target_n": int(target_n),
        "length_bounds": {"min_tok_len": lo, "max_tok_len": hi if hi > 0 else None},
        "quota_requested": quota,
        "quota_effective_row": quota_adj,
        "quota_effective_unique": quota_final,
        "unique_id_availability_under_bounds": {k: int(v) for k, v in uniq_counts.items()},
        "availability_under_bounds": {k: int(v) for k, v in counts.items()},
        "out_ids": str(out_ids),
        "out_meta": str(out_meta),
        "rows_scanned": int(rows_scanned),
        "seed": int(args.seed),
    }
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_ids}", flush=True)
    print(f"[ok] wrote {out_meta}", flush=True)
    print(f"[ok] wrote {out_manifest}", flush=True)


if __name__ == "__main__":
    main()
