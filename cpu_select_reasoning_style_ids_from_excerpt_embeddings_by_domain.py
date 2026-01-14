#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from hashlib import sha1
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


def _stable_u31(s: str) -> int:
    h = sha1((s or "").encode("utf-8")).hexdigest()
    return int(h[:8], 16) & ((1 << 31) - 1)


def _parse_quota_json(spec: str, *, target_k: int) -> dict[str, int]:
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
    if sum(out.values()) != target_k:
        raise SystemExit(f"--domain_quota_json sum must equal target_k={target_k}")
    if any(x < 0 for x in out.values()):
        raise SystemExit("--domain_quota_json must have non-negative counts")
    return out


def _default_quota(target_k: int) -> dict[str, int]:
    # Equal-weight default across major domains; fixes "math-only" skew.
    doms = ["math", "proof", "science", "agentic", "chat_if"]
    base = {d: target_k // len(doms) for d in doms}
    rem = target_k - sum(base.values())
    for i in range(rem):
        base[doms[i % len(doms)]] += 1
    return base


def _dedup_table_column_names(tbl: pa.Table) -> pa.Table:
    names = list(tbl.schema.names)
    seen: dict[str, int] = {}
    out: list[str] = []
    changed = False
    for n in names:
        k = seen.get(n, 0)
        if k == 0:
            out.append(n)
        else:
            out.append(f"{n}__dup{k}")
            changed = True
        seen[n] = k + 1
    return tbl.rename_columns(out) if changed else tbl


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Select a reasoning_style pack from excerpt embeddings with explicit per-domain quotas (LSH-diverse)."
    )
    ap.add_argument("--in_dir", type=str, required=True, help="Directory containing excerpt embedding *.parquet")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--target_k", type=int, default=10_000)
    ap.add_argument("--mix_group", type=str, default="reasoning")
    ap.add_argument("--min_prompt_tokens", type=int, default=1024)
    ap.add_argument("--max_prompt_tokens", type=int, default=2048)
    ap.add_argument("--domain_quota_json", type=str, default="")
    ap.add_argument("--bits", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dense_fraction", type=float, default=0.80)
    ap.add_argument("--score_field", type=str, default="prompt_tokens")
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--max_rows", type=int, default=0, help="Debug cap (0=all)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_k = int(args.target_k)
    if target_k <= 0:
        raise SystemExit("--target_k must be > 0")

    quota = (
        _parse_quota_json(args.domain_quota_json, target_k=target_k)
        if args.domain_quota_json.strip()
        else _default_quota(target_k)
    )

    bits = int(args.bits)
    if bits <= 0 or bits > 63:
        raise SystemExit("--bits must be in [1,63]")

    parquet_files = sorted(in_dir.rglob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"no parquet files under {in_dir}")

    dataset = ds.dataset([str(p) for p in parquet_files], format="parquet")
    cols = set(dataset.schema.names)
    required = {"id", "embedding", "mix_group", "dataset", "split", "meta_domain", "prompt_tokens"}
    missing = required - cols
    if missing:
        raise SystemExit(f"dataset missing required columns: {sorted(missing)}")
    score_field = str(args.score_field)
    if score_field not in cols:
        raise SystemExit(f"missing score_field {score_field!r} in dataset")

    filt = ds.field("mix_group") == str(args.mix_group)
    lo = int(args.min_prompt_tokens)
    hi = int(args.max_prompt_tokens)
    if lo > 0:
        filt = filt & (ds.field("prompt_tokens") >= lo)
    if hi > 0:
        filt = filt & (ds.field("prompt_tokens") <= hi)

    rng = np.random.default_rng(int(args.seed))
    weights = _uint64_weights(bits)
    proj_mat: np.ndarray | None = None
    dim: int | None = None

    # domain -> bucket_key -> [count, best_score, best_id, best_dataset, best_split, best_prompt_tokens]
    agg: dict[str, dict[int, list[Any]]] = {d: {} for d, k in quota.items() if int(k) > 0}
    total = 0

    read_cols = ["id", "embedding", "dataset", "split", "meta_domain", "prompt_tokens", score_field]
    # Avoid projecting the same column twice (can yield duplicate field names like "prompt_tokens").
    read_cols = list(dict.fromkeys(read_cols))
    scanner = dataset.scanner(columns=read_cols, filter=filt, batch_size=int(args.batch_size), use_threads=True)
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        if args.max_rows and total >= int(args.max_rows):
            break
        tbl = _dedup_table_column_names(pa.Table.from_batches([batch]))
        if args.max_rows and (total + tbl.num_rows) > int(args.max_rows):
            tbl = tbl.slice(0, int(args.max_rows) - total)

        ids = [str(x or "") for x in tbl["id"].to_pylist()]
        ds_vals = [str(x or "") for x in tbl["dataset"].to_pylist()]
        sp_vals = [str(x or "") for x in tbl["split"].to_pylist()]
        doms = [str(x or "").strip() or "unknown" for x in tbl["meta_domain"].to_pylist()]
        ptoks = [int(x or 0) for x in tbl["prompt_tokens"].to_pylist()]
        scores_raw = tbl[score_field].to_pylist()

        emb_vals, d = _embedding_array_to_numpy(tbl["embedding"].chunk(0))
        if dim is None:
            dim = d
        elif dim != d:
            raise RuntimeError(f"embedding dim changed from {dim} to {d}")
        if proj_mat is None:
            proj_mat = rng.standard_normal(size=(d, bits)).astype(np.float32)

        proj = emb_vals.astype(np.float32) @ proj_mat
        bits_arr = proj > 0
        keys = (bits_arr.astype(np.uint64) * weights).sum(axis=1).astype(np.uint64)

        for i, k in enumerate(keys.tolist()):
            dom = doms[i]
            if dom not in agg:
                continue
            rid = ids[i]
            if not rid:
                continue
            try:
                score_f = float(scores_raw[i]) if scores_raw[i] is not None else float(ptoks[i])
            except Exception:
                score_f = float(ptoks[i])

            # Mix domain into key to avoid cross-domain bucket collisions.
            dom_hash = _stable_u31(dom)
            comp_key = int((dom_hash << bits) | int(k))

            rec = agg[dom].get(comp_key)
            if rec is None:
                agg[dom][comp_key] = [1, score_f, rid, ds_vals[i], sp_vals[i], ptoks[i]]
                continue
            rec[0] += 1
            if score_f > float(rec[1]):
                rec[1] = score_f
                rec[2] = rid
                rec[3] = ds_vals[i]
                rec[4] = sp_vals[i]
                rec[5] = ptoks[i]

        total += int(tbl.num_rows)

    # Reconcile per-domain quotas against availability. If some domains have fewer distinct
    # buckets than requested, we deterministically reallocate the deficit to other domains
    # that still have spare buckets, keeping the overall pack size fixed.
    available = {d: len(agg.get(d, {})) for d in quota.keys()}
    eff_quota = {d: int(quota[d]) for d in quota.keys()}
    deficit = 0
    for d, want in list(eff_quota.items()):
        cap = int(available.get(d, 0))
        if want > cap:
            deficit += want - cap
            eff_quota[d] = cap
    if deficit:
        donors: list[tuple[str, int]] = []
        for d, cap in available.items():
            spare = int(cap) - int(eff_quota.get(d, 0))
            if spare > 0:
                donors.append((d, spare))
        donors.sort(key=lambda t: (-t[1], t[0]))
        for d, spare in donors:
            take = min(int(spare), int(deficit))
            eff_quota[d] += take
            deficit -= take
            if deficit == 0:
                break
        if deficit:
            raise SystemExit(
                f"cannot satisfy target_k={target_k}: insufficient buckets after reallocation "
                f"(remaining_deficit={deficit}, available={available})"
            )

    (out_dir / "quota_requested.json").write_text(json.dumps(quota, indent=2, sort_keys=True))
    (out_dir / "quota_effective.json").write_text(json.dumps(eff_quota, indent=2, sort_keys=True))
    (out_dir / "bucket_availability.json").write_text(json.dumps(available, indent=2, sort_keys=True))

    out_rows: list[dict[str, Any]] = []
    out_ids: list[str] = []
    dom_counts: Counter[str] = Counter()
    for dom, want in sorted(eff_quota.items(), key=lambda kv: kv[0]):
        want = int(want)
        if want <= 0:
            continue
        buckets = list(agg.get(dom, {}).items())
        if not buckets:
            continue
        buckets.sort(key=lambda kv: int(kv[1][0]), reverse=True)

        dense_n = int(round(want * float(args.dense_fraction)))
        dense_n = max(0, min(dense_n, want))
        tail_n = want - dense_n

        dense = buckets[:dense_n]
        tail_pool = buckets[dense_n:]
        if tail_n > 0 and tail_pool:
            idx = rng.choice(np.arange(len(tail_pool)), size=min(tail_n, len(tail_pool)), replace=False)
            tail = [tail_pool[int(i)] for i in idx.tolist()]
        else:
            tail = []
        chosen = dense + tail

        # Backfill from the remaining buckets if needed.
        if len(chosen) < want:
            chosen = chosen + tail_pool[: (want - len(chosen))]
        if len(chosen) < want:
            raise SystemExit(
                f"internal error: domain {dom} insufficient buckets after quota reconciliation: "
                f"have {len(chosen)} want {want}"
            )

        for key, rec in chosen[:want]:
            rid = str(rec[2])
            out_rows.append(
                {
                    "bucket_key": int(key),
                    "bucket_count": int(rec[0]),
                    "score": float(rec[1]),
                    "id": rid,
                    "dataset": str(rec[3]),
                    "split": str(rec[4]),
                    "prompt_tokens": int(rec[5]),
                    "mix_group": str(args.mix_group),
                    "meta_domain": dom,
                }
            )
            out_ids.append(rid)
            dom_counts[dom] += 1

    # Dedup (safety) and validate size.
    ids_unique = list(dict.fromkeys([x for x in out_ids if x]))
    if len(ids_unique) != target_k:
        raise SystemExit(f"selection size mismatch: got {len(ids_unique)} expected {target_k}")

    safe_name = "reasoning_style"
    out_ids_path = out_dir / f"{safe_name}_ids_{target_k}.txt"
    out_meta_path = out_dir / f"{safe_name}_selected_meta.parquet"
    out_report_path = out_dir / f"{safe_name}_selection_report.md"
    out_manifest_path = out_dir / "selection_manifest.json"

    out_ids_path.write_text("\n".join(ids_unique) + "\n", encoding="utf-8")
    pq.write_table(pa.Table.from_pylist(out_rows), out_meta_path, compression="zstd")

    report = [
        "# Reasoning Style (domain-quota) Selection Report",
        f"- generated_at: `{_now()}`",
        f"- in_dir: `{in_dir}`",
        f"- rows_scanned: `{total}`",
        f"- target_k: `{target_k}`",
        f"- mix_group: `{args.mix_group}`",
        f"- prompt_tokens_range: `{lo}..{hi}`",
        f"- bits: `{bits}` seed: `{int(args.seed)}` dense_fraction: `{float(args.dense_fraction):.3f}`",
        f"- quota_requested: `{quota}`",
        f"- quota_effective: `{eff_quota}`",
        "",
        "## Domain counts (selected)",
        "",
        "\n".join([f"- {k}: `{v}`" for k, v in dom_counts.items()]) or "- (none)",
        "",
    ]
    out_report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    manifest = {
        "generated_at": _now(),
        "in_dir": str(in_dir),
        "rows_scanned": int(total),
        "target_k": int(target_k),
        "mix_group": str(args.mix_group),
        "prompt_tokens_range": {"min": lo, "max": hi},
        "bits": int(bits),
        "seed": int(args.seed),
        "dense_fraction": float(args.dense_fraction),
        "domain_quota": quota,
        "domain_counts": dict(dom_counts),
        "out_ids_path": str(out_ids_path),
        "out_meta_path": str(out_meta_path),
        "out_report_path": str(out_report_path),
    }
    out_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[ok] wrote {out_ids_path}", flush=True)
    print(f"[ok] wrote {out_meta_path}", flush=True)
    print(f"[ok] wrote {out_report_path}", flush=True)
    print(f"[ok] wrote {out_manifest_path}", flush=True)


if __name__ == "__main__":
    main()
