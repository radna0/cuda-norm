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
from hashlib import sha1


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


def _parse_trace_sketch(embed_text: str) -> dict[str, Any]:
    # Parse the behavior trace sketch from `harmony_text.build_behavior_signature`.
    #
    # We only need low-entropy structured fields for quota gating; do NOT parse tool payloads.
    out: dict[str, Any] = {"tool_seq": "none", "calls": 0, "returns": "none"}
    s = (embed_text or "").strip()
    if not s:
        return out
    for line in s.splitlines():
        ln = line.strip()
        if ln.startswith("TOOL_SEQ:"):
            out["tool_seq"] = ln.split(":", 1)[1].strip() or "none"
        elif ln.startswith("CALLS:"):
            try:
                out["calls"] = int(ln.split(":", 1)[1].strip())
            except Exception:
                out["calls"] = 0
        elif ln.startswith("RETURNS:"):
            out["returns"] = ln.split(":", 1)[1].strip() or "none"
    return out


def _tool_seq_key(seq: str) -> str:
    # Normalize tool sequence for quota keys.
    s = (seq or "").strip()
    if not s or s == "none":
        return "none"
    return " ".join([p.strip() for p in s.split("->") if p.strip()])[:400]


def _stable_u31(s: str) -> int:
    h = sha1((s or "").encode("utf-8")).hexdigest()
    return int(h[:8], 16) & ((1 << 31) - 1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Select diverse IDs from embeddings using LSH buckets (medoid-per-bucket style)"
    )
    ap.add_argument("--in_dir", type=str, required=True, help="Directory containing embedding *.parquet")
    ap.add_argument(
        "--name",
        type=str,
        default="",
        help="Output name prefix (default: derived from mix_group/meta_domain).",
    )
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
        "--meta_domain",
        type=str,
        default="",
        help="Optional filter on meta_domain (e.g. agentic, math, proof, science, chat_if).",
    )
    ap.add_argument(
        "--require_quality_has_tool",
        action="store_true",
        help="Require quality_has_tool==True (if column exists).",
    )
    ap.add_argument(
        "--require_quality_valid_tool_schema",
        action="store_true",
        help="Require quality_valid_tool_schema==True (if column exists).",
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
    ap.add_argument(
        "--candidates_dir",
        type=str,
        default="",
        help="Optional candidates dir containing embed_text (trace sketch). Enables tool_seq quotas and stricter gating.",
    )
    ap.add_argument(
        "--min_calls",
        type=int,
        default=1,
        help="When --candidates_dir is set, require CALLS >= this (default: 1).",
    )
    ap.add_argument(
        "--require_returns_nonmissing",
        action="store_true",
        help="When --candidates_dir is set, require RETURNS to not be all 'missing'.",
    )
    ap.add_argument(
        "--max_per_tool_seq",
        type=int,
        default=50,
        help="Max selections per tool sequence (quota key extracted from TOOL_SEQ).",
    )
    ap.add_argument(
        "--min_unique_tool_seq",
        type=int,
        default=200,
        help="Soft target: try to cover at least this many unique tool sequences (best-effort).",
    )
    ap.add_argument(
        "--tool_seq_none_ok",
        action="store_true",
        help="Allow TOOL_SEQ=none (default rejects when --candidates_dir is set).",
    )
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

    cand_dir = Path(args.candidates_dir) if args.candidates_dir.strip() else None
    cand_map: dict[str, dict[str, Any]] = {}
    if cand_dir is not None:
        cand_files = sorted(cand_dir.rglob("*.parquet"))
        if not cand_files:
            raise SystemExit(f"--candidates_dir has no parquet files: {cand_dir}")
        cand_ds = ds.dataset([str(p) for p in cand_files], format="parquet")
        cand_cols = set(cand_ds.schema.names)
        if "id" not in cand_cols or "embed_text" not in cand_cols:
            raise SystemExit("--candidates_dir must contain columns: id, embed_text")
        print(f"[*] loading candidate trace sketches: {cand_dir} files={len(cand_files)}", flush=True)
        scan = cand_ds.scanner(columns=["id", "embed_text"], batch_size=131072, use_threads=True)
        for batch in scan.to_batches():
            ids = [str(x or "") for x in batch.column(0).to_pylist()]
            texts = [str(x or "") for x in batch.column(1).to_pylist()]
            for rid, txt in zip(ids, texts, strict=True):
                if not rid or rid in cand_map:
                    continue
                cand_map[rid] = _parse_trace_sketch(txt)
        print(f"[*] loaded trace sketches: ids={len(cand_map)}", flush=True)

    filter_expr = ds.field("mix_group") == args.mix_group
    meta_domain = (args.meta_domain or "").strip()
    if meta_domain:
        if "meta_domain" not in cols:
            raise SystemExit("requested --meta_domain but embeddings dataset has no 'meta_domain' column")
        filter_expr = filter_expr & (ds.field("meta_domain") == meta_domain)
    if args.require_quality_has_tool and "quality_has_tool" in cols:
        filter_expr = filter_expr & (ds.field("quality_has_tool") == True)  # noqa: E712
    if args.require_quality_valid_tool_schema and "quality_valid_tool_schema" in cols:
        filter_expr = filter_expr & (ds.field("quality_valid_tool_schema") == True)  # noqa: E712
    read_cols = ["id", "embedding", "mix_group", "dataset", "split", score_field]
    for c in ("meta_domain", "quality_has_tool", "quality_valid_tool_schema"):
        if c in cols:
            read_cols.append(c)
    if "prompt_tokens" in cols and score_field != "prompt_tokens":
        read_cols.append("prompt_tokens")

    rng = np.random.default_rng(int(args.seed))
    weights = _uint64_weights(int(args.bits))
    proj_mat: np.ndarray | None = None
    dim: int | None = None

    # composite_key -> [count, best_score, best_id, best_dataset, best_split, best_prompt_tokens, tool_seq_key]
    # composite_key is (tool_seq_hash<<bits)|lsh_bucket when candidates_dir is provided; otherwise it's just lsh_bucket.
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
            tool_seq = ""
            calls = 0
            returns_s = ""
            if cand_dir is not None:
                info = cand_map.get(rid)
                if not info:
                    continue
                tool_seq = _tool_seq_key(str(info.get("tool_seq") or "none"))
                calls = int(info.get("calls") or 0)
                returns_s = str(info.get("returns") or "")
                if calls < int(args.min_calls):
                    continue
                if not args.tool_seq_none_ok and tool_seq == "none":
                    continue
                if args.require_returns_nonmissing:
                    # Reject if all returns are "missing".
                    parts = [p.strip() for p in returns_s.split("|") if p.strip()]
                    if parts and all(p == "missing" for p in parts):
                        continue
            score = scores[i]
            try:
                score_f = float(score) if score is not None else math.nan
            except Exception:
                score_f = math.nan
            if math.isnan(score_f):
                score_f = -1.0
            prompt_tokens_i = int(ptoks[i] or 0) if ptoks is not None else 0

            if cand_dir is not None:
                # Mix tool_seq into key to enforce per-seq diversity in LSH buckets.
                seq_hash = _stable_u31(tool_seq)
                comp_key = int((seq_hash << int(args.bits)) | int(k))
            else:
                comp_key = int(k)

            rec = agg.get(comp_key)
            if rec is None:
                agg[comp_key] = [1, score_f, rid, ds_vals[i], sp_vals[i], prompt_tokens_i, tool_seq]
                continue

            rec[0] += 1
            if score_f > float(rec[1]):
                rec[1] = score_f
                rec[2] = rid
                rec[3] = ds_vals[i]
                rec[4] = sp_vals[i]
                rec[5] = prompt_tokens_i
                rec[6] = tool_seq

        total += tbl.num_rows

    if not agg:
        raise SystemExit("no rows after filtering; check --mix_group and input data")

    # Select buckets (composite buckets when candidates_dir is set).
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

    # Emit selected rows (best per bucket), enforcing per-tool-seq quota if enabled.
    out_rows: list[dict[str, Any]] = []
    ids_out: list[str] = []
    tool_seq_counts: Counter[str] = Counter()
    max_per_tool_seq = int(args.max_per_tool_seq)
    for k in selected_keys:
        rec = agg[int(k)]
        tool_seq = str(rec[6] or "none")
        if cand_dir is not None:
            if tool_seq_counts[tool_seq] >= max_per_tool_seq:
                continue
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
                "tool_seq": tool_seq,
            }
        )
        ids_out.append(str(rec[2]))
        if cand_dir is not None:
            tool_seq_counts[tool_seq] += 1

    # If tool_seq quota caused underfill, backfill from remaining items (largest buckets first).
    if cand_dir is not None and len(ids_out) < target_k:
        need = target_k - len(ids_out)
        # First pass: prefer unseen tool sequences to increase coverage.
        want_unique = int(args.min_unique_tool_seq)
        for k, _cnt in items:
            if need <= 0:
                break
            if k in selected_keys:
                continue
            rec = agg[int(k)]
            tool_seq = str(rec[6] or "none")
            if tool_seq_counts[tool_seq] >= max_per_tool_seq:
                continue
            if want_unique > 0 and len(tool_seq_counts) < want_unique and tool_seq in tool_seq_counts:
                continue
            rid = str(rec[2])
            if not rid or rid in ids_out:
                continue
            out_rows.append(
                {
                    "bucket_key": int(k),
                    "bucket_count": int(rec[0]),
                    "score": float(rec[1]),
                    "id": rid,
                    "dataset": str(rec[3]),
                    "split": str(rec[4]),
                    "prompt_tokens": int(rec[5]),
                    "mix_group": str(args.mix_group),
                    "tool_seq": tool_seq,
                }
            )
            ids_out.append(rid)
            tool_seq_counts[tool_seq] += 1
            need -= 1

        # Second pass: allow already-seen tool sequences to fill remaining.
        if need > 0:
            for k, _cnt in items:
                if need <= 0:
                    break
                if k in selected_keys:
                    continue
                rec = agg[int(k)]
                tool_seq = str(rec[6] or "none")
                if tool_seq_counts[tool_seq] >= max_per_tool_seq:
                    continue
                rid = str(rec[2])
                if not rid or rid in ids_out:
                    continue
                out_rows.append(
                    {
                        "bucket_key": int(k),
                        "bucket_count": int(rec[0]),
                        "score": float(rec[1]),
                        "id": rid,
                        "dataset": str(rec[3]),
                        "split": str(rec[4]),
                        "prompt_tokens": int(rec[5]),
                        "mix_group": str(args.mix_group),
                        "tool_seq": tool_seq,
                    }
                )
                ids_out.append(rid)
                tool_seq_counts[tool_seq] += 1
                need -= 1

    # Dedup (should already be unique).
    ids_unique = list(dict.fromkeys([x for x in ids_out if x]))
    if len(ids_unique) != len(ids_out):
        raise RuntimeError(f"duplicate ids detected: {len(ids_out)} -> {len(ids_unique)}")
    if len(ids_unique) != target_k:
        raise RuntimeError(f"selection size mismatch: got {len(ids_unique)} expected {target_k}")

    name = (args.name or "").strip()
    if not name:
        if meta_domain:
            name = f"{args.mix_group}_{meta_domain}"
        else:
            name = str(args.mix_group)
    safe_name = "".join(c if (c.isalnum() or c in {"_", "-"}) else "_" for c in name) or "selection"

    out_ids_path = out_dir / f"{safe_name}_ids_{len(ids_unique)}.txt"
    out_meta_path = out_dir / f"{safe_name}_selected_meta.parquet"
    out_report_path = out_dir / f"{safe_name}_selection_report.md"
    out_manifest_path = out_dir / "selection_manifest.json"

    out_ids_path.write_text("\n".join(ids_unique) + "\n", encoding="utf-8")
    pq.write_table(pa.Table.from_pylist(out_rows), out_meta_path, compression="zstd")

    ds_counts = Counter([r["dataset"] for r in out_rows])
    top_ds = "\n".join([f"- {k}: `{v}`" for k, v in ds_counts.most_common(25)])
    tool_seq_lines = ""
    if cand_dir is not None:
        tool_seq_lines = "\n".join([f"- {k}: `{v}`" for k, v in tool_seq_counts.most_common(25)])
    report = [
        "# LSH Bucket Selection Report",
        f"- generated_at: `{_now()}`",
        f"- in_dir: `{in_dir}`",
        f"- parquet_files: `{len(parquet_files)}`",
        f"- name: `{safe_name}`",
        f"- mix_group: `{args.mix_group}`",
        f"- meta_domain: `{meta_domain or ''}`",
        f"- candidates_dir: `{str(cand_dir) if cand_dir is not None else ''}`",
        f"- tool_seq_quota: `{max_per_tool_seq if cand_dir is not None else ''}`",
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
        "## Top Tool Sequences (selected)",
        tool_seq_lines or "- (n/a)",
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
