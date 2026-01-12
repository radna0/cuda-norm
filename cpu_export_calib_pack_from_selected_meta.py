#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def _dataset_dir_name(dataset: str) -> str:
    # normalized/<dataset_tag>/... uses / replaced by __
    return (dataset or "").replace("/", "__")


@dataclass(frozen=True)
class Task:
    task_index: int
    parquet_path: str
    ids: tuple[str, ...]


def _process_task(task: Task, *, out_tmp: Path, id_col: str, text_col: str) -> dict[str, Any]:
    pf = Path(task.parquet_path)
    ids_arr = pa.array(list(task.ids), type=pa.string())
    parquet = pq.ParquetFile(pf)
    cols = set(parquet.schema_arrow.names)
    want_cols = [
        id_col,
        text_col,
        "dataset",
        "split",
        "meta_domain",
        "meta_difficulty_bin",
        "meta_correctness",
        "quality_has_tool",
        "quality_valid_tool_schema",
    ]
    read_cols = [c for c in want_cols if c in cols]
    table = parquet.read(columns=read_cols)
    if id_col not in table.column_names:
        raise RuntimeError(f"missing required column {id_col!r} in {pf}")
    if text_col not in table.column_names:
        raise RuntimeError(f"missing required column {text_col!r} in {pf}")

    mask = pc.is_in(table[id_col], value_set=ids_arr)
    table = table.filter(mask)
    if table.num_rows == 0:
        return {"task_index": task.task_index, "parquet_path": str(pf), "rows_out": 0, "tmp_path": ""}

    tmp_path = out_tmp / f"match-{task.task_index:06d}.parquet"
    pq.write_table(table, tmp_path, compression="zstd")
    return {
        "task_index": task.task_index,
        "parquet_path": str(pf),
        "rows_out": int(table.num_rows),
        "tmp_path": str(tmp_path),
    }


def _process_task_star(args: tuple[Any, ...]) -> dict[str, Any]:
    task = args[0]
    kwargs = args[1]
    return _process_task(task, **kwargs)  # type: ignore[arg-type]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export a Harmony-text calibration pack by joining selected ids back to normalized shards"
    )
    ap.add_argument(
        "--normalized_root",
        type=str,
        required=True,
        help="Path to normalized/ (contains <dataset_tag>/data/<split>/*.parquet)",
    )
    ap.add_argument(
        "--selected_meta",
        type=str,
        required=True,
        help="Parquet file containing at least columns: id, dataset, split",
    )
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--pack_name", type=str, default="tool_agentic_10k")
    ap.add_argument("--id_col", type=str, default="id")
    ap.add_argument("--text_col", type=str, default="text")
    ap.add_argument("--num_workers", type=int, default=0, help="0=auto")
    ap.add_argument("--maxtasksperchild", type=int, default=16)
    args = ap.parse_args()

    normalized_root = Path(args.normalized_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tmp = out_dir / "_tmp_matches"
    out_tmp.mkdir(parents=True, exist_ok=True)

    selected = pq.read_table(args.selected_meta, columns=["id", "dataset", "split"])
    ids = [str(x or "") for x in selected["id"].to_pylist()]
    dss = [str(x or "") for x in selected["dataset"].to_pylist()]
    sps = [str(x or "") for x in selected["split"].to_pylist()]
    wanted_ids = [i for i in ids if i]
    wanted_set = set(wanted_ids)
    if not wanted_set:
        raise SystemExit("no ids in selected_meta")

    # Group ids by dataset/split to reduce scan scope.
    group_map: dict[tuple[str, str], set[str]] = {}
    for rid, d, sp in zip(ids, dss, sps):
        rid = rid.strip()
        d = d.strip()
        sp = sp.strip()
        if not rid or not d or not sp:
            continue
        key = (d, sp)
        group_map.setdefault(key, set()).add(rid)

    tasks: list[Task] = []
    task_i = 0
    for (dataset_name, split_name), idset in sorted(group_map.items(), key=lambda x: (x[0][0], x[0][1])):
        ds_tag = _dataset_dir_name(dataset_name)
        base = normalized_root / ds_tag / "data" / split_name
        parquet_files = sorted(base.rglob("*.parquet"))
        if not parquet_files:
            raise SystemExit(f"no parquet files for dataset={dataset_name} split={split_name} under {base}")
        id_list = sorted(idset)
        for pf in parquet_files:
            tasks.append(Task(task_index=task_i, parquet_path=str(pf), ids=tuple(id_list)))
            task_i += 1

    num_workers = int(args.num_workers) if args.num_workers else max(1, min(64, os.cpu_count() or 1))
    t0 = time.time()
    print(f"[*] exporting {len(wanted_set)} ids across {len(tasks)} files using workers={num_workers}", flush=True)

    ctx = mp.get_context("fork")
    kwargs = {"out_tmp": out_tmp, "id_col": args.id_col, "text_col": args.text_col}
    results: list[dict[str, Any]] = []
    with ctx.Pool(processes=num_workers, maxtasksperchild=int(args.maxtasksperchild)) as pool:
        for r in pool.imap_unordered(_process_task_star, [(t, kwargs) for t in tasks], chunksize=4):
            results.append(r)
            if len(results) % 200 == 0:
                done = sum(1 for x in results if x.get("tmp_path"))
                print(f"[prog] tasks={len(results)}/{len(tasks)} nonempty={done}", flush=True)

    tmp_files = [Path(r["tmp_path"]) for r in results if r.get("tmp_path")]
    if not tmp_files:
        raise SystemExit("no matches found; check ids and normalized_root")

    tables: list[pa.Table] = []
    for p in sorted(tmp_files):
        tables.append(pq.read_table(p))
    merged = pa.concat_tables(tables) if len(tables) > 1 else tables[0]

    # Dedup by id (safety). Keep first occurrence.
    ids_col = merged[args.id_col].to_pylist()
    keep_idx: list[int] = []
    seen: set[str] = set()
    for i, rid in enumerate(ids_col):
        s = str(rid or "")
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        keep_idx.append(i)
    merged = merged.take(pa.array(keep_idx, type=pa.int32()))

    found_ids = set(str(x or "") for x in merged[args.id_col].to_pylist())
    missing = sorted(wanted_set - found_ids)
    if missing:
        raise SystemExit(f"missing {len(missing)}/{len(wanted_set)} ids in export (example: {missing[:5]})")

    out_parquet = out_dir / f"{args.pack_name}.parquet"
    pq.write_table(merged, out_parquet, compression="zstd")

    out_jsonl = out_dir / f"{args.pack_name}.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        texts = merged[args.text_col].to_pylist()
        for t in texts:
            s = t if isinstance(t, str) else ""
            if not s:
                continue
            f.write(json.dumps({"text": s}, ensure_ascii=False) + "\n")

    manifest = {
        "generated_at": _now(),
        "normalized_root": str(normalized_root),
        "selected_meta": str(Path(args.selected_meta).resolve()),
        "pack_name": args.pack_name,
        "wanted_ids": int(len(wanted_set)),
        "rows_out": int(merged.num_rows),
        "out_parquet": str(out_parquet),
        "out_jsonl": str(out_jsonl),
        "elapsed_s": time.time() - t0,
        "workers": int(num_workers),
    }
    (out_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[ok] wrote {out_parquet}", flush=True)
    print(f"[ok] wrote {out_jsonl}", flush=True)
    print(f"[ok] wrote {out_dir / 'export_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
