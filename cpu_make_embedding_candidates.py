#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from harmony_text import (
    build_behavior_signature,
    extract_assistant_reasoning_excerpt_text,
    extract_user_prompt_text,
)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def _fill_null_false(mask: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    return pc.fill_null(mask, False)


def _word_count(text: str) -> int:
    if not text:
        return 0
    return len([w for w in text.split() if w])


def _stable_hash_int(hex_sha1: str) -> int:
    # Deterministic sampling hash for ids.
    #
    # IMPORTANT: do NOT use `int(id[:8], 16)` here. Even when `id` looks like sha1 hex, upstream
    # selection steps may have already biased the distribution of the leading digits, which makes
    # modulo sampling non-uniform. Hashing the full id string avoids that bias.
    s = (hex_sha1 or "").strip().lower()
    if not s:
        return 0
    return int(sha1(s.encode("utf-8")).hexdigest()[:8], 16)


_IDS_SET: set[str] | None = None
_MIX_GROUP_MAP: dict[str, str] | None = None

# Tokenizer is initialized in worker processes (when enabled).
_TOKENIZER: Any | None = None
_TOKENIZER_MAX_TOKENS: int = 0
_TOKENIZER_ADD_SPECIAL_TOKENS: bool = False


def _load_mix_group_map(path: Path) -> dict[str, str]:
    if not path.exists():
        raise SystemExit(f"--mix_group_map does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        table = pq.read_table(path, columns=["id", "mix_group"])
        ids = table["id"].to_pylist()
        groups = table["mix_group"].to_pylist()
        out: dict[str, str] = {}
        for i, g in zip(ids, groups):
            k = str(i or "")
            if not k:
                continue
            out[k] = str(g or "")
        return out
    # Default: TSV/CSV-ish "id<TAB>mix_group" or "id,mix_group" lines.
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "\t" in s:
            k, v = s.split("\t", 1)
        elif "," in s:
            k, v = s.split(",", 1)
        else:
            continue
        k = k.strip()
        v = v.strip()
        if k:
            out[k] = v
    return out


def _snapshot_tokenizer(model_id: str, *, local_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    allow = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "config.json",
    ]
    snapshot_download(
        repo_id=model_id,
        repo_type="model",
        allow_patterns=allow,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    return local_dir


def _init_worker_tokenizer(
    tokenizer_path: str,
    trust_remote_code: bool,
    add_special_tokens: bool,
    max_tokens: int,
) -> None:
    global _TOKENIZER, _TOKENIZER_MAX_TOKENS, _TOKENIZER_ADD_SPECIAL_TOKENS
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # We use tokenizers only (no torch), so suppress the noisy framework advisory warning.
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    _TOKENIZER = tok
    _TOKENIZER_MAX_TOKENS = int(max_tokens)
    _TOKENIZER_ADD_SPECIAL_TOKENS = bool(add_special_tokens)


@dataclass(frozen=True)
class Task:
    task_index: int
    parquet_path: str
    row_groups: tuple[int, ...]


def _segment_row_groups(path: Path, *, target_rows: int) -> list[tuple[int, ...]]:
    parquet = pq.ParquetFile(path)
    row_groups: list[tuple[int, ...]] = []
    current: list[int] = []
    current_rows = 0
    for rg in range(parquet.num_row_groups):
        n = parquet.metadata.row_group(rg).num_rows
        if current and (current_rows + n) > target_rows:
            row_groups.append(tuple(current))
            current = []
            current_rows = 0
        current.append(rg)
        current_rows += n
    if current:
        row_groups.append(tuple(current))
    return row_groups


def _process_task(
    task: Task,
    *,
    out_dir: Path,
    view: str,
    text_column: str,
    require_valid_harmony: bool,
    require_completion_nonempty: bool,
    require_valid_tool_schema: bool,
    require_has_tool: bool,
    require_domains: set[str] | None,
    require_difficulty_bins: set[str] | None,
    require_mix_groups: set[str] | None,
    correctness_ge: float | None,
    hash_mod: int,
    hash_keep_lt: int,
    include_source_text: bool,
    attach_mix_group: bool,
    fail_on_missing_mix_group: bool,
    tokenize: bool,
    min_tok_len: int | None,
    max_tok_len: int | None,
) -> dict[str, Any]:
    t0 = time.time()
    pf = Path(task.parquet_path)
    parquet = pq.ParquetFile(pf)

    want_cols = {
        "id",
        text_column,
        "dataset",
        "split",
        "meta_domain",
        "meta_difficulty_bin",
        "meta_correctness",
        "quality_valid_harmony",
        "quality_completion_nonempty",
        "quality_has_tool",
        "quality_valid_tool_schema",
    }
    cols = set(parquet.schema.names)
    read_cols = [c for c in want_cols if c in cols]
    if "id" not in read_cols:
        raise RuntimeError(f"missing required column 'id' in {pf}")
    if text_column not in read_cols:
        raise RuntimeError(f"missing required text column {text_column!r} in {pf}")

    parts: list[pa.Table] = []
    for rg in task.row_groups:
        parts.append(parquet.read_row_group(rg, columns=read_cols))
    table = pa.concat_tables(parts) if len(parts) > 1 else parts[0]

    # Vectorized filters (cheap, avoids parsing Harmony for rows we will drop anyway).
    mask = None
    if require_valid_harmony and "quality_valid_harmony" in table.column_names:
        m = _fill_null_false(table["quality_valid_harmony"])
        mask = m if mask is None else pc.and_(mask, m)
    if require_completion_nonempty and "quality_completion_nonempty" in table.column_names:
        m = _fill_null_false(table["quality_completion_nonempty"])
        mask = m if mask is None else pc.and_(mask, m)
    if require_valid_tool_schema and "quality_valid_tool_schema" in table.column_names:
        m = _fill_null_false(table["quality_valid_tool_schema"])
        mask = m if mask is None else pc.and_(mask, m)
    if require_has_tool and "quality_has_tool" in table.column_names:
        m = _fill_null_false(table["quality_has_tool"])
        mask = m if mask is None else pc.and_(mask, m)
    if require_domains and "meta_domain" in table.column_names:
        dom = table["meta_domain"]
        m = pc.is_in(dom, value_set=pa.array(sorted(require_domains), type=pa.string()))
        mask = m if mask is None else pc.and_(mask, m)
    if require_difficulty_bins and "meta_difficulty_bin" in table.column_names:
        db = table["meta_difficulty_bin"]
        m = pc.is_in(db, value_set=pa.array(sorted(require_difficulty_bins), type=pa.string()))
        mask = m if mask is None else pc.and_(mask, m)
    if correctness_ge is not None and "meta_correctness" in table.column_names:
        mc = table["meta_correctness"]
        m = _fill_null_false(pc.greater_equal(mc, correctness_ge))
        mask = m if mask is None else pc.and_(mask, m)

    if mask is not None:
        table = table.filter(mask)

    ids = table["id"].to_pylist()
    texts = table[text_column].to_pylist()
    keep_ids: list[str] = []
    embed_texts: list[str] = []
    word_counts: list[int] = []
    keep_row_indices: list[int] = []
    mix_groups: list[str] = []

    kept = 0
    dropped = 0
    ids_set = _IDS_SET
    mix_map = _MIX_GROUP_MAP
    need_mix = bool(attach_mix_group or require_mix_groups)
    if need_mix and mix_map is None:
        raise RuntimeError("mix_group filtering requested but mix_group_map is not loaded")

    for i, (row_id, raw_text) in enumerate(zip(ids, texts)):
        row_id = str(row_id or "")
        if not row_id:
            dropped += 1
            continue
        if ids_set is not None and row_id not in ids_set:
            dropped += 1
            continue

        mg = ""
        if need_mix:
            assert mix_map is not None
            mg = str(mix_map.get(row_id, "") or "")
            if not mg:
                dropped += 1
                if fail_on_missing_mix_group:
                    raise RuntimeError(f"missing mix_group for id={row_id}")
                continue
            if require_mix_groups is not None and mg not in require_mix_groups:
                dropped += 1
                continue

        # Apply sampling only after cheap membership filters (ids_set/mix_group), so the sampling
        # rate is stable and we don't waste hashing work on rows that will be dropped anyway.
        if hash_mod > 0:
            if _stable_hash_int(row_id) % hash_mod >= hash_keep_lt:
                dropped += 1
                continue

        raw_text = raw_text if isinstance(raw_text, str) else ""
        if view == "prompt":
            embed = extract_user_prompt_text(raw_text)
        elif view == "behavior":
            embed = build_behavior_signature(raw_text)
        elif view == "reasoning_excerpt":
            embed = extract_assistant_reasoning_excerpt_text(raw_text)
        else:
            raise RuntimeError(f"unsupported view={view!r}")
        embed = (embed or "").strip()
        if not embed:
            dropped += 1
            continue

        keep_ids.append(row_id)
        embed_texts.append(embed)
        word_counts.append(_word_count(embed))
        keep_row_indices.append(i)
        if attach_mix_group:
            mix_groups.append(mg)
        kept += 1

    # Avoid creating empty Parquet files (huge overhead at scale when filtering by IDs).
    if kept == 0:
        return {
            "task_index": task.task_index,
            "parquet_path": task.parquet_path,
            "row_groups": list(task.row_groups),
            "rows_in": len(ids),
            "rows_out": 0,
            "rows_dropped": dropped,
            "out_path": "",
            "elapsed_s": time.time() - t0,
            "wrote_file": False,
        }

    input_ids: list[list[int]] = []
    tok_lens: list[int] = []
    if tokenize:
        tok = _TOKENIZER
        if tok is None:
            raise RuntimeError("tokenize requested but tokenizer is not initialized")
        enc = tok(
            embed_texts,
            add_special_tokens=_TOKENIZER_ADD_SPECIAL_TOKENS,
            truncation=True,
            max_length=_TOKENIZER_MAX_TOKENS if _TOKENIZER_MAX_TOKENS > 0 else None,
            padding=False,
            return_attention_mask=False,
        )
        input_ids = enc.get("input_ids") or []
        if len(input_ids) != len(embed_texts):
            raise RuntimeError(
                f"tokenizer returned {len(input_ids)} sequences for {len(embed_texts)} texts"
            )
        # Normalize types and compute lengths.
        for seq in input_ids:
            seq_i = [int(x) for x in (seq or [])]
            tok_lens.append(len(seq_i))
        input_ids = [[int(x) for x in (seq or [])] for seq in input_ids]

    # Optional: enforce token-length bounds (requires tokenization).
    if tokenize and (min_tok_len is not None or max_tok_len is not None):
        lo = int(min_tok_len) if min_tok_len is not None else 0
        hi = int(max_tok_len) if max_tok_len is not None else 0
        if lo < 0:
            lo = 0
        if hi and hi < lo:
            raise RuntimeError(f"invalid tok_len bounds: min_tok_len={lo} max_tok_len={hi}")

        keep_mask: list[bool] = []
        for tl in tok_lens:
            if tl < lo:
                keep_mask.append(False)
            elif hi and tl > hi:
                keep_mask.append(False)
            else:
                keep_mask.append(True)

        if not any(keep_mask):
            return {
                "task_index": task.task_index,
                "parquet_path": task.parquet_path,
                "row_groups": list(task.row_groups),
                "rows_in": len(ids),
                "rows_out": 0,
                "rows_dropped": len(ids),
                "out_path": "",
                "elapsed_s": time.time() - t0,
                "wrote_file": False,
            }

        def _filter_list(xs: list[Any]) -> list[Any]:
            return [x for x, keep in zip(xs, keep_mask) if keep]

        keep_ids = [str(x) for x in _filter_list(keep_ids)]
        embed_texts = [str(x) for x in _filter_list(embed_texts)]
        word_counts = [int(x) for x in _filter_list(word_counts)]
        keep_row_indices = [int(x) for x in _filter_list(keep_row_indices)]
        input_ids = [[int(y) for y in (x or [])] for x in _filter_list(input_ids)]
        tok_lens = [int(x) for x in _filter_list(tok_lens)]
        if attach_mix_group:
            mix_groups = [str(x) for x in _filter_list(mix_groups)]

        kept = len(keep_ids)

    out_path = out_dir / f"part-{task.task_index:06d}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_cols: dict[str, Any] = {
        "id": pa.array(keep_ids, type=pa.string()),
        "embed_text": pa.array(embed_texts, type=pa.string()),
        "stats_embed_word_count": pa.array(word_counts, type=pa.int32()),
    }
    if attach_mix_group:
        out_cols["mix_group"] = pa.array(mix_groups, type=pa.string())
    if tokenize:
        out_cols["input_ids"] = pa.array(input_ids, type=pa.list_(pa.int32()))
        out_cols["stats_tok_len"] = pa.array(tok_lens, type=pa.int32())
    # Passthrough useful metadata columns, excluding the large Harmony `text` column.
    if keep_row_indices:
        idx = pa.array(keep_row_indices, type=pa.int32())
        kept_table = table.take(idx)
        if include_source_text:
            out_cols["text"] = kept_table[text_column]
        # Keep a stable schema across shards (fill missing columns with nulls).
        passthrough_schema: dict[str, pa.DataType] = {
            "dataset": pa.string(),
            "split": pa.string(),
            "meta_domain": pa.string(),
            "meta_difficulty_bin": pa.string(),
            "meta_correctness": pa.float64(),
            "quality_has_tool": pa.bool_(),
            "quality_valid_tool_schema": pa.bool_(),
        }
        for c, dtype in passthrough_schema.items():
            if c in kept_table.column_names:
                out_cols[c] = kept_table[c].cast(dtype)
            else:
                out_cols[c] = pa.nulls(len(keep_ids), type=dtype)
    pq.write_table(pa.table(out_cols), out_path, compression="zstd")

    return {
        "task_index": task.task_index,
        "parquet_path": task.parquet_path,
        "row_groups": list(task.row_groups),
        "rows_in": len(ids),
        "rows_out": kept,
        "rows_dropped": dropped,
        "out_path": str(out_path),
        "elapsed_s": time.time() - t0,
        "wrote_file": True,
    }


def _process_task_star(args: tuple[Any, ...]) -> dict[str, Any]:
    task = args[0]
    kwargs = args[1]
    return _process_task(task, **kwargs)  # type: ignore[arg-type]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build embedding candidate Parquet shards from normalized Harmony shards"
    )
    ap.add_argument("--in_dir", type=str, required=True, help="Directory containing normalized *.parquet")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")
    ap.add_argument(
        "--view",
        type=str,
        choices=["prompt", "behavior", "reasoning_excerpt"],
        required=True,
        help="Embedding view: prompt=user-only, behavior=tool trace sketch, reasoning_excerpt=assistant reasoning excerpt",
    )
    ap.add_argument("--text_column", type=str, default="text", help="Input column containing Harmony text")
    ap.add_argument("--target_rows_per_task", type=int, default=100_000)
    ap.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Parallel workers (0=auto)",
    )
    ap.add_argument(
        "--maxtasksperchild",
        type=int,
        default=8,
        help="Recycle worker processes after N tasks (0=disable). Higher is faster; lower is safer for memory leaks.",
    )
    ap.add_argument("--max_tasks", type=int, default=0, help="Debug: cap number of tasks (0=all)")
    ap.add_argument(
        "--ids_file",
        type=str,
        default="",
        help="Optional newline-delimited IDs file; when set, keep only these IDs (disables hash sampling).",
    )
    ap.add_argument("--require_valid_harmony", action="store_true")
    ap.add_argument("--require_completion_nonempty", action="store_true")
    ap.add_argument("--require_valid_tool_schema", action="store_true")
    ap.add_argument("--require_has_tool", action="store_true")
    ap.add_argument("--require_domain", nargs="*", default=[])
    ap.add_argument("--require_difficulty_bin", nargs="*", default=[])
    ap.add_argument("--correctness_ge", type=float, default=float("nan"))
    ap.add_argument("--hash_mod", type=int, default=0, help="Deterministic sampling modulus (0=off)")
    ap.add_argument(
        "--hash_keep_lt",
        type=int,
        default=0,
        help="Keep rows where (hash(id) %% mod) < this",
    )
    ap.add_argument(
        "--include_source_text",
        action="store_true",
        help="Include the original Harmony `text` column in output shards (useful for later view changes; larger output).",
    )
    ap.add_argument(
        "--mix_group_map",
        type=str,
        default="",
        help="Optional mapping file to attach mix_group (parquet with columns id,mix_group or TSV id<TAB>mix_group).",
    )
    ap.add_argument(
        "--require_mix_group",
        nargs="*",
        default=[],
        help="Optional filter on mix_group values (requires --mix_group_map). Example: --require_mix_group reasoning",
    )
    ap.add_argument(
        "--fail_on_missing_mix_group",
        action="store_true",
        help="When --mix_group_map is set, abort if any kept id is missing a mix_group entry.",
    )
    ap.add_argument(
        "--tokenize_model_id",
        type=str,
        default="",
        help="Optional HF model_id to pretokenize embed_text into input_ids (CPU-side).",
    )
    ap.add_argument(
        "--tokenize_model_path",
        type=str,
        default="",
        help="Optional local tokenizer directory (overrides --tokenize_model_id).",
    )
    ap.add_argument(
        "--tokenize_max_tokens",
        type=int,
        default=0,
        help="Truncate tokenized input_ids to this length (required when tokenization is enabled).",
    )
    ap.add_argument(
        "--min_tok_len",
        type=int,
        default=0,
        help="Optional: drop rows with tokenized length < this (requires tokenization). 0=disabled.",
    )
    ap.add_argument(
        "--max_tok_len",
        type=int,
        default=0,
        help="Optional: drop rows with tokenized length > this (requires tokenization). 0=disabled.",
    )
    ap.add_argument(
        "--tokenize_add_special_tokens",
        action="store_true",
        help="Tokenizer option: include special tokens when pretokenizing.",
    )
    ap.add_argument(
        "--tokenize_trust_remote_code",
        action="store_true",
        help="Tokenizer option: trust_remote_code=True for AutoTokenizer.",
    )
    ap.add_argument(
        "--tokenizer_cache_dir",
        type=str,
        default="",
        help="Optional directory to snapshot tokenizer files into (defaults to <out_dir>/.tokenizer_cache/<model>).",
    )
    ap.add_argument("--manifest", type=str, default="candidates_manifest.json")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(in_dir.rglob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"no parquet files under {in_dir}")

    num_workers = int(args.num_workers)
    if num_workers == 0:
        num_workers = os.cpu_count() or 1
    num_workers = max(1, num_workers)
    if num_workers > 1:
        # Avoid thread oversubscription inside worker processes.
        pa.set_cpu_count(1)

    correctness_ge = args.correctness_ge
    if isinstance(correctness_ge, float) and math.isnan(correctness_ge):
        correctness_ge = None

    require_domains = set([d for d in (args.require_domain or []) if d])
    require_domains_set = require_domains if require_domains else None
    require_bins = set([d for d in (args.require_difficulty_bin or []) if d])
    require_bins_set = require_bins if require_bins else None

    ids_file = (args.ids_file or "").strip() or None
    if ids_file:
        # Load ID set in the parent before forking so children share pages (copy-on-write).
        global _IDS_SET
        ids_path = Path(ids_file)
        if not ids_path.exists():
            raise SystemExit(f"--ids_file does not exist: {ids_path}")
        ids: set[str] = set()
        for line in ids_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s:
                ids.add(s)
        _IDS_SET = ids
        print(f"[*] loaded ids_file: {ids_path} (n={len(ids)})", flush=True)

    mix_group_map_path = (args.mix_group_map or "").strip() or None
    if mix_group_map_path:
        global _MIX_GROUP_MAP
        mg_path = Path(mix_group_map_path)
        mg = _load_mix_group_map(mg_path)
        _MIX_GROUP_MAP = mg
        print(f"[*] loaded mix_group_map: {mg_path} (n={len(mg)})", flush=True)

    require_mix_groups = set([str(x) for x in (args.require_mix_group or []) if str(x).strip()])
    require_mix_groups_set = require_mix_groups if require_mix_groups else None
    if require_mix_groups_set and not mix_group_map_path:
        raise SystemExit("--require_mix_group requires --mix_group_map")

    tokenize_model_path = (args.tokenize_model_path or "").strip() or None
    tokenize_model_id = (args.tokenize_model_id or "").strip() or None
    tokenize_enabled = bool(tokenize_model_path or tokenize_model_id)
    if tokenize_enabled:
        max_toks = int(args.tokenize_max_tokens)
        if max_toks <= 0:
            raise SystemExit("--tokenize_max_tokens must be > 0 when tokenization is enabled")
        if tokenize_model_path:
            tok_source = Path(tokenize_model_path)
            if not tok_source.exists():
                raise SystemExit(f"--tokenize_model_path does not exist: {tok_source}")
            tok_path = str(tok_source)
        else:
            assert tokenize_model_id
            cache_dir = (args.tokenizer_cache_dir or "").strip()
            if cache_dir:
                tok_dir = Path(cache_dir)
            else:
                tok_dir = out_dir / ".tokenizer_cache" / tokenize_model_id.replace("/", "__")
            tok_path = str(_snapshot_tokenizer(tokenize_model_id, local_dir=tok_dir))
    else:
        if int(args.min_tok_len) > 0 or int(args.max_tok_len) > 0:
            raise SystemExit("--min_tok_len/--max_tok_len require tokenization (set --tokenize_model_id or --tokenize_model_path)")

    hash_mod = int(args.hash_mod)
    hash_keep_lt = int(args.hash_keep_lt)
    if ids_file:
        hash_mod = 0
        hash_keep_lt = 0
    elif hash_mod:
        if hash_keep_lt <= 0 or hash_keep_lt > hash_mod:
            raise SystemExit("--hash_keep_lt must be in [1, hash_mod] when --hash_mod is set")
    else:
        hash_keep_lt = 0

    tasks: list[Task] = []
    task_index = 0
    for pf in parquet_files:
        segments = _segment_row_groups(pf, target_rows=int(args.target_rows_per_task))
        for rgs in segments:
            tasks.append(Task(task_index=task_index, parquet_path=str(pf), row_groups=rgs))
            task_index += 1
            if args.max_tasks and len(tasks) >= args.max_tasks:
                break
        if args.max_tasks and len(tasks) >= args.max_tasks:
            break

    print(
        f"[*] building candidates view={args.view} files={len(parquet_files)} tasks={len(tasks)} workers={num_workers} at {_now()}",
        flush=True,
    )

    t0 = time.time()
    results: list[dict[str, Any]] = []
    total_out = 0
    total_in = 0
    ctx = mp.get_context("fork")
    mtpc = int(args.maxtasksperchild)
    pool_init = None
    pool_init_args: tuple[Any, ...] = tuple()
    if tokenize_enabled:
        pool_init = _init_worker_tokenizer
        pool_init_args = (
            tok_path,
            bool(args.tokenize_trust_remote_code),
            bool(args.tokenize_add_special_tokens),
            int(args.tokenize_max_tokens),
        )
    with ctx.Pool(
        processes=num_workers,
        maxtasksperchild=(mtpc if mtpc > 0 else None),
        initializer=pool_init,
        initargs=pool_init_args,
    ) as pool:
        common_kwargs = {
            "out_dir": out_dir,
            "view": args.view,
            "text_column": args.text_column,
            "require_valid_harmony": bool(args.require_valid_harmony),
            "require_completion_nonempty": bool(args.require_completion_nonempty),
            "require_valid_tool_schema": bool(args.require_valid_tool_schema),
            "require_has_tool": bool(args.require_has_tool),
            "require_domains": require_domains_set,
            "require_difficulty_bins": require_bins_set,
            "require_mix_groups": require_mix_groups_set,
            "correctness_ge": correctness_ge,
            "hash_mod": hash_mod,
            "hash_keep_lt": hash_keep_lt,
            "include_source_text": bool(args.include_source_text),
            "attach_mix_group": bool(mix_group_map_path),
            "fail_on_missing_mix_group": bool(args.fail_on_missing_mix_group),
            "tokenize": bool(tokenize_enabled),
            "min_tok_len": (int(args.min_tok_len) if int(args.min_tok_len) > 0 else None),
            "max_tok_len": (int(args.max_tok_len) if int(args.max_tok_len) > 0 else None),
        }
        it = ((t, common_kwargs) for t in tasks)
        for res in pool.imap_unordered(_process_task_star, it, chunksize=1):
            results.append(res)
            total_out += int(res.get("rows_out") or 0)
            total_in += int(res.get("rows_in") or 0)
            if len(results) % max(1, min(100, len(tasks))) == 0:
                dt = time.time() - t0
                print(
                    f"[prog] tasks_done={len(results)}/{len(tasks)} rows_out={total_out} rows_in={total_in} dt={dt:.1f}s",
                    flush=True,
                )

    manifest = {
        "generated_at": _now(),
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "view": args.view,
        "text_column": args.text_column,
        "num_workers": num_workers,
        "target_rows_per_task": int(args.target_rows_per_task),
        "filters": {
            "require_valid_harmony": bool(args.require_valid_harmony),
            "require_completion_nonempty": bool(args.require_completion_nonempty),
            "require_valid_tool_schema": bool(args.require_valid_tool_schema),
            "require_has_tool": bool(args.require_has_tool),
            "require_domain": sorted(require_domains),
            "require_difficulty_bin": sorted(require_bins),
            "require_mix_group": sorted(require_mix_groups),
            "correctness_ge": correctness_ge,
            "ids_file": str(ids_file) if ids_file else "",
            "hash_mod": hash_mod,
            "hash_keep_lt": hash_keep_lt,
            "include_source_text": bool(args.include_source_text),
            "mix_group_map": str(mix_group_map_path) if mix_group_map_path else "",
            "tokenize_model_id": tokenize_model_id or "",
            "tokenize_model_path": tokenize_model_path or "",
            "tokenize_max_tokens": int(args.tokenize_max_tokens),
            "min_tok_len": int(args.min_tok_len),
            "max_tok_len": int(args.max_tok_len),
            "tokenize_add_special_tokens": bool(args.tokenize_add_special_tokens),
            "tokenize_trust_remote_code": bool(args.tokenize_trust_remote_code),
        },
        "tasks": results,
        "totals": {
            "rows_in": total_in,
            "rows_out": total_out,
            "rows_dropped": max(0, total_in - total_out),
            "elapsed_s": time.time() - t0,
        },
    }
    (out_dir / args.manifest).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[ok] wrote {out_dir}/{args.manifest}", flush=True)
    print(f"[done] rows_out={total_out} dt={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
