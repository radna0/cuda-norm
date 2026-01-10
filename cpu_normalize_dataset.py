#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import time
from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
import requests
import yaml
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url

from harmony_text import basic_quality_flags_from_text, render_harmony, sha1_text_id


NORMALIZED_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("dataset", pa.string()),
        pa.field("split", pa.string()),
        pa.field("text", pa.string()),
        pa.field("loss_mode", pa.string()),
        pa.field("quality_valid_harmony", pa.bool_()),
        pa.field("quality_completion_nonempty", pa.bool_()),
        pa.field("quality_has_tool", pa.bool_()),
        pa.field("quality_valid_tool_schema", pa.bool_()),
        pa.field("stats_char_len", pa.int64()),
        pa.field("meta_domain", pa.string()),
        pa.field("meta_difficulty_bin", pa.string()),
        pa.field("source_uuid", pa.string()),
        pa.field("source_license", pa.string()),
        pa.field("source_used_in", pa.list_(pa.string())),
        pa.field("source_data_source", pa.string()),
        pa.field("source_changed_answer_to_majority", pa.bool_()),
        pa.field("source_capability_target", pa.string()),
        pa.field("source_reasoning", pa.string()),
        pa.field("source_tools_id", pa.string()),
        pa.field("source_tools_count", pa.int64()),
        pa.field("meta_correctness", pa.float64()),
        pa.field("meta_eval_best_setting", pa.string()),
        pa.field("meta_pass_k", pa.int64()),
        pa.field("meta_correct_count", pa.int64()),
        pa.field("meta_correctness_low", pa.float64()),
        pa.field("meta_correctness_medium", pa.float64()),
        pa.field("meta_correctness_high", pa.float64()),
        pa.field("meta_correctness_high_with_tool", pa.float64()),
        pa.field("meta_correctness_high_no_tool", pa.float64()),
    ]
)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _sha1_json(value: Any) -> str:
    s = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha1(s.encode("utf-8")).hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def _parse_frontmatter(readme_text: str) -> dict[str, Any]:
    lines = readme_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return {}
    fm_text = "\n".join(lines[1:end])
    try:
        data = yaml.safe_load(fm_text)
        return data or {}
    except Exception:
        return {}


def _split_to_path_from_frontmatter(frontmatter: dict[str, Any]) -> dict[str, str]:
    configs = frontmatter.get("configs") or []
    if not isinstance(configs, list):
        return {}
    for cfg in configs:
        if not isinstance(cfg, dict):
            continue
        if (cfg.get("config_name") or "default") != "default":
            continue
        data_files = cfg.get("data_files") or []
        if not isinstance(data_files, list):
            continue
        mapping: dict[str, str] = {}
        for entry in data_files:
            if not isinstance(entry, dict):
                continue
            split = entry.get("split")
            path = entry.get("path")
            if isinstance(split, str) and isinstance(path, str):
                mapping[split] = path
        if mapping:
            return mapping
    return {}


def _infer_split_from_data_path(path: str) -> str | None:
    name = Path(path).name
    if name.endswith(".parquet"):
        # e.g. low-00000-of-00017.parquet, high_part01-00000-of-00073.parquet
        return name.split("-", 1)[0]

    if name.endswith(".jsonl"):
        stem = name[: -len(".jsonl")]
        m = re.match(r"^high\.part_(\d\d)$", stem)
        if m:
            return f"high_part{m.group(1)}"
        return stem

    if name.endswith(".jsonl.gz"):
        stem = name[: -len(".jsonl.gz")]
        return stem

    return None


def _list_data_files_by_split(dataset: str) -> dict[str, list[dict[str, Any]]]:
    api = HfApi()
    files = list(api.list_repo_tree(dataset, repo_type="dataset", path_in_repo="data"))
    by_split: dict[str, list[dict[str, Any]]] = {}
    for f in files:
        split = _infer_split_from_data_path(f.path)
        if not split:
            continue
        by_split.setdefault(split, []).append({"path": f.path, "size": getattr(f, "size", None)})
    for split in by_split:
        by_split[split].sort(key=lambda x: x["path"])
    return by_split


def _iter_jsonl_from_hub(dataset: str, *, path_in_repo: str, timeout_s: int) -> Iterator[dict[str, Any]]:
    url = hf_hub_url(repo_id=dataset, repo_type="dataset", filename=path_in_repo)
    headers = {"Accept-Encoding": "identity"}
    with requests.get(url, stream=True, timeout=timeout_s, headers=headers) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=False):
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if isinstance(rec, dict):
                yield rec


def _iter_parquet_text_from_hub(dataset: str, *, paths_in_repo: list[str]) -> Iterator[str]:
    for path_in_repo in paths_in_repo:
        local = hf_hub_download(repo_id=dataset, repo_type="dataset", filename=path_in_repo)
        parquet = pq.ParquetFile(local)
        for batch in parquet.iter_batches(columns=["text"], batch_size=8192):
            texts = batch.column(0).to_pylist()
            for t in texts:
                if isinstance(t, str):
                    yield t


def _difficulty_from_split(split: str) -> str | None:
    if split.startswith("low"):
        return "low"
    if split.startswith("medium"):
        return "medium"
    if split.startswith("high"):
        return "high"
    return None


def _domain_from_dataset(dataset: str) -> str:
    if "Math-Proofs" in dataset or "Proofs" in dataset:
        return "proof"
    if "Science" in dataset:
        return "science"
    if "Agentic" in dataset:
        return "agentic"
    if "Instruction-Following" in dataset:
        return "chat_if"
    return "math"


MATH_V2_SYSTEM_TEMPLATE = (
    "You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\n\n"
    "Reasoning: {difficulty}\n\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every message."
)

MATH_V2_DEVELOPER = (
    "# Instructions\n\n"
    "You will solve the problem and return the final answer in \\boxed{}. "
    "Do not guess the answer, unless specifically given permission to."
)

PROOFS_SYSTEM = (
    "You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\n\n"
    "# Task: Convert the user's math statement into Lean 4."
)

PROOFS_DEVELOPER = (
    "# Instructions\n\n"
    "Convert the math problem into a Lean 4 theorem statement. "
    "Return Lean code only."
)


def _messages_to_harmony_text(
    *,
    source_messages: list[dict[str, Any]],
    system_prompt: str | None,
    developer_prompt: str | None,
) -> str:
    out: list[dict[str, Any]] = []
    if system_prompt is not None:
        out.append({"role": "system", "content": system_prompt})
    if developer_prompt is not None:
        out.append({"role": "developer", "content": developer_prompt})

    for m in source_messages:
        role = m.get("role")
        if not isinstance(role, str) or not role:
            continue

        if role == "assistant":
            reasoning = m.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                out.append({"role": "assistant", "channel": "analysis", "content": reasoning})

            tool_calls = m.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                # Nemotron tool calling: tool_calls is a list of OpenAI-style tool call objects.
                # We render python code execution as "assistant to=python" messages terminated by <|call|>,
                # followed by tool output messages with role=<tool_name> channel=commentary.
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    if not isinstance(fn, dict):
                        continue
                    tool_name = fn.get("name")
                    arguments = fn.get("arguments")

                    code = None
                    if isinstance(arguments, str) and arguments.strip():
                        try:
                            arg_obj = json.loads(arguments)
                            if isinstance(arg_obj, dict) and isinstance(arg_obj.get("code"), str):
                                code = arg_obj["code"]
                            else:
                                code = json.dumps(arg_obj, ensure_ascii=False, sort_keys=True)
                        except Exception:
                            code = arguments

                    if not code:
                        continue

                    # Match existing Harmony tools formatting.
                    tool_alias = (
                        "python"
                        if tool_name in {"stateful_python_code_exec", "python_code_exec", "python"}
                        else tool_name
                    )
                    assistant_role = f"assistant to={tool_alias}" if tool_alias else "assistant"
                    out.append(
                        {
                            "role": assistant_role,
                            "channel": "analysis",
                            "content": code,
                            "end_tag": "<|call|>",
                        }
                    )
            else:
                content = m.get("content", "")
                if content is None:
                    content = ""
                if isinstance(content, str) and content.strip():
                    out.append({"role": "assistant", "channel": "final", "content": content})
            continue

        if role == "tool":
            tool_name = m.get("name")
            if isinstance(tool_name, str) and tool_name.strip():
                role = tool_name.strip()
            out.append({"role": role, "channel": "commentary", "content": m.get("content", "")})
            continue

        out.append({"role": role, "content": m.get("content", "")})

    return render_harmony(out, add_return_tag=True)


def _extract_common_meta(rec: dict[str, Any], *, domain: str, split: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["meta_domain"] = domain
    out["meta_difficulty_bin"] = _difficulty_from_split(split)
    out["source_uuid"] = rec.get("uuid") if isinstance(rec.get("uuid"), str) else None
    out["source_license"] = rec.get("license") if isinstance(rec.get("license"), str) else None
    used_in = rec.get("used_in")
    if isinstance(used_in, list) and all(isinstance(x, str) for x in used_in):
        out["source_used_in"] = used_in
    else:
        out["source_used_in"] = None

    # Optional fields present in some Nemotron datasets.
    if isinstance(rec.get("capability_target"), str):
        out["source_capability_target"] = rec.get("capability_target")
    if isinstance(rec.get("reasoning"), str):
        out["source_reasoning"] = rec.get("reasoning")
    return out


def _extract_nemotron_math_v2_meta(rec: dict[str, Any], *, split: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["meta_domain"] = "math"
    out["meta_difficulty_bin"] = _difficulty_from_split(split)
    out["source_uuid"] = rec.get("uuid") if isinstance(rec.get("uuid"), str) else None
    out["source_license"] = rec.get("license") if isinstance(rec.get("license"), str) else None
    out["source_data_source"] = rec.get("data_source") if isinstance(rec.get("data_source"), str) else None
    out["source_changed_answer_to_majority"] = (
        rec.get("changed_answer_to_majority")
        if isinstance(rec.get("changed_answer_to_majority"), bool)
        else None
    )

    meta = rec.get("metadata")
    if not isinstance(meta, dict):
        return out

    eval_rows: list[tuple[str, float, int | None, int | None]] = []
    for k, v in meta.items():
        if not isinstance(k, str) or not isinstance(v, dict):
            continue
        acc = _safe_float(v.get("accuracy"))
        if acc is None:
            continue
        count = _safe_int(v.get("count"))
        passed = _safe_int(v.get("pass"))
        eval_rows.append((k, acc, count, passed))

    if not eval_rows:
        return out

    best = max(eval_rows, key=lambda t: t[1])
    out["meta_correctness"] = best[1]
    out["meta_eval_best_setting"] = best[0]
    out["meta_pass_k"] = best[2]
    out["meta_correct_count"] = best[3]

    def _best(prefix: str) -> float | None:
        vals = [acc for k, acc, _, _ in eval_rows if k.startswith(prefix)]
        return max(vals) if vals else None

    out["meta_correctness_low"] = _best("reason_low")
    out["meta_correctness_medium"] = _best("reason_medium")
    out["meta_correctness_high"] = _best("reason_high")
    out["meta_correctness_high_with_tool"] = _best("reason_high_with_tool")
    out["meta_correctness_high_no_tool"] = _best("reason_high_no_tool")

    return out


@dataclass
class ShardWriter:
    out_dir: Path
    dataset: str
    split: str
    rows_per_shard: int
    compression: str
    hf_layout: bool
    name_prefix: str | None = None
    batch_rows: int = 8192

    shard_index: int = 0
    rows_in_shard: int = 0
    buffered_rows: int = 0
    buffers: dict[str, list[Any]] = field(
        default_factory=lambda: {f.name: [] for f in NORMALIZED_SCHEMA}
    )
    total_rows_written: int = 0
    _writer: pq.ParquetWriter | None = None
    _current_path: Path | None = None

    def _path_for_shard(self) -> Path:
        if self.hf_layout:
            if self.name_prefix:
                return (
                    self.out_dir
                    / "data"
                    / self.split
                    / f"part-{self.name_prefix}-{self.shard_index:05d}.parquet"
                )
            return self.out_dir / "data" / self.split / f"part-{self.shard_index:05d}.parquet"
        safe_dataset = self.dataset.replace("/", "__")
        prefix = f"{self.name_prefix}__" if self.name_prefix else ""
        name = f"{safe_dataset}__{self.split}__{prefix}{self.shard_index:05d}.parquet"
        return self.out_dir / name

    def _open_writer_if_needed(self) -> None:
        if self._writer is not None:
            return
        path = self._path_for_shard()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._current_path = path
        self._writer = pq.ParquetWriter(str(path), NORMALIZED_SCHEMA, compression=self.compression)

    def _write_buffer(self) -> None:
        if not self.buffered_rows:
            return
        self._open_writer_if_needed()
        arrays = [pa.array(self.buffers[f.name], type=f.type) for f in NORMALIZED_SCHEMA]
        table = pa.Table.from_arrays(arrays, schema=NORMALIZED_SCHEMA)
        assert self._writer is not None
        self._writer.write_table(table)
        self.total_rows_written += self.buffered_rows
        self.buffers = {f.name: [] for f in NORMALIZED_SCHEMA}
        self.buffered_rows = 0

    def _close_shard(self) -> None:
        if not self.rows_in_shard and not self.buffered_rows and self._writer is None:
            return
        self._write_buffer()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            self._current_path = None
        self.shard_index += 1
        self.rows_in_shard = 0

    def add(self, row: dict[str, Any]) -> None:
        for f in NORMALIZED_SCHEMA:
            self.buffers[f.name].append(row.get(f.name))
        self.rows_in_shard += 1
        self.buffered_rows += 1
        if self.buffered_rows >= self.batch_rows:
            self._write_buffer()
        if self.rows_in_shard >= self.rows_per_shard:
            self._close_shard()

    def flush(self) -> None:
        if not self.rows_in_shard and not self.buffered_rows:
            return
        self._close_shard()


@dataclass(frozen=True)
class JsonlSegmentTask:
    dataset: str
    split: str
    domain: str
    path_in_repo: str
    local_path: str
    start: int
    end: int
    segment_index: int
    out_dir: str
    rows_per_shard: int
    compression: str
    hf_layout: bool
    system_prompt: str | None
    developer_prompt: str | None
    drop_invalid_harmony: bool
    drop_empty_completion: bool


def _estimate_avg_line_bytes(local_path: str, *, sample_lines: int = 4096) -> int:
    total = 0
    n = 0
    with open(local_path, "rb") as f:
        for _ in range(sample_lines):
            line = f.readline()
            if not line:
                break
            total += len(line)
            n += 1
    if n <= 0:
        return 2048
    return max(1, total // n)


def _segment_offsets_for_jsonl(local_path: str, *, segments: int) -> list[tuple[int, int]]:
    size = os.path.getsize(local_path)
    if segments <= 1 or size <= 0:
        return [(0, size)]

    raw_bounds = [int(size * i / segments) for i in range(segments + 1)]
    starts: list[int] = [0]
    with open(local_path, "rb") as f:
        for b in raw_bounds[1:-1]:
            f.seek(b)
            f.readline()  # advance to next newline (line boundary)
            starts.append(f.tell())
    starts.append(size)

    out: list[tuple[int, int]] = []
    for i in range(len(starts) - 1):
        a = starts[i]
        b = starts[i + 1]
        if a < b:
            out.append((a, b))
    return out


def _normalize_jsonl_segment(task: JsonlSegmentTask) -> tuple[dict[str, Any], dict[str, Any]]:
    # Avoid PyArrow oversubscription in multi-process mode.
    try:
        pa.set_cpu_count(1)
    except Exception:
        pass

    writer = ShardWriter(
        out_dir=Path(task.out_dir),
        dataset=task.dataset,
        split=task.split,
        rows_per_shard=task.rows_per_shard,
        compression=task.compression,
        hf_layout=bool(task.hf_layout),
        name_prefix=f"seg{task.segment_index:05d}",
    )

    num_in = 0
    num_out = 0
    num_dropped = 0
    tools_catalog: dict[str, Any] = {}

    with open(task.local_path, "rb") as f:
        f.seek(task.start)
        while True:
            pos = f.tell()
            if pos >= task.end:
                break
            line = f.readline()
            if not line:
                break

            num_in += 1
            try:
                rec = json.loads(line)
            except Exception:
                num_dropped += 1
                continue
            if not isinstance(rec, dict):
                num_dropped += 1
                continue

            if task.dataset == "nvidia/Nemotron-Math-Proofs-v1":
                problem = rec.get("problem")
                formal_statement = rec.get("formal_statement")
                lean_header = rec.get("lean_header")
                if not isinstance(problem, str) or not isinstance(formal_statement, str):
                    num_dropped += 1
                    continue
                completion = formal_statement
                if isinstance(lean_header, str) and lean_header.strip():
                    completion = lean_header.rstrip() + "\n\n" + completion.lstrip()
                source_messages = [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": completion},
                ]
            else:
                source_messages = rec.get("messages") or []
                if not isinstance(source_messages, list) or not source_messages:
                    num_dropped += 1
                    continue

            text = _messages_to_harmony_text(
                source_messages=source_messages,
                system_prompt=task.system_prompt,
                developer_prompt=task.developer_prompt,
            )
            text_id = sha1_text_id(text)

            q = basic_quality_flags_from_text(text)
            if task.drop_invalid_harmony and not q["valid_harmony"]:
                num_dropped += 1
                continue
            if task.drop_empty_completion and not q["completion_nonempty"]:
                num_dropped += 1
                continue

            meta: dict[str, Any]
            if task.dataset == "nvidia/Nemotron-Math-v2":
                meta = _extract_nemotron_math_v2_meta(rec, split=task.split)
                meta.update(_extract_common_meta(rec, domain=task.domain, split=task.split))
            elif task.dataset == "nvidia/Nemotron-Math-Proofs-v1":
                meta = _extract_common_meta(rec, domain=task.domain, split=task.split)
                if isinstance(rec.get("source"), str):
                    meta["source_data_source"] = rec.get("source")
            else:
                meta = _extract_common_meta(rec, domain=task.domain, split=task.split)

            tools = rec.get("tools")
            if isinstance(tools, list) or isinstance(tools, dict):
                tools_id = _sha1_json(tools)
                meta["source_tools_id"] = tools_id
                meta["source_tools_count"] = len(tools) if isinstance(tools, list) else len(tools)
                if tools_id not in tools_catalog:
                    tools_catalog[tools_id] = tools
            else:
                meta["source_tools_id"] = None
                meta["source_tools_count"] = None

            row = {
                "id": text_id,
                "dataset": task.dataset,
                "split": task.split,
                "text": text,
                "loss_mode": "assistant_all",
                "quality_valid_harmony": bool(q["valid_harmony"]),
                "quality_completion_nonempty": bool(q["completion_nonempty"]),
                "quality_has_tool": bool(q["has_tool"]),
                "quality_valid_tool_schema": bool(q["valid_tool_schema"]),
                "stats_char_len": len(text),
            }
            row.update(meta)
            writer.add(row)
            num_out += 1

    writer.flush()
    stats = {
        "segment_index": task.segment_index,
        "num_in": num_in,
        "num_out": num_out,
        "num_dropped": num_dropped,
        "shards_written": writer.shard_index,
    }
    print(
        f"[seg done] split={task.split} seg={task.segment_index} in={num_in} out={num_out} dropped={num_dropped} shards={writer.shard_index}",
        flush=True,
    )
    return stats, tools_catalog


def main() -> None:
    ap = argparse.ArgumentParser(description="CPU-side normalization + metadata extraction to Parquet")
    ap.add_argument("--dataset", type=str, required=True, help="HF dataset repo_id")
    ap.add_argument("--splits", nargs="*", default=[], help="Optional: only these splits")
    ap.add_argument("--out_dir", type=str, default="cpu_out", help="Output directory")
    ap.add_argument("--rows_per_shard", type=int, default=100_000)
    ap.add_argument("--compression", type=str, default="zstd")
    ap.add_argument(
        "--hf_layout",
        action="store_true",
        help="Write shards under data/<split>/part-*.parquet (HF-friendly layout)",
    )
    ap.add_argument(
        "--write_readme",
        action="store_true",
        help="Write README.md with HF dataset card frontmatter (best with --hf_layout)",
    )
    ap.add_argument("--timeout_s", type=int, default=60)
    ap.add_argument("--max_records", type=int, default=0, help="Debug: cap records per split (0=all)")
    ap.add_argument("--dedup", action="store_true", help="Exact dedup by sha1(text)")
    ap.add_argument("--drop_invalid_harmony", action="store_true", help="Drop rows failing harmony parse")
    ap.add_argument("--drop_empty_completion", action="store_true", help="Drop rows with empty completion")
    ap.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Parallel workers for JSONL splits (1=off; 0=auto)",
    )
    ap.add_argument(
        "--min_segment_mb",
        type=int,
        default=32,
        help="When --num_workers>1, split JSONL files into ~this many MB per worker task (smaller => more parallelism)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_workers = int(args.num_workers)
    if num_workers == 0:
        num_workers = os.cpu_count() or 1
    num_workers = max(1, num_workers)
    if num_workers > 1:
        # With multiprocessing we want to avoid per-process thread oversubscription.
        pa.set_cpu_count(1)

    split_files = _list_data_files_by_split(args.dataset)
    if not split_files:
        raise SystemExit(f"no data files found for dataset={args.dataset}")

    splits = args.splits or sorted(split_files.keys())
    domain = _domain_from_dataset(args.dataset)

    manifest: dict[str, Any] = {
        "generated_at": _now(),
        "dataset": args.dataset,
        "domain": domain,
        "splits": {},
    }

    seen: set[str] = set()
    tools_catalog: dict[str, Any] = {}

    for split in splits:
        files = split_files.get(split) or []
        if not files:
            print(f"[skip] split={split}: no files")
            continue

        paths = [f["path"] for f in files]
        is_parquet = all(p.endswith(".parquet") for p in paths)
        is_jsonl = all(p.endswith(".jsonl") for p in paths)
        if not (is_parquet or is_jsonl):
            raise SystemExit(f"split={split}: mixed/unknown file types: {paths[:3]}")

        num_in = 0
        num_out = 0
        num_dropped = 0
        num_dupe = 0

        if is_parquet:
            writer = ShardWriter(
                out_dir=out_dir,
                dataset=args.dataset,
                split=split,
                rows_per_shard=args.rows_per_shard,
                compression=args.compression,
                hf_layout=bool(args.hf_layout),
            )
            # Assume already Harmony-rendered with a 'text' column.
            it: Iterable[str] = _iter_parquet_text_from_hub(args.dataset, paths_in_repo=paths)
            for text in it:
                num_in += 1
                if args.max_records and num_in > args.max_records:
                    break

                text_id = sha1_text_id(text)
                if args.dedup:
                    if text_id in seen:
                        num_dupe += 1
                        continue
                    seen.add(text_id)

                q = basic_quality_flags_from_text(text)
                if args.drop_invalid_harmony and not q["valid_harmony"]:
                    num_dropped += 1
                    continue
                if args.drop_empty_completion and not q["completion_nonempty"]:
                    num_dropped += 1
                    continue

                row = {
                    "id": text_id,
                    "dataset": args.dataset,
                    "split": split,
                    "meta_domain": domain,
                    "meta_difficulty_bin": _difficulty_from_split(split),
                    "text": text,
                    "loss_mode": "assistant_all",
                    "quality_valid_harmony": bool(q["valid_harmony"]),
                    "quality_completion_nonempty": bool(q["completion_nonempty"]),
                    "quality_has_tool": bool(q["has_tool"]),
                    "quality_valid_tool_schema": bool(q["valid_tool_schema"]),
                    "stats_char_len": len(text),
                }
                writer.add(row)
                num_out += 1

            writer.flush()
            shards_written = writer.shard_index

        else:
            # JSONL source with messages/metadata; render to Harmony and extract meta.
            path_in_repo = paths[0]

            if args.dataset == "nvidia/Nemotron-Math-v2":
                system_prompt = MATH_V2_SYSTEM_TEMPLATE.format(
                    difficulty=_difficulty_from_split(split) or "unknown"
                )
                developer_prompt = MATH_V2_DEVELOPER
            elif args.dataset == "nvidia/Nemotron-Math-Proofs-v1":
                system_prompt = PROOFS_SYSTEM
                developer_prompt = PROOFS_DEVELOPER
            else:
                system_prompt = None
                developer_prompt = None

            if num_workers > 1 and not args.max_records and not args.dedup:
                # Download locally for parallel segment processing.
                local_path = hf_hub_download(
                    repo_id=args.dataset, repo_type="dataset", filename=path_in_repo
                )
                size = os.path.getsize(local_path)
                min_seg_bytes = max(1, int(args.min_segment_mb)) * 1024 * 1024
                segments = min(num_workers, max(1, int(math.ceil(size / min_seg_bytes))))
                offsets = _segment_offsets_for_jsonl(local_path, segments=segments)
                print(
                    f"[*] split={split} local={local_path} size_gb={size/1e9:.2f} segments={len(offsets)}",
                    flush=True,
                )

                tasks = [
                    JsonlSegmentTask(
                        dataset=args.dataset,
                        split=split,
                        domain=domain,
                        path_in_repo=path_in_repo,
                        local_path=local_path,
                        start=a,
                        end=b,
                        segment_index=i,
                        out_dir=str(out_dir),
                        rows_per_shard=args.rows_per_shard,
                        compression=args.compression,
                        hf_layout=bool(args.hf_layout),
                        system_prompt=system_prompt,
                        developer_prompt=developer_prompt,
                        drop_invalid_harmony=bool(args.drop_invalid_harmony),
                        drop_empty_completion=bool(args.drop_empty_completion),
                    )
                    for i, (a, b) in enumerate(offsets)
                ]

                ctx = mp.get_context("fork")
                with ctx.Pool(processes=len(offsets), maxtasksperchild=1) as pool:
                    shards_written = 0
                    for seg_stats, seg_tools in pool.imap_unordered(
                        _normalize_jsonl_segment, tasks, chunksize=1
                    ):
                        num_in += int(seg_stats.get("num_in") or 0)
                        num_out += int(seg_stats.get("num_out") or 0)
                        num_dropped += int(seg_stats.get("num_dropped") or 0)
                        shards_written += int(seg_stats.get("shards_written") or 0)
                        for k, v in (seg_tools or {}).items():
                            if k not in tools_catalog:
                                tools_catalog[k] = v
                num_dupe = 0
            else:
                writer = ShardWriter(
                    out_dir=out_dir,
                    dataset=args.dataset,
                    split=split,
                    rows_per_shard=args.rows_per_shard,
                    compression=args.compression,
                    hf_layout=bool(args.hf_layout),
                )
                for rec in _iter_jsonl_from_hub(
                    args.dataset, path_in_repo=path_in_repo, timeout_s=args.timeout_s
                ):
                    num_in += 1
                    if args.max_records and num_in > args.max_records:
                        break

                    if args.dataset == "nvidia/Nemotron-Math-Proofs-v1":
                        problem = rec.get("problem")
                        formal_statement = rec.get("formal_statement")
                        lean_header = rec.get("lean_header")
                        if not isinstance(problem, str) or not isinstance(formal_statement, str):
                            num_dropped += 1
                            continue
                        completion = formal_statement
                        if isinstance(lean_header, str) and lean_header.strip():
                            completion = lean_header.rstrip() + "\n\n" + completion.lstrip()
                        source_messages = [
                            {"role": "user", "content": problem},
                            {"role": "assistant", "content": completion},
                        ]
                    else:
                        source_messages = rec.get("messages") or []
                        if not isinstance(source_messages, list) or not source_messages:
                            num_dropped += 1
                            continue

                    text = _messages_to_harmony_text(
                        source_messages=source_messages,
                        system_prompt=system_prompt,
                        developer_prompt=developer_prompt,
                    )
                    text_id = sha1_text_id(text)
                    if args.dedup:
                        if text_id in seen:
                            num_dupe += 1
                            continue
                        seen.add(text_id)

                    q = basic_quality_flags_from_text(text)
                    if args.drop_invalid_harmony and not q["valid_harmony"]:
                        num_dropped += 1
                        continue
                    if args.drop_empty_completion and not q["completion_nonempty"]:
                        num_dropped += 1
                        continue

                    meta: dict[str, Any]
                    if args.dataset == "nvidia/Nemotron-Math-v2":
                        meta = _extract_nemotron_math_v2_meta(rec, split=split)
                        meta.update(_extract_common_meta(rec, domain=domain, split=split))
                    elif args.dataset == "nvidia/Nemotron-Math-Proofs-v1":
                        meta = _extract_common_meta(rec, domain=domain, split=split)
                        if isinstance(rec.get("source"), str):
                            meta["source_data_source"] = rec.get("source")
                    else:
                        meta = _extract_common_meta(rec, domain=domain, split=split)

                    tools = rec.get("tools")
                    if isinstance(tools, list) or isinstance(tools, dict):
                        tools_id = _sha1_json(tools)
                        meta["source_tools_id"] = tools_id
                        meta["source_tools_count"] = (
                            len(tools) if isinstance(tools, list) else len(tools)
                        )
                        if tools_id not in tools_catalog:
                            tools_catalog[tools_id] = tools
                    else:
                        meta["source_tools_id"] = None
                        meta["source_tools_count"] = None

                    row = {
                        "id": text_id,
                        "dataset": args.dataset,
                        "split": split,
                        "text": text,
                        "loss_mode": "assistant_all",
                        "quality_valid_harmony": bool(q["valid_harmony"]),
                        "quality_completion_nonempty": bool(q["completion_nonempty"]),
                        "quality_has_tool": bool(q["has_tool"]),
                        "quality_valid_tool_schema": bool(q["valid_tool_schema"]),
                        "stats_char_len": len(text),
                    }
                    row.update(meta)
                    writer.add(row)
                    num_out += 1

                writer.flush()
                shards_written = writer.shard_index

        manifest["splits"][split] = {
            "input_files": paths,
            "num_in": num_in,
            "num_out": num_out,
            "num_dropped": num_dropped,
            "num_dupe": num_dupe,
            "shards_written": shards_written,
        }
        print(
            f"[done] {args.dataset} split={split} in={num_in} out={num_out} "
            f"dropped={num_dropped} dupe={num_dupe} shards={shards_written}"
        )

    manifest_path = out_dir / f"{args.dataset.replace('/', '__')}__manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ok] wrote {manifest_path}")

    if tools_catalog:
        tools_path = out_dir / f"{args.dataset.replace('/', '__')}__tools_catalog.json"
        tools_path.write_text(
            json.dumps(tools_catalog, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[ok] wrote {tools_path} (unique_tool_sets={len(tools_catalog)})")

    if args.write_readme and args.hf_layout:
        readme_path = out_dir / "README.md"
        split_to_glob = {s: f"data/{s}/*.parquet" for s in manifest["splits"].keys()}
        fm = {
            "language": ["en"],
            "configs": [
                {
                    "config_name": "default",
                    "data_files": [
                        {"split": split, "path": path}
                        for split, path in split_to_glob.items()
                    ],
                }
            ],
            "tags": ["harmony", "text", "filtering", "nll-scoring-ready"],
        }
        fm_text = yaml.safe_dump(fm, sort_keys=False).strip()
        body = (
            f"---\n{fm_text}\n---\n\n"
            f"# Normalized dataset: `{args.dataset}`\n\n"
            "This repo contains CPU-normalized Parquet shards with:\n"
            "- `text` (Harmony formatted)\n"
            "- `loss_mode` (how to mask completion-only loss)\n"
            "- `meta_*` and `quality_*` fields for filtering\n\n"
            f"Generated by `cpu_normalize_dataset.py` on {_now()}.\n"
        )
        readme_path.write_text(body, encoding="utf-8")
        print(f"[ok] wrote {readme_path}")


if __name__ == "__main__":
    main()
