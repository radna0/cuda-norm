#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import requests
import yaml
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, list):
        return "list"
    return type(value).__name__


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _format_bytes(num_bytes: int | None) -> str:
    if not num_bytes:
        return "n/a"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(num_bytes)
    unit = 0
    while size >= 1024.0 and unit < len(units) - 1:
        size /= 1024.0
        unit += 1
    return f"{size:.2f} {units[unit]}"


def _quantile(sorted_values: list[int], q: float) -> int | None:
    if not sorted_values:
        return None
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    idx = int(round(q * (len(sorted_values) - 1)))
    return sorted_values[idx]


@dataclass
class NumSummary:
    values: list[int] = field(default_factory=list)

    def add(self, value: int) -> None:
        self.values.append(int(value))

    def summary(self) -> dict[str, float | int | None]:
        if not self.values:
            return {
                "n": 0,
                "min": None,
                "mean": None,
                "median": None,
                "p95": None,
                "max": None,
            }
        vals = sorted(self.values)
        mean = statistics.fmean(vals)
        median = statistics.median(vals)
        p95 = _quantile(vals, 0.95)
        return {
            "n": len(vals),
            "min": vals[0],
            "mean": float(mean),
            "median": float(median),
            "p95": float(p95) if p95 is not None else None,
            "max": vals[-1],
        }


def _render_num_summary(ns: NumSummary) -> str:
    s = ns.summary()
    if not s["n"]:
        return "n=0"
    return (
        f"n={s['n']}, mean={s['mean']:.1f}, median={s['median']:.1f}, "
        f"p95={s['p95']:.1f}, max={s['max']}"
    )


@dataclass
class FieldSummary:
    present: int = 0
    null: int = 0
    type_counts: dict[str, int] = field(default_factory=dict)
    str_len: NumSummary = field(default_factory=NumSummary)
    list_len: NumSummary = field(default_factory=NumSummary)
    dict_len: NumSummary = field(default_factory=NumSummary)

    def observe(self, value: Any) -> None:
        self.present += 1
        t = _type_name(value)
        self.type_counts[t] = self.type_counts.get(t, 0) + 1
        if value is None:
            self.null += 1
            return
        if isinstance(value, str):
            self.str_len.add(len(value))
        elif isinstance(value, list):
            self.list_len.add(len(value))
        elif isinstance(value, dict):
            self.dict_len.add(len(value))


@dataclass
class MessageSummary:
    present_records: int = 0
    null_records: int = 0
    non_list_records: int = 0

    role_counts: dict[str, int] = field(default_factory=dict)
    message_count: NumSummary = field(default_factory=NumSummary)

    key_counts: dict[str, int] = field(default_factory=dict)
    field_type_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    tool_calls_per_message: NumSummary = field(default_factory=NumSummary)
    content_chars_per_message: NumSummary = field(default_factory=NumSummary)
    reasoning_chars_per_message: NumSummary = field(default_factory=NumSummary)

    total_content_chars_per_record: NumSummary = field(default_factory=NumSummary)
    total_reasoning_chars_per_record: NumSummary = field(default_factory=NumSummary)
    messages_with_tool_calls_per_record: NumSummary = field(default_factory=NumSummary)

    def observe(self, value: Any) -> None:
        self.present_records += 1
        if value is None:
            self.null_records += 1
            return
        if not isinstance(value, list):
            self.non_list_records += 1
            return

        self.message_count.add(len(value))
        total_content_chars = 0
        total_reasoning_chars = 0
        msgs_with_tool_calls = 0

        for msg in value:
            if not isinstance(msg, dict):
                continue

            for k, v in msg.items():
                self.key_counts[k] = self.key_counts.get(k, 0) + 1
                t = _type_name(v)
                if k not in self.field_type_counts:
                    self.field_type_counts[k] = {}
                self.field_type_counts[k][t] = self.field_type_counts[k].get(t, 0) + 1

            role = msg.get("role")
            if isinstance(role, str):
                self.role_counts[role] = self.role_counts.get(role, 0) + 1

            content = msg.get("content")
            if isinstance(content, str):
                clen = len(content)
                self.content_chars_per_message.add(clen)
                total_content_chars += clen

            reasoning = msg.get("reasoning_content")
            if isinstance(reasoning, str):
                rlen = len(reasoning)
                self.reasoning_chars_per_message.add(rlen)
                total_reasoning_chars += rlen

            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list):
                self.tool_calls_per_message.add(len(tool_calls))
                if tool_calls:
                    msgs_with_tool_calls += 1
            elif tool_calls is not None:
                self.tool_calls_per_message.add(0)

        self.total_content_chars_per_record.add(total_content_chars)
        self.total_reasoning_chars_per_record.add(total_reasoning_chars)
        self.messages_with_tool_calls_per_record.add(msgs_with_tool_calls)


@dataclass
class SplitStats:
    dataset: str
    split: str
    path_in_repo: str
    url: str
    file_size_bytes: int | None

    sample_records: int = 0
    sampled_lines: int = 0
    sampled_bytes: int = 0
    json_parse_errors: int = 0

    fields: dict[str, FieldSummary] = field(default_factory=dict)
    messages: MessageSummary = field(default_factory=MessageSummary)
    license_values: dict[str, int] = field(default_factory=dict)

    def observe_record(self, record: dict[str, Any]) -> None:
        self.sample_records += 1
        for k, v in record.items():
            if k not in self.fields:
                self.fields[k] = FieldSummary()
            self.fields[k].observe(v)

        if "messages" in record:
            self.messages.observe(record.get("messages"))

        lic = record.get("license")
        if isinstance(lic, str) and lic:
            self.license_values[lic] = self.license_values.get(lic, 0) + 1

    @property
    def avg_line_bytes(self) -> float | None:
        if self.sampled_lines <= 0:
            return None
        return self.sampled_bytes / self.sampled_lines

    @property
    def estimated_rows(self) -> int | None:
        if not self.file_size_bytes:
            return None
        avg = self.avg_line_bytes
        if not avg:
            return None
        return int(self.file_size_bytes / avg)


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


def _iter_head_lines(*, url: str, max_lines: int, timeout_s: int) -> Iterable[bytes]:
    headers = {"Accept-Encoding": "identity"}
    with requests.get(url, stream=True, timeout=timeout_s, headers=headers) as resp:
        resp.raise_for_status()
        for i, line in enumerate(resp.iter_lines(decode_unicode=False)):
            if i >= max_lines:
                break
            if line:
                yield line


def _fetch_range_bytes(*, url: str, start: int, size: int, timeout_s: int) -> bytes | None:
    headers = {
        "Range": f"bytes={start}-{start + size - 1}",
        "Accept-Encoding": "identity",
    }
    with requests.get(url, timeout=timeout_s, headers=headers) as resp:
        if resp.status_code != 206:
            return None
        return resp.content


def _iter_lines_from_range_chunk(chunk: bytes, *, drop_first_partial: bool) -> Iterable[bytes]:
    if not chunk:
        return
    lines = chunk.splitlines()
    if drop_first_partial and lines:
        lines = lines[1:]
    if not chunk.endswith(b"\n") and lines:
        lines = lines[:-1]
    for line in lines:
        if line:
            yield line


def _observe_jsonl_lines(stats: SplitStats, lines: Iterable[bytes]) -> None:
    for line in lines:
        stats.sampled_lines += 1
        stats.sampled_bytes += len(line) + 1  # +1 newline (approx)
        try:
            rec = json.loads(line)
        except Exception:
            stats.json_parse_errors += 1
            continue
        if isinstance(rec, dict):
            stats.observe_record(rec)


def _sorted_type_counts(type_counts: dict[str, int]) -> str:
    if not type_counts:
        return ""
    return ", ".join(
        f"{k}:{v}"
        for k, v in sorted(type_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )


def _write_report(
    *,
    out_md: Path,
    out_csv: Path,
    dataset_order: list[str],
    dataset_meta: dict[str, dict[str, Any]],
    split_stats: dict[tuple[str, str], SplitStats],
    args: argparse.Namespace,
) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)

    md: list[str] = []
    md.append("# NVIDIA Nemotron datasets: structure + sample metrics\n")
    md.append("This report is **sample-based** (it does not scan full datasets).\n")
    md.append("## Sampling parameters\n")
    md.append(f"- Generated: `{time.strftime('%Y-%m-%d %H:%M:%S %z')}`")
    md.append(f"- head_lines_per_split: `{args.head_lines}`")
    md.append(
        f"- random_chunks_per_split: `{args.random_chunks}` (chunk_bytes=`{args.chunk_bytes}`)"
    )
    md.append(f"- max_lines_per_random_chunk: `{args.random_lines_per_chunk}`\n")

    for dataset in dataset_order:
        meta = dataset_meta.get(dataset, {})
        split_to_path: dict[str, str] = meta.get("split_to_path") or {}

        md.append(f"## `{dataset}`\n")
        md.append("### Metadata\n")
        md.append(f"- usedStorage: `{_format_bytes(meta.get('usedStorage'))}`")
        md.append(f"- downloads: `{meta.get('downloads', 'n/a')}`")
        md.append(f"- likes: `{meta.get('likes', 'n/a')}`")
        tags = meta.get("tags") or []
        if tags:
            md.append(f"- tags (first 12): `{', '.join(tags[:12])}`")
        md.append(f"- splits: `{', '.join(split_to_path.keys())}`\n")

        md.append("### Splits (file + sample metrics)\n")
        md.append(
            "| split | path | size | sample_records | parse_errors | avg_line_bytes | est_rows | top_fields | avg_msgs | p95_msgs | avg_msg_content_chars |\n"
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )

        for split, path in split_to_path.items():
            st = split_stats.get((dataset, split))
            if not st:
                continue
            msg_count = st.messages.message_count.summary()
            tot_content = st.messages.total_content_chars_per_record.summary()
            md.append(
                "| "
                + " | ".join(
                    [
                        split,
                        path,
                        _format_bytes(st.file_size_bytes),
                        str(st.sample_records),
                        str(st.json_parse_errors),
                        f"{st.avg_line_bytes:.1f}" if st.avg_line_bytes else "n/a",
                        f"{st.estimated_rows:,}" if st.estimated_rows else "n/a",
                        str(len(st.fields)),
                        f"{msg_count['mean']:.1f}"
                        if msg_count.get("mean") is not None
                        else "n/a",
                        f"{msg_count['p95']:.1f}"
                        if msg_count.get("p95") is not None
                        else "n/a",
                        f"{tot_content['mean']:.1f}"
                        if tot_content.get("mean") is not None
                        else "n/a",
                    ]
                )
                + " |"
            )

        for split in split_to_path.keys():
            st = split_stats.get((dataset, split))
            if not st:
                continue
            md.append(f"\n### Split: `{split}`\n")
            md.append(f"- file: `{st.path_in_repo}` ({_format_bytes(st.file_size_bytes)})")
            if st.avg_line_bytes:
                md.append(
                    f"- sample_records: `{st.sample_records}`, parse_errors: `{st.json_parse_errors}`, "
                    f"avg_line_bytes: `{st.avg_line_bytes:.1f}`"
                )
            else:
                md.append(
                    f"- sample_records: `{st.sample_records}`, parse_errors: `{st.json_parse_errors}`"
                )
            if st.license_values:
                lic = ", ".join(
                    f"{k}:{v}"
                    for k, v in sorted(st.license_values.items(), key=lambda kv: (-kv[1], kv[0]))
                )
                md.append(f"- observed `license` values (sample): `{lic}`")

            md.append("\n#### Top-level fields (types + length stats)\n")
            md.append(
                "| field | present% | types (count) | null% | str_len | list_len | dict_len |\n"
                "|---|---:|---|---:|---|---|---|\n"
            )
            for field_name, fs in sorted(
                st.fields.items(), key=lambda kv: (-kv[1].present, kv[0])
            ):
                present_pct = (
                    100.0 * fs.present / st.sample_records if st.sample_records else 0.0
                )
                null_pct = (100.0 * fs.null / fs.present) if fs.present else 0.0
                md.append(
                    "| "
                    + " | ".join(
                        [
                            field_name,
                            f"{present_pct:.1f}",
                            _sorted_type_counts(fs.type_counts),
                            f"{null_pct:.1f}",
                            _render_num_summary(fs.str_len),
                            _render_num_summary(fs.list_len),
                            _render_num_summary(fs.dict_len),
                        ]
                    )
                    + " |"
                )

            msg = st.messages
            if msg.present_records:
                md.append("\n#### `messages` field analysis\n")
                md.append(
                    f"- present_records: `{msg.present_records}`, null_records: `{msg.null_records}`, "
                    f"non_list_records: `{msg.non_list_records}`"
                )
                md.append(f"- messages_per_record: `{_render_num_summary(msg.message_count)}`")
                md.append(
                    f"- total_content_chars_per_record: `{_render_num_summary(msg.total_content_chars_per_record)}`"
                )
                md.append(
                    f"- total_reasoning_chars_per_record: `{_render_num_summary(msg.total_reasoning_chars_per_record)}`"
                )
                md.append(
                    f"- messages_with_tool_calls_per_record: `{_render_num_summary(msg.messages_with_tool_calls_per_record)}`"
                )
                if msg.role_counts:
                    top_roles = sorted(msg.role_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                    md.append(
                        f"- roles (top 12): `{', '.join([f'{r}:{c}' for r, c in top_roles[:12]])}`"
                    )

                md.append("\n| message_field | types (count) |\n|---|---|\n")
                for k in sorted(msg.field_type_counts.keys()):
                    md.append(f"| {k} | {_sorted_type_counts(msg.field_type_counts[k])} |")

    out_md.write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "split",
                "path_in_repo",
                "file_size_bytes",
                "file_size_human",
                "sample_records",
                "parse_errors",
                "avg_line_bytes",
                "estimated_rows",
                "top_level_fields",
                "avg_messages_per_record",
                "p95_messages_per_record",
                "avg_total_message_content_chars_per_record",
            ],
        )
        writer.writeheader()
        for dataset in dataset_order:
            meta = dataset_meta.get(dataset, {})
            split_to_path: dict[str, str] = meta.get("split_to_path") or {}
            for split in split_to_path.keys():
                st = split_stats.get((dataset, split))
                if not st:
                    continue
                msg_count = st.messages.message_count.summary()
                tot_content = st.messages.total_content_chars_per_record.summary()
                writer.writerow(
                    {
                        "dataset": dataset,
                        "split": split,
                        "path_in_repo": st.path_in_repo,
                        "file_size_bytes": st.file_size_bytes,
                        "file_size_human": _format_bytes(st.file_size_bytes),
                        "sample_records": st.sample_records,
                        "parse_errors": st.json_parse_errors,
                        "avg_line_bytes": f"{st.avg_line_bytes:.1f}"
                        if st.avg_line_bytes
                        else None,
                        "estimated_rows": st.estimated_rows,
                        "top_level_fields": len(st.fields),
                        "avg_messages_per_record": msg_count.get("mean"),
                        "p95_messages_per_record": msg_count.get("p95"),
                        "avg_total_message_content_chars_per_record": tot_content.get("mean"),
                    }
                )


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect NVIDIA Nemotron datasets (sample-based)")
    ap.add_argument("--out_dir", type=str, default="reports")
    ap.add_argument("--head_lines", type=int, default=2000)
    ap.add_argument("--random_chunks", type=int, default=3)
    ap.add_argument("--chunk_bytes", type=int, default=1_048_576)
    ap.add_argument("--random_lines_per_chunk", type=int, default=200)
    ap.add_argument("--timeout_s", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=[
            "nvidia/Nemotron-Math-v2",
            "nvidia/Nemotron-Math-Proofs-v1",
            "nvidia/Nemotron-Science-v1",
            "nvidia/Nemotron-Agentic-v1",
            "nvidia/Nemotron-Instruction-Following-Chat-v1",
        ],
    )
    args = ap.parse_args()

    random.seed(args.seed)
    api = HfApi()

    dataset_meta: dict[str, dict[str, Any]] = {}
    split_stats: dict[tuple[str, str], SplitStats] = {}

    for dataset in args.datasets:
        info = api.dataset_info(dataset)
        meta: dict[str, Any] = {
            "usedStorage": _safe_int(getattr(info, "usedStorage", None)),
            "downloads": getattr(info, "downloads", None),
            "likes": getattr(info, "likes", None),
            "tags": list(getattr(info, "tags", []) or []),
        }

        readme_path = hf_hub_download(
            repo_id=dataset, repo_type="dataset", filename="README.md"
        )
        readme_text = Path(readme_path).read_text(encoding="utf-8")
        fm = _parse_frontmatter(readme_text)
        split_to_path = _split_to_path_from_frontmatter(fm)

        repo_files = list(
            api.list_repo_tree(dataset, repo_type="dataset", path_in_repo="data")
        )
        size_by_path = {f.path: _safe_int(getattr(f, "size", None)) for f in repo_files}

        if not split_to_path:
            split_to_path = {
                Path(f.path).stem: f.path for f in repo_files if isinstance(f.path, str)
            }

        meta["split_to_path"] = split_to_path
        meta["size_by_path"] = size_by_path
        dataset_meta[dataset] = meta

        for split, path_in_repo in split_to_path.items():
            url = hf_hub_url(
                repo_id=dataset,
                repo_type="dataset",
                filename=path_in_repo,
            )
            st = SplitStats(
                dataset=dataset,
                split=split,
                path_in_repo=path_in_repo,
                url=url,
                file_size_bytes=size_by_path.get(path_in_repo),
            )

            _observe_jsonl_lines(
                st,
                _iter_head_lines(
                    url=url, max_lines=args.head_lines, timeout_s=args.timeout_s
                ),
            )

            if (
                args.random_chunks
                and st.file_size_bytes
                and st.file_size_bytes > args.chunk_bytes
                and args.chunk_bytes > 0
            ):
                for _ in range(args.random_chunks):
                    start = random.randint(0, st.file_size_bytes - args.chunk_bytes)
                    chunk = _fetch_range_bytes(
                        url=url,
                        start=start,
                        size=args.chunk_bytes,
                        timeout_s=args.timeout_s,
                    )
                    if not chunk:
                        break
                    lines = _iter_lines_from_range_chunk(chunk, drop_first_partial=True)
                    limited: list[bytes] = []
                    for i, line in enumerate(lines):
                        if i >= args.random_lines_per_chunk:
                            break
                        limited.append(line)
                    _observe_jsonl_lines(st, limited)

            split_stats[(dataset, split)] = st

    out_dir = Path(args.out_dir)
    out_md = out_dir / "nvidia_nemotron_datasets_report.md"
    out_csv = out_dir / "nvidia_nemotron_datasets_metrics.csv"
    _write_report(
        out_md=out_md,
        out_csv=out_csv,
        dataset_order=args.datasets,
        dataset_meta=dataset_meta,
        split_stats=split_stats,
        args=args,
    )
    print(f"[ok] wrote {out_md}")
    print(f"[ok] wrote {out_csv}")


if __name__ == "__main__":
    main()
