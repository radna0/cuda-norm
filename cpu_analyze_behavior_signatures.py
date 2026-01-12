#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


_KV_RE = re.compile(r"^([a-zA-Z0-9_]+)=(.*)$")


def _parse_signature(sig: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not sig:
        return out
    for line in sig.splitlines():
        m = _KV_RE.match(line.strip())
        if not m:
            continue
        out[m.group(1)] = m.group(2)
    return out


def _quantiles(values: list[int], qs: list[float]) -> dict[str, float]:
    if not values:
        return {f"p{int(q*100):02d}": math.nan for q in qs}
    vals = sorted(values)
    n = len(vals)
    out: dict[str, float] = {}
    for q in qs:
        idx = int(round(q * (n - 1)))
        out[f"p{int(q*100):02d}"] = float(vals[idx])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze embed_behavior signature diversity/coverage")
    ap.add_argument("--in_dir", type=str, required=True, help="Directory with candidate *.parquet")
    ap.add_argument("--max_rows", type=int, default=200_000, help="Cap rows scanned (0=all)")
    ap.add_argument("--progress_every", type=int, default=50_000)
    ap.add_argument("--out", type=str, default="behavior_signature_report.md")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    files = sorted(in_dir.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {in_dir}")

    dataset = ds.dataset([str(p) for p in files], format="parquet")
    cols = set(dataset.schema.names)
    if "embed_text" not in cols:
        raise SystemExit("missing required column embed_text")

    read_cols = ["id", "embed_text"]
    for c in ("stats_embed_word_count", "dataset", "split", "meta_domain", "meta_difficulty_bin", "quality_has_tool"):
        if c in cols:
            read_cols.append(c)

    total = 0
    empty = 0
    unique_sig: set[str] = set()
    unique_tool_seq: set[str] = set()
    unique_tool_keysets: set[str] = set()
    tool_calls_pos = 0
    tool_keysets_pos = 0
    final_type = Counter()
    tool_keysets_counter = Counter()
    tool_roles_counter = Counter()
    tool_call_count_vals: list[int] = []
    word_count_vals: list[int] = []
    t0 = time.time()

    scanner = dataset.scanner(columns=read_cols, batch_size=8192, use_threads=True)
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        tbl = pa.Table.from_batches([batch])
        sigs = tbl["embed_text"].to_pylist()
        wc = tbl["stats_embed_word_count"].to_pylist() if "stats_embed_word_count" in tbl.column_names else None

        for i, sig in enumerate(sigs):
            if args.max_rows and total >= args.max_rows:
                break
            total += 1
            sig = sig if isinstance(sig, str) else ""
            if not sig.strip():
                empty += 1
                continue
            if len(unique_sig) < 500_000:
                unique_sig.add(sig)

            fields = _parse_signature(sig)
            tc = fields.get("tool_call_count") or "0"
            try:
                tc_i = int(tc)
            except Exception:
                tc_i = 0
            tool_call_count_vals.append(tc_i)
            if tc_i > 0:
                tool_calls_pos += 1
                seq = fields.get("tool_call_seq") or ""
                if seq and seq != "none" and len(unique_tool_seq) < 500_000:
                    unique_tool_seq.add(seq)

            ks = (fields.get("tool_output_keysets") or "").strip()
            if ks and ks != "none":
                tool_keysets_pos += 1
                if len(unique_tool_keysets) < 500_000:
                    unique_tool_keysets.add(ks)
                tool_keysets_counter[ks] += 1

            roles = (fields.get("tool_output_roles") or "").strip()
            if roles and roles != "none":
                tool_roles_counter[roles] += 1

            ft = fields.get("final_type") or "unknown"
            final_type[ft] += 1

            if wc is not None:
                try:
                    word_count_vals.append(int(wc[i] or 0))
                except Exception:
                    word_count_vals.append(0)

        if args.max_rows and total >= args.max_rows:
            break
        if args.progress_every and total and (total % args.progress_every == 0):
            dt = time.time() - t0
            print(f"[prog] rows={total} dt={dt:.1f}s", flush=True)

    dt = time.time() - t0
    report_lines: list[str] = []
    report_lines.append(f"# Behavior Signature Report\n")
    report_lines.append(f"- generated_at: `{_now()}`")
    report_lines.append(f"- in_dir: `{in_dir}`")
    report_lines.append(f"- rows_scanned: `{total}`")
    report_lines.append(f"- elapsed_s: `{dt:.1f}`\n")

    report_lines.append("## Key Metrics\n")
    report_lines.append(f"- empty_signatures: `{empty}` ({(empty/total*100.0) if total else 0.0:.2f}%)")
    report_lines.append(f"- unique_signatures: `{len(unique_sig)}` (cap 500k)")
    report_lines.append(f"- tool_call_count>0: `{tool_calls_pos}` ({(tool_calls_pos/total*100.0) if total else 0.0:.2f}%)")
    report_lines.append(f"- unique_tool_call_seq: `{len(unique_tool_seq)}` (cap 500k)\n")
    report_lines.append(f"- tool_output_keysets!=none: `{tool_keysets_pos}` ({(tool_keysets_pos/total*100.0) if total else 0.0:.2f}%)")
    report_lines.append(f"- unique_tool_output_keysets: `{len(unique_tool_keysets)}` (cap 500k)\n")

    q_tc = _quantiles(tool_call_count_vals, [0.50, 0.90, 0.99])
    report_lines.append("## Tool Calls (quantiles)\n")
    report_lines.append(f"- tool_call_count_p50: `{q_tc['p50']:.0f}`")
    report_lines.append(f"- tool_call_count_p90: `{q_tc['p90']:.0f}`")
    report_lines.append(f"- tool_call_count_p99: `{q_tc['p99']:.0f}`\n")

    if word_count_vals:
        q_wc = _quantiles(word_count_vals, [0.50, 0.90, 0.99])
        report_lines.append("## Signature Length (word count quantiles)\n")
        report_lines.append(f"- stats_embed_word_count_p50: `{q_wc['p50']:.0f}`")
        report_lines.append(f"- stats_embed_word_count_p90: `{q_wc['p90']:.0f}`")
        report_lines.append(f"- stats_embed_word_count_p99: `{q_wc['p99']:.0f}`\n")

    report_lines.append("## Final Answer Types\n")
    for k, v in final_type.most_common():
        report_lines.append(f"- {k}: `{v}` ({(v/total*100.0) if total else 0.0:.2f}%)")

    if tool_keysets_counter:
        report_lines.append("\n## Top Tool Output Keysets\n")
        for k, v in tool_keysets_counter.most_common(20):
            report_lines.append(f"- `{k}`: `{v}`")

    if tool_roles_counter:
        report_lines.append("\n## Top Tool Output Roles\n")
        for k, v in tool_roles_counter.most_common(20):
            report_lines.append(f"- `{k}`: `{v}`")

    Path(args.out).write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
