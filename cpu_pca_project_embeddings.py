#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def _parse_bucket_edges(spec: str) -> list[int]:
    parts = [p.strip() for p in (spec or "").split(",") if p.strip()]
    if not parts or parts[0] != "0" or parts[-1].lower() != "inf":
        raise SystemExit("--len_bucket_edges must look like: 0,64,128,256,512,1024,inf")
    edges: list[int] = []
    for p in parts[:-1]:
        try:
            edges.append(int(p))
        except Exception as e:
            raise SystemExit(f"bad --len_bucket_edges part {p!r}: {e}") from e
    if edges != sorted(edges):
        raise SystemExit("--len_bucket_edges must be sorted ascending")
    if edges[0] != 0:
        raise SystemExit("--len_bucket_edges must start with 0")
    return edges


def _len_bucket_label(lo: int, hi: int | None) -> str:
    if hi is None:
        return f"{lo:04d}_inf"
    return f"{lo:04d}_{hi:04d}"


def _assign_len_bucket(tok: int, edges: list[int]) -> str:
    t = int(tok or 0)
    if t < 0:
        t = 0
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else None
        if hi is None:
            return _len_bucket_label(lo, None)
        if lo <= t < hi:
            return _len_bucket_label(lo, hi)
    return _len_bucket_label(edges[-1], None)


def _canonical_difficulty(x: Any) -> str:
    s = str(x or "").strip().lower()
    if s in {"low", "medium", "high"}:
        return s
    return "unknown"


def _embedding_array_to_numpy(arr: pa.Array) -> tuple[np.ndarray, int]:
    if not pa.types.is_fixed_size_list(arr.type):
        raise TypeError(f"expected FixedSizeListArray embedding, got {arr.type}")
    dim = int(arr.type.list_size)
    values = arr.values.to_numpy(zero_copy_only=False)
    if values.size % dim != 0:
        raise RuntimeError(f"embedding values size {values.size} not divisible by dim {dim}")
    return values.reshape(-1, dim), dim


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute a 2D PCA projection for a sample of embedding Parquet shards (map view)."
    )
    ap.add_argument("--in_dir", type=str, required=True, help="Directory containing embedding *.parquet")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--sample_per_file", type=int, default=25_000)
    ap.add_argument("--max_samples", type=int, default=200_000)
    ap.add_argument("--filter_mix_group", type=str, default="", help="Optional mix_group filter (e.g. reasoning)")
    ap.add_argument("--num_threads", type=int, default=0, help="0=auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--len_bucket_edges",
        type=str,
        default="0,64,128,256,512,1024,inf",
        help="Prompt-token bucket edges for map coloring/analysis.",
    )
    ap.add_argument(
        "--write_html",
        action="store_true",
        help="Also emit a lightweight Plotly HTML (loads plotly.js from CDN).",
    )
    ap.add_argument("--html_max_points", type=int, default=100_000)
    ap.add_argument("--html_name", type=str, default="map_view.html")
    ap.add_argument(
        "--pca_dims",
        type=int,
        default=2,
        help="PCA output dimensions for visualization (2 or 3).",
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    len_edges = _parse_bucket_edges(args.len_bucket_edges)

    parquet_files = sorted(in_dir.rglob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"no parquet files under {in_dir}")

    sample_per_file = int(args.sample_per_file)
    max_samples = int(args.max_samples)
    if sample_per_file <= 0 or max_samples <= 0:
        raise SystemExit("--sample_per_file and --max_samples must be > 0")

    rng = np.random.default_rng(int(args.seed))
    rng_html = np.random.default_rng(int(args.seed) + 999)

    want_cols = [
        "id",
        "embedding",
        "dataset",
        "split",
        "mix_group",
        "meta_domain",
        "meta_difficulty_bin",
        "meta_correctness",
        "prompt_tokens",
    ]

    rows: list[dict[str, Any]] = []
    emb_list: list[np.ndarray] = []
    total_read = 0

    for pf in parquet_files:
        if len(rows) >= max_samples:
            break
        parquet = pq.ParquetFile(pf)
        cols = set(parquet.schema_arrow.names)
        if "embedding" not in cols or "id" not in cols:
            continue
        read_cols = [c for c in want_cols if c in cols]
        # Read a reasonably small slice from each file; do not materialize the whole shard.
        # We read in batches to allow optional mix_group filtering without scanning everything.
        taken = 0
        for batch in parquet.iter_batches(columns=read_cols, batch_size=65_536):
            if taken >= sample_per_file or len(rows) >= max_samples:
                break
            tbl = pa.Table.from_batches([batch])
            if tbl.num_rows == 0:
                continue
            if args.filter_mix_group and "mix_group" in tbl.column_names:
                mg = [str(x or "") for x in tbl["mix_group"].to_pylist()]
                mask = [m == args.filter_mix_group for m in mg]
                tbl = tbl.filter(pa.array(mask))
                if tbl.num_rows == 0:
                    continue

            need = min(sample_per_file - taken, max_samples - len(rows), int(tbl.num_rows))
            if need <= 0:
                break
            # Random subset of rows within this batch for better coverage.
            if need < tbl.num_rows:
                idx = rng.choice(np.arange(tbl.num_rows), size=need, replace=False)
                tbl = tbl.take(pa.array(idx, type=pa.int32()))
            else:
                tbl = tbl.slice(0, need)

            emb_np, dim = _embedding_array_to_numpy(tbl["embedding"].chunk(0))
            emb_list.append(emb_np.astype(np.float32, copy=False))
            taken += int(tbl.num_rows)
            total_read += int(tbl.num_rows)

            cols_present = set(tbl.column_names)
            for i in range(tbl.num_rows):
                r: dict[str, Any] = {}
                for c in want_cols:
                    if c == "embedding":
                        continue
                    if c in cols_present:
                        v = tbl[c][i].as_py()
                        r[c] = v
                r["_source_file"] = str(pf)
                rows.append(r)

        if taken:
            print(f"[file] {pf} took={taken}", flush=True)

    if not rows:
        raise SystemExit("no rows sampled (check --filter_mix_group)")

    X = np.vstack(emb_list)
    if X.shape[0] != len(rows):
        raise RuntimeError(f"row/embedding mismatch: X={X.shape[0]} rows={len(rows)}")

    # PCA via Faiss (fast, multithreaded).
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise SystemExit(f"faiss not available: {e}") from e

    num_threads = int(args.num_threads) if args.num_threads else (os.cpu_count() or 1)
    faiss.omp_set_num_threads(int(num_threads))

    dim = int(X.shape[1])
    pca_dims = int(args.pca_dims)
    if pca_dims not in {2, 3}:
        raise SystemExit("--pca_dims must be 2 or 3")
    pca = faiss.PCAMatrix(dim, pca_dims)
    t0 = time.time()
    pca.train(X)
    Y = pca.apply_py(X)
    elapsed = time.time() - t0

    for r, vec in zip(rows, Y.tolist()):
        r["pca_x"] = float(vec[0])
        r["pca_y"] = float(vec[1])
        if pca_dims >= 3:
            r["pca_z"] = float(vec[2])
        r["difficulty_bin"] = _canonical_difficulty(r.get("meta_difficulty_bin"))
        r["len_bucket"] = _assign_len_bucket(int(r.get("prompt_tokens") or 0), len_edges)

    if pca_dims == 2:
        out_parquet = out_dir / "pca_2d_sample.parquet"
        manifest_path = out_dir / "pca_manifest.json"
    else:
        out_parquet = out_dir / f"pca_{pca_dims}d_sample.parquet"
        manifest_path = out_dir / f"pca_{pca_dims}d_manifest.json"
    pq.write_table(pa.Table.from_pylist(rows), out_parquet, compression="zstd")

    ev = getattr(pca, "eigenvalues", None)
    if ev is not None:
        ev_arr = faiss.vector_to_array(ev)
        ev_list = [float(x) for x in ev_arr.tolist()]
    else:
        ev_list = []
    ev_sum = float(sum(ev_list)) if ev_list else math.nan
    manifest = {
        "generated_at": _now(),
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "filter_mix_group": str(args.filter_mix_group),
        "sample_per_file": int(sample_per_file),
        "max_samples": int(max_samples),
        "rows": int(len(rows)),
        "dim": int(dim),
        "pca_dims": int(pca_dims),
        "len_bucket_edges": [0] + list(len_edges[1:]) + ["inf"],
        "faiss_threads": int(num_threads),
        "elapsed_s": float(elapsed),
        "eigenvalues": ev_list[:8],
        "eigenvalues_sum": ev_sum,
        "out_parquet": str(out_parquet),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[ok] wrote {out_parquet}", flush=True)
    print(f"[ok] wrote {manifest_path}", flush=True)

    if args.write_html:
        html_max = int(args.html_max_points)
        html_max = max(1000, min(html_max, len(rows)))
        idx = np.arange(len(rows))
        if html_max < len(rows):
            idx = rng_html.choice(idx, size=html_max, replace=False)
        idx = np.sort(idx)

        def col(name: str) -> list[Any]:
            return [rows[i].get(name) for i in idx.tolist()]

        payload = {
            "x": col("pca_x"),
            "y": col("pca_y"),
            "z": col("pca_z") if pca_dims >= 3 else [],
            "id": col("id"),
            "dataset": col("dataset"),
            "meta_domain": col("meta_domain"),
            "mix_group": col("mix_group"),
            "difficulty_bin": col("difficulty_bin"),
            "len_bucket": col("len_bucket"),
            "prompt_tokens": col("prompt_tokens"),
        }

        html_path = out_dir / args.html_name
        html_path.write_text(
            _render_plotly_html(
                payload,
                title=f"PCA Map View ({args.filter_mix_group or 'all'}) n={html_max}",
            ),
            encoding="utf-8",
        )
        print(f"[ok] wrote {html_path}", flush=True)


def _render_plotly_html(payload: dict[str, Any], *, title: str) -> str:
    # Minimal Plotly.js HTML (CDN) for 2D/3D scatter with a dropdown to color points by key.
    # This keeps dependencies out of the CPU environment.
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    title_json = json.dumps(title, ensure_ascii=False)
    title_html = html.escape(title, quote=True)
    is_3d = bool(payload.get("z")) and len(payload.get("z") or []) == len(payload.get("x") or [])
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title_html}</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    :root {{
      --bg: #ffffff;
      --fg: #0f172a;
      --muted: #334155;
      --border: #e2e8f0;
      --chip_bg: rgba(148, 163, 184, 0.12);
      --chip_border: rgba(148, 163, 184, 0.35);
      --control_bg: #ffffff;
    }}
    :root[data-theme="dark"] {{
      --bg: #0b1020;
      --fg: #e2e8f0;
      --muted: #cbd5e1;
      --border: rgba(148, 163, 184, 0.25);
      --chip_bg: rgba(148, 163, 184, 0.10);
      --chip_border: rgba(148, 163, 184, 0.22);
      --control_bg: rgba(15, 23, 42, 0.85);
    }}
    body {{
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      background: var(--bg);
      color: var(--fg);
    }}
    #topbar {{
      display: flex;
      gap: 12px;
      padding: 10px 12px;
      align-items: center;
      border-bottom: 1px solid var(--border);
      background: var(--bg);
    }}
    #legend {{ font-size: 12px; color: var(--muted); max-height: 120px; overflow: auto; }}
    #plot {{ width: 100%; height: calc(100vh - 64px); background: var(--bg); }}
    select {{
      padding: 4px 8px;
      background: var(--control_bg);
      color: var(--fg);
      border: 1px solid var(--border);
      border-radius: 6px;
    }}
    label {{ user-select: none; }}
    input[type="checkbox"] {{ accent-color: #38bdf8; }}
    .chip {{
      display: inline-block;
      padding: 2px 6px;
      margin: 2px 6px 2px 0;
      border-radius: 10px;
      border: 1px solid var(--chip_border);
      background: var(--chip_bg);
    }}
    .swatch {{ display: inline-block; width: 10px; height: 10px; margin-right: 6px; border-radius: 2px; vertical-align: middle; }}
  </style>
</head>
<body>
  <div id="topbar">
    <div><strong>{title_html}</strong></div>
    <div>
      Color by:
      <select id="colorBy">
        <option value="dataset">dataset</option>
        <option value="meta_domain">domain</option>
        <option value="mix_group">mix_group</option>
        <option value="difficulty_bin">difficulty</option>
        <option value="len_bucket">length bucket</option>
      </select>
    </div>
    <div>
      <label><input id="darkMode" type="checkbox" checked /> dark</label>
    </div>
    <div id="legend"></div>
  </div>
  <div id="plot"></div>
  <script>
  const payload = {data_json};
  const N = payload.x.length;
  const palette = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f",
    "#bcbd22","#17becf","#393b79","#637939","#8c6d31","#843c39","#7b4173","#3182bd",
    "#31a354","#756bb1","#636363","#e6550d"
  ];

  function unique(values) {{
    const seen = new Map();
    for (const v of values) {{
      const k = (v === null || v === undefined || v === "") ? "unknown" : String(v);
      if (!seen.has(k)) seen.set(k, true);
    }}
    return Array.from(seen.keys());
  }}

  function colorMap(categories) {{
    const m = new Map();
    for (let i = 0; i < categories.length; i++) {{
      m.set(categories[i], palette[i % palette.length]);
    }}
    return m;
  }}

  function buildColors(values, cmap) {{
    const out = new Array(values.length);
    for (let i = 0; i < values.length; i++) {{
      const k = (values[i] === null || values[i] === undefined || values[i] === "") ? "unknown" : String(values[i]);
      out[i] = cmap.get(k) || "#999";
    }}
    return out;
  }}

  function setLegend(cmap) {{
    const el = document.getElementById("legend");
    const keys = Array.from(cmap.keys());
    keys.sort();
    const parts = [];
    for (const k of keys) {{
      const c = cmap.get(k);
      parts.push(`<span class="chip"><span class="swatch" style="background:${{c}}"></span>${{k}}</span>`);
    }}
    el.innerHTML = parts.join("");
  }}

  function hoverText(i) {{
    const id = payload.id[i] || "";
    const ds = payload.dataset[i] || "";
    const dom = payload.meta_domain[i] || "";
    const mg = payload.mix_group[i] || "";
    const diff = payload.difficulty_bin[i] || "";
    const lb = payload.len_bucket[i] || "";
    const pt = payload.prompt_tokens[i] || 0;
    const z = (payload.z && payload.z.length === N) ? payload.z[i] : null;
    const zline = (z === null || z === undefined) ? "" : `<br>pca_z=${{z}}`;
    return `id=${{id}}<br>dataset=${{ds}}<br>domain=${{dom}}<br>mix_group=${{mg}}<br>difficulty=${{diff}}<br>len_bucket=${{lb}}<br>prompt_tokens=${{pt}}${{zline}}`;
  }}

  const hover = new Array(N);
  for (let i = 0; i < N; i++) hover[i] = hoverText(i);

  const is3d = {str(is_3d).lower()};
  function render(colorByKey, dark) {{
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    const values = payload[colorByKey];
    const cats = unique(values);
    const cmap = colorMap(cats);
    const colors = buildColors(values, cmap);
    setLegend(cmap);
    const template = dark ? "plotly_dark" : "plotly_white";
    const bg = dark ? "#0b1020" : "#ffffff";
    const fg = dark ? "#e2e8f0" : "#0f172a";
    const grid = dark ? "rgba(148,163,184,0.22)" : "rgba(15,23,42,0.12)";
    const marker = {{ size: is3d ? 2 : 3, opacity: 0.75, color: colors }};
    const trace = is3d ? {{
      type: "scatter3d",
      mode: "markers",
      x: payload.x,
      y: payload.y,
      z: payload.z,
      text: hover,
      hoverinfo: "text",
      marker,
    }} : {{
      type: "scattergl",
      mode: "markers",
      x: payload.x,
      y: payload.y,
      text: hover,
      hoverinfo: "text",
      marker,
    }};
    const layout = is3d ? {{
      title: {title_json},
      template,
      paper_bgcolor: bg,
      font: {{ color: fg }},
      scene: {{
        bgcolor: bg,
        xaxis: {{ title: "PCA-1", zeroline: false, showbackground: true, backgroundcolor: bg }},
        yaxis: {{ title: "PCA-2", zeroline: false, showbackground: true, backgroundcolor: bg }},
        zaxis: {{ title: "PCA-3", zeroline: false, showbackground: true, backgroundcolor: bg }},
        dragmode: "orbit",
      }},
      margin: {{ l: 0, r: 0, t: 40, b: 0 }},
    }} : {{
      title: {title_json},
      template,
      paper_bgcolor: bg,
      plot_bgcolor: bg,
      font: {{ color: fg }},
      xaxis: {{ title: "PCA-1", zeroline: false }},
      yaxis: {{ title: "PCA-2", zeroline: false }},
      margin: {{ l: 50, r: 10, t: 40, b: 50 }},
    }};
    // Force dark/light axis/grid styling to avoid “dark chrome, white canvas” mismatch.
    if (is3d) {{
      layout.scene.xaxis.gridcolor = grid;
      layout.scene.yaxis.gridcolor = grid;
      layout.scene.zaxis.gridcolor = grid;
      layout.scene.xaxis.color = fg;
      layout.scene.yaxis.color = fg;
      layout.scene.zaxis.color = fg;
      layout.scene.xaxis.zerolinecolor = grid;
      layout.scene.yaxis.zerolinecolor = grid;
      layout.scene.zaxis.zerolinecolor = grid;
    }} else {{
      layout.xaxis.gridcolor = grid;
      layout.yaxis.gridcolor = grid;
      layout.xaxis.color = fg;
      layout.yaxis.color = fg;
      layout.xaxis.zerolinecolor = grid;
      layout.yaxis.zerolinecolor = grid;
    }}
    Plotly.react("plot", [trace], layout, {{displayModeBar: true, scrollZoom: true, responsive: true, displaylogo: false}});
  }}

  const sel = document.getElementById("colorBy");
  const darkSel = document.getElementById("darkMode");
  function rerender() {{
    render(sel.value, darkSel.checked);
  }}
  sel.addEventListener("change", rerender);
  darkSel.addEventListener("change", rerender);
  rerender();
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
