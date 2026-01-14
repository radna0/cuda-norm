#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def _render_html(payload: dict[str, Any], *, title: str) -> str:
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    title_json = json.dumps(title, ensure_ascii=False)
    title_html = html.escape(title, quote=True)
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
      background: var(--bg);
      color: var(--fg);
      margin: 0;
    }}
    #topbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      background: var(--bg);
      z-index: 10;
    }}
    #plot {{
      width: 100%;
      height: calc(100vh - 64px);
      background: var(--bg);
    }}
    select {{
      padding: 4px 8px;
      background: var(--control_bg);
      color: var(--fg);
      border: 1px solid var(--border);
      border-radius: 6px;
    }}
    label {{ user-select: none; }}
    input[type="checkbox"] {{ accent-color: #38bdf8; }}
    .muted {{ color: var(--muted); font-size: 12px; }}
  </style>
</head>
<body>
  <div id="topbar">
    <div>
      <div><strong>{title_html}</strong></div>
      <div class="muted">Density heatmap (full 2M) from PCA-3 bins; toggle group/plane/log; dark mode affects canvas.</div>
    </div>
    <div>
      Group:
      <select id="group"></select>
    </div>
    <div>
      Plane:
      <select id="plane">
        <option value="xy">PCA1 vs PCA2</option>
        <option value="xz">PCA1 vs PCA3</option>
        <option value="yz">PCA2 vs PCA3</option>
      </select>
    </div>
    <div>
      <label><input id="logScale" type="checkbox" checked /> log1p</label>
      <label style="margin-left:10px"><input id="darkMode" type="checkbox" checked /> dark</label>
    </div>
  </div>
  <div id="plot"></div>
  <script>
  const payload = {data_json};
  const groups = payload.groups;
  const grid = payload.grid2d;
  const ranges = payload.ranges; // {{x:[lo,hi], y:[lo,hi], z:[lo,hi]}}
  const bins = payload.bins; // {{xy:{{groupKey:{{bx:[],by:[],c:[]}}}}, ...}}

  function setTheme(dark) {{
    document.documentElement.dataset.theme = dark ? "dark" : "light";
  }}

  function linspace(lo, hi, n) {{
    const out = new Array(n);
    const step = (hi - lo) / n;
    for (let i = 0; i < n; i++) out[i] = lo + (i + 0.5) * step;
    return out;
  }}

  function dense2d(points, grid) {{
    // returns 2D array [y][x]
    const z = Array.from({{length: grid}}, () => new Array(grid).fill(0));
    const bx = points.bx, by = points.by, c = points.c;
    for (let i = 0; i < c.length; i++) {{
      const x = bx[i], y = by[i];
      z[y][x] = c[i];
    }}
    return z;
  }}

  function render() {{
    const plane = document.getElementById("plane").value;
    const groupKey = document.getElementById("group").value;
    const dark = document.getElementById("darkMode").checked;
    const logScale = document.getElementById("logScale").checked;
    setTheme(dark);

    const bg = dark ? "#0b1020" : "#ffffff";
    const fg = dark ? "#e2e8f0" : "#0f172a";
    const gridColor = dark ? "rgba(148,163,184,0.22)" : "rgba(15,23,42,0.12)";

    const pts = (bins[plane] && bins[plane][groupKey]) ? bins[plane][groupKey] : {{bx:[],by:[],c:[]}};
    let z = dense2d(pts, grid);
    if (logScale) {{
      for (let y = 0; y < grid; y++) {{
        for (let x = 0; x < grid; x++) {{
          z[y][x] = Math.log1p(z[y][x]);
        }}
      }}
    }}

    let xAxisTitle = "PCA-1";
    let yAxisTitle = "PCA-2";
    let xr = ranges.x, yr = ranges.y;
    if (plane === "xz") {{ yAxisTitle = "PCA-3"; yr = ranges.z; }}
    if (plane === "yz") {{ xAxisTitle = "PCA-2"; yAxisTitle = "PCA-3"; xr = ranges.y; yr = ranges.z; }}
    const xs = linspace(xr[0], xr[1], grid);
    const ys = linspace(yr[0], yr[1], grid);

    const trace = {{
      type: "heatmap",
      x: xs,
      y: ys,
      z: z,
      colorscale: dark ? "Turbo" : "Viridis",
      hovertemplate: `x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>density=%{{z:.3f}}<extra></extra>`,
    }};

    const layout = {{
      title: {title_json} + ` — ${{groupKey}} — ${{plane}}`,
      paper_bgcolor: bg,
      plot_bgcolor: bg,
      font: {{ color: fg }},
      xaxis: {{ title: xAxisTitle, gridcolor: gridColor, zerolinecolor: gridColor, color: fg }},
      yaxis: {{ title: yAxisTitle, gridcolor: gridColor, zerolinecolor: gridColor, color: fg, scaleanchor: "x", scaleratio: 1 }},
      margin: {{ l: 60, r: 10, t: 50, b: 55 }},
    }};

    Plotly.react("plot", [trace], layout, {{displayModeBar: true, scrollZoom: true, responsive: true, displaylogo: false}});
  }}

  // Populate group dropdown
  const sel = document.getElementById("group");
  for (const g of groups) {{
    const opt = document.createElement("option");
    opt.value = g;
    opt.textContent = g;
    sel.appendChild(opt);
  }}
  sel.value = groups[0] || "";

  document.getElementById("group").addEventListener("change", render);
  document.getElementById("plane").addEventListener("change", render);
  document.getElementById("logScale").addEventListener("change", render);
  document.getElementById("darkMode").addEventListener("change", render);
  render();
  </script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Render an interactive density heatmap HTML from density_bins.parquet.")
    ap.add_argument("--density_parquet", type=str, required=True)
    ap.add_argument("--density_manifest", type=str, required=True)
    ap.add_argument("--out_html", type=str, required=True)
    ap.add_argument("--title", type=str, default="PCA Density View")
    args = ap.parse_args()

    dens = pq.read_table(args.density_parquet)
    manifest = json.loads(Path(args.density_manifest).read_text(encoding="utf-8"))
    grid2d = int(manifest["grid_2d"])
    ranges = manifest["coord_range"]

    df = dens.to_pandas()
    df = df[df["kind"].isin(["xy", "xz", "yz"])].copy()

    # Build group keys and per-plane sparse bins.
    group_cols = manifest["group_cols"]
    if group_cols != ["dataset", "mix_group"]:
        # Still support arbitrary cols, but group key string may change.
        pass
    df["groupKey"] = df[group_cols].astype(str).agg("|".join, axis=1)
    groups = sorted(df["groupKey"].unique().tolist())

    bins: dict[str, dict[str, dict[str, list[int]]]] = {"xy": {}, "xz": {}, "yz": {}}
    for kind in ["xy", "xz", "yz"]:
        sub = df[df["kind"] == kind]
        for g in groups:
            sg = sub[sub["groupKey"] == g]
            bins[kind][g] = {
                "bx": sg["bx"].astype(int).tolist(),
                "by": sg["by"].astype(int).tolist(),
                "c": sg["count"].astype(int).tolist(),
            }

    payload = {
        "generated_at": _now(),
        "grid2d": grid2d,
        "ranges": {"x": ranges["x"], "y": ranges["y"], "z": ranges["z"]},
        "groups": groups,
        "bins": bins,
    }

    out_path = Path(args.out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render_html(payload, title=args.title), encoding="utf-8")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()

